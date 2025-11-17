import { useState, useEffect } from "react";
import type { PlayerType, Action, CardType } from "../types";
import { useGameAPI } from "../hooks/useGameAPI";
import { PlayerPanel } from "../components/PlayerPanel";
import { GameGrid } from "../components/GameGrid";
import { GameControls } from "../components/GameControls";
import { GameStats } from "../components/GameStats";
import { ActionQueue } from "../components/ActionQueue";
import { CellTooltip } from "../components/CellTooltip";
import { GameLegend } from "../components/GameLegend";
import { CardHand } from "../components/CardHand";
import ClassSelector from "../components/ClassSelector";

export function Game() {
  const {
    gameState,
    loading,
    message,
    createGame,
    runSampling,
    resetGame,
    toggleReady,
    nextRound,
    previewSampling,
    batchActions,
    getAllCards,
    playCard,
    getAllClasses,
  } = useGameAPI();

  const [selectedCell, setSelectedCell] = useState<[number, number] | null>(
    null
  );
  const [edgeMode, setEdgeMode] = useState(false);
  const [shiftPressed, setShiftPressed] = useState(false);
  const [currentPlayer, setCurrentPlayer] = useState<PlayerType>("A");
  const [roundWinner, setRoundWinner] = useState<string | null>(null);
  const [previewMode, setPreviewMode] = useState(false);
  const [previewData, setPreviewData] = useState<any>(null);
  const [isAnimating, setIsAnimating] = useState(false);

  // PHASE 2: Action queue for undo functionality
  const [actionQueue, setActionQueue] = useState<Action[]>([]);
  const [showCouplings] = useState(true); // Always show by default

  // PHASE 3: Tooltip state
  const [hoveredCell] = useState<[number, number] | null>(null);
  const [tooltipPosition] = useState({ x: 0, y: 0 });

  // CARD SYSTEM: Selected card state
  const [selectedCard, setSelectedCard] = useState<CardType | null>(null);

  // CLASS SYSTEM: Track if game has been initialized
  const [gameInitialized, setGameInitialized] = useState(false);

  // Shift key handling for temporary mode toggle
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Shift" && !shiftPressed) {
        setShiftPressed(true);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === "Shift") {
        setShiftPressed(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [shiftPressed]);

  const handleClassSelection = async (
    playerAClass: string | undefined,
    playerBClass: string | undefined
  ) => {
    await createGame(playerAClass, playerBClass);
    setGameInitialized(true);
  };

  const handleResetGame = async () => {
    await resetGame();
    setSelectedCell(null);
    setEdgeMode(false);
    setCurrentPlayer("A");
    setRoundWinner(null);
  };

  const handleToggleReady = async () => {
    if (!gameState) return;
    const isReady =
      currentPlayer === "A"
        ? gameState.player_a_ready
        : gameState.player_b_ready;
    await toggleReady(currentPlayer, isReady);
  };

  const handleNextRound = async () => {
    const data = await nextRound();
    if (data) {
      setRoundWinner(data.round_winner);
    }
  };

  const handlePreview = async () => {
    setPreviewMode(true);
    const data = await previewSampling();
    if (data) {
      setPreviewData(data);
    }
  };

  // PHASE 2: Queue action instead of immediate execution
  const queueAction = (action: Action) => {
    setActionQueue((prev) => [...prev, action]);
  };

  const handleCellClick = async (row: number, col: number) => {
    // CARD SYSTEM: If a card is selected, play it at this location
    if (selectedCard) {
      await playCard(selectedCard, row, col, currentPlayer);
      setSelectedCard(null);
      return;
    }

    // Effective mode: toggle if shift is pressed
    const effectiveEdgeMode = shiftPressed ? !edgeMode : edgeMode;

    // REDESIGN: Remove confirm dialogs - use shift key for direction
    if (effectiveEdgeMode) {
      // Edge coupling mode
      if (selectedCell === null) {
        setSelectedCell([row, col]);
      } else {
        // Check if cells are neighbors
        const [r1, c1] = selectedCell;
        const isNeighbor =
          (Math.abs(r1 - row) === 1 && c1 === col) ||
          (Math.abs(c1 - col) === 1 && r1 === row);

        if (isNeighbor) {
          // Default to increase coupling (can add shift-click later for decrease)
          const direction = 1;

          // Queue the action instead of executing immediately
          queueAction({
            id: `coupling-${Date.now()}`,
            type: "coupling",
            description: `Strengthen edge (${r1},${c1})↔(${row},${col})`,
            params: {
              cell1: selectedCell,
              cell2: [row, col],
              direction,
              player: currentPlayer,
            },
          });
        }
        setSelectedCell(null);
      }
    } else {
      // Bias mode - default to player's direction
      // Player A = +1, Player B = -1
      const direction = currentPlayer === "A" ? 1 : -1;

      // Queue the action instead of executing immediately
      queueAction({
        id: `bias-${Date.now()}`,
        type: "bias",
        description: `Bias cell (${row},${col}) → ${
          currentPlayer === "A" ? "Blue" : "Red"
        }`,
        params: {
          row,
          col,
          direction,
          player: currentPlayer,
        },
      });
    }
  };

  // PHASE 2: Undo last action
  const handleUndo = () => {
    setActionQueue((prev) => prev.slice(0, -1));
  };

  // PHASE 2: Clear all queued actions
  const handleClearAll = () => {
    setActionQueue([]);
  };

  // PHASE 2: Commit all queued actions (using batch API)
  const handleCommit = async () => {
    if (actionQueue.length === 0) return;
    await batchActions(actionQueue);
    setActionQueue([]);
  };

  // NEW: Commit and mark ready in one action
  const handleCommitAndReady = async () => {
    if (!gameState) return;

    // Commit actions if any
    if (actionQueue.length > 0) {
      await batchActions(actionQueue);
      setActionQueue([]);
    }

    // Mark as ready
    const isReady =
      currentPlayer === "A"
        ? gameState.player_a_ready
        : gameState.player_b_ready;
    await toggleReady(currentPlayer, isReady);

    // Auto-switch to other player if they're not ready yet
    if (!isReady) {
      // We just marked current player as ready
      const otherPlayerReady =
        currentPlayer === "A"
          ? gameState.player_b_ready
          : gameState.player_a_ready;

      if (!otherPlayerReady) {
        // Switch to the other player so they can take their turn
        setCurrentPlayer(currentPlayer === "A" ? "B" : "A");
      }
    }
  };

  const handleRunSampling = async () => {
    setIsAnimating(true);
    setPreviewMode(false);
    setPreviewData(null);
    await runSampling();
    // Animation lasts 2 seconds
    setTimeout(() => setIsAnimating(false), 2000);
  };

  // Show class selector before game starts
  if (!gameInitialized) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900">
        <ClassSelector
          onSelectClasses={handleClassSelection}
          getAllClasses={getAllClasses}
        />
      </div>
    );
  }

  if (!gameState) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen">
        <h1 className="text-4xl font-bold mb-4">Thermodynamic Tactics</h1>
        <p className="text-gray-400">Loading...</p>
      </div>
    );
  }

  const bothReady = gameState.player_a_ready && gameState.player_b_ready;

  // Effective mode considering shift key
  const effectiveEdgeMode = shiftPressed ? !edgeMode : edgeMode;

  return (
    <div className="max-w-full h-screen m-0 p-2 flex flex-col overflow-hidden">
      <GameControls
        gameState={gameState}
        edgeMode={effectiveEdgeMode}
        shiftPressed={shiftPressed}
        loading={loading}
        onSetEdgeMode={setEdgeMode}
        onSetSelectedCell={setSelectedCell}
        onResetGame={handleResetGame}
        onCreateGame={createGame}
      />

      {message && (
        <div className="bg-blue-500 text-white px-4 py-2 rounded-md mb-2 text-sm text-center animate-[slideIn_0.3s_ease-out]">
          {message}
        </div>
      )}

      {gameState.game_winner && (
        <div className="text-xl font-bold text-emerald-500 px-4 py-2 mb-2 bg-gradient-to-r from-emerald-500/20 to-emerald-600/20 rounded-md border-2 border-emerald-500 text-center animate-pulse">
          🏆 Game Winner: Player {gameState.game_winner}!
        </div>
      )}
      {roundWinner && !gameState.game_winner && (
        <div className="text-base font-semibold text-amber-400 px-3 py-2 mb-2 bg-amber-400/10 rounded-md border-2 border-amber-400 text-center">
          Last Round: Player {roundWinner} Won
        </div>
      )}

      <div className="grid grid-cols-[200px_1fr_200px] gap-3 flex-1 overflow-hidden max-lg:grid-cols-1">
        <div className="bg-neutral-900 rounded-lg p-3 border-[3px] border-neutral-800 flex flex-col overflow-y-auto transition-all duration-300 max-lg:max-h-[200px]">
          <PlayerPanel
            player="A"
            gameState={gameState}
            currentPlayer={currentPlayer}
            bothReady={bothReady}
            onSwitchPlayer={setCurrentPlayer}
            onToggleReady={handleToggleReady}
          />

          {/* CARD SYSTEM: Player A's hand */}
          {currentPlayer === "A" && gameState.player_a_budget && (
            <div className="mt-4 pt-4 border-t border-neutral-700">
              <CardHand
                hand={gameState.player_a_budget.hand}
                playedCards={gameState.player_a_budget.played_cards}
                player="A"
                biasTokensAvailable={
                  gameState.player_a_budget.bias_tokens -
                  gameState.player_a_budget.bias_tokens_used
                }
                edgeTokensAvailable={
                  gameState.player_a_budget.edge_tokens -
                  gameState.player_a_budget.edge_tokens_used
                }
                onCardSelect={setSelectedCard}
                selectedCard={selectedCard}
                getAllCards={getAllCards}
              />
            </div>
          )}
        </div>

        <div className="flex flex-col items-center overflow-y-auto">
          {/* PHASE 2: Action Queue */}
          <ActionQueue
            actions={actionQueue}
            onUndo={handleUndo}
            onClearAll={handleClearAll}
            onCommit={handleCommit}
            onCommitAndReady={handleCommitAndReady}
            disabled={loading}
          />

          <GameGrid
            gameState={gameState}
            selectedCell={selectedCell}
            edgeMode={effectiveEdgeMode}
            onCellClick={handleCellClick}
            previewMode={previewMode}
            previewData={previewData}
            isAnimating={isAnimating}
            showCouplings={showCouplings}
          />

          {/* PHASE 3: Cell Tooltip */}
          {hoveredCell && gameState && (
            <CellTooltip
              gameState={gameState}
              row={hoveredCell[0]}
              col={hoveredCell[1]}
              visible={true}
              position={tooltipPosition}
            />
          )}

          {previewMode && previewData && (
            <div className="bg-purple-500/20 border-2 border-purple-500 rounded-md px-4 py-2 mb-2 text-sm">
              <strong>📊 Preview:</strong> Player A:{" "}
              {previewData.predicted_a_count.toFixed(1)} cells, Player B:{" "}
              {previewData.predicted_b_count.toFixed(1)} cells (Confidence:{" "}
              {(previewData.confidence * 100).toFixed(0)}%)
            </div>
          )}

          <div className="flex gap-2 justify-center flex-wrap mb-2 max-sm:flex-col max-sm:w-full">
            <button
              onClick={handlePreview}
              disabled={loading || !currentPlayer}
              className="px-4 py-2 text-sm font-semibold bg-purple-600 border-none rounded-md text-white cursor-pointer transition-all duration-200 hover:enabled:bg-purple-700 hover:enabled:shadow-[0_4px_12px_rgba(168,85,247,0.4)] disabled:opacity-50 disabled:cursor-not-allowed max-sm:w-full"
            >
              🔮 Preview Outcome
            </button>

            <button
              onClick={handleRunSampling}
              disabled={loading || !bothReady}
              className="px-6 py-3 text-base font-bold bg-gradient-to-br from-emerald-500 to-emerald-600 border-none rounded-lg text-white cursor-pointer transition-all duration-300 shadow-[0_4px_12px_rgba(16,185,129,0.3)] hover:enabled:-translate-y-0.5 hover:enabled:shadow-[0_6px_20px_rgba(16,185,129,0.4)] disabled:opacity-50 disabled:cursor-not-allowed max-sm:w-full"
            >
              {bothReady ? "⚡ Run Sampling" : "⏳ Waiting..."}
            </button>

            {gameState.last_board && (
              <button
                onClick={handleNextRound}
                disabled={loading}
                className="px-5 py-2.5 text-sm font-semibold bg-blue-500 border-none rounded-md text-white cursor-pointer transition-all duration-200 hover:enabled:bg-blue-600 hover:enabled:shadow-[0_4px_12px_rgba(59,130,246,0.3)] max-sm:w-full"
              >
                Next Round →
              </button>
            )}
          </div>

          <GameStats gameState={gameState} />

          <div className="mt-3">
            <GameLegend />
          </div>
        </div>

        <div className="bg-neutral-900 rounded-lg p-3 border-[3px] border-neutral-800 flex flex-col overflow-y-auto transition-all duration-300 max-lg:max-h-[200px]">
          <PlayerPanel
            player="B"
            gameState={gameState}
            currentPlayer={currentPlayer}
            bothReady={bothReady}
            onSwitchPlayer={setCurrentPlayer}
            onToggleReady={handleToggleReady}
          />

          {/* CARD SYSTEM: Player B's hand */}
          {currentPlayer === "B" && gameState.player_b_budget && (
            <div className="mt-4 pt-4 border-t border-neutral-700">
              <CardHand
                hand={gameState.player_b_budget.hand}
                playedCards={gameState.player_b_budget.played_cards}
                player="B"
                biasTokensAvailable={
                  gameState.player_b_budget.bias_tokens -
                  gameState.player_b_budget.bias_tokens_used
                }
                edgeTokensAvailable={
                  gameState.player_b_budget.edge_tokens -
                  gameState.player_b_budget.edge_tokens_used
                }
                onCardSelect={setSelectedCard}
                selectedCard={selectedCard}
                getAllCards={getAllCards}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
