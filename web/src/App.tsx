import { useState, useEffect } from "react";
import "./App.css";
import type { PlayerType } from "./types";
import { useGameAPI } from "./hooks/useGameAPI";
import { PlayerPanel } from "./components/PlayerPanel";
import { GameGrid } from "./components/GameGrid";
import { GameControls } from "./components/GameControls";
import { GameStats } from "./components/GameStats";

function App() {
  const {
    gameState,
    loading,
    message,
    createGame,
    updateBias,
    updateCoupling,
    runSampling,
    resetGame,
    toggleReady,
    nextRound,
  } = useGameAPI();

  const [selectedCell, setSelectedCell] = useState<[number, number] | null>(
    null
  );
  const [edgeMode, setEdgeMode] = useState(false);
  const [showConfidence, setShowConfidence] = useState(false);
  const [currentPlayer, setCurrentPlayer] = useState<PlayerType>("A");
  const [roundWinner, setRoundWinner] = useState<string | null>(null);

  useEffect(() => {
    createGame();
  }, [createGame]);

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

  const handleCellClick = (row: number, col: number) => {
    if (edgeMode) {
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
          // Ask for direction
          const direction = window.confirm(
            "Increase coupling? (OK = increase, Cancel = decrease)"
          )
            ? 1
            : -1;
          updateCoupling(selectedCell, [row, col], direction, currentPlayer);
        }
        setSelectedCell(null);
      }
    } else {
      // Bias mode
      const direction = window.confirm(
        `Adjust bias at (${row}, ${col})?\n\nOK = Push to +1 (blue)\nCancel = Push to -1 (red)`
      )
        ? 1
        : -1;
      updateBias(row, col, direction, currentPlayer);
    }
  };

  if (!gameState) {
    return (
      <div className="app">
        <h1>THRMLHack Energy Battle</h1>
        <p>Loading...</p>
      </div>
    );
  }

  const bothReady = gameState.player_a_ready && gameState.player_b_ready;

  return (
    <div className="app-container">
      <GameControls
        gameState={gameState}
        edgeMode={edgeMode}
        loading={loading}
        onSetEdgeMode={setEdgeMode}
        onSetSelectedCell={setSelectedCell}
        onResetGame={handleResetGame}
        onCreateGame={createGame}
      />

      {message && <div className="message">{message}</div>}

      {gameState.game_winner && (
        <div className="game-winner">
          🏆 Game Winner: Player {gameState.game_winner}!
        </div>
      )}
      {roundWinner && !gameState.game_winner && (
        <div className="round-winner">Last Round: Player {roundWinner} Won</div>
      )}

      <div className="game-layout">
        <div className="side-panel">
          <PlayerPanel
            player="A"
            gameState={gameState}
            currentPlayer={currentPlayer}
            bothReady={bothReady}
            onSwitchPlayer={setCurrentPlayer}
            onToggleReady={handleToggleReady}
          />
        </div>

        <div className="center-panel">
          <GameGrid
            gameState={gameState}
            selectedCell={selectedCell}
            edgeMode={edgeMode}
            onCellClick={handleCellClick}
          />

          <div className="action-bar">
            <button
              onClick={runSampling}
              disabled={loading || !bothReady}
              className="sample-btn"
            >
              {bothReady ? "⚡ Run Sampling" : "⏳ Waiting..."}
            </button>

            {gameState.last_board && (
              <button
                onClick={handleNextRound}
                disabled={loading}
                className="next-btn"
              >
                Next Round →
              </button>
            )}

            <button
              className={`view-btn ${!showConfidence ? "active" : ""}`}
              onClick={() => setShowConfidence(false)}
            >
              Spins
            </button>
            <button
              className={`view-btn ${showConfidence ? "active" : ""}`}
              onClick={() => setShowConfidence(true)}
              disabled={!gameState.spin_confidence}
            >
              Confidence
            </button>
          </div>

          <GameStats gameState={gameState} />
        </div>

        <div className="side-panel">
          <PlayerPanel
            player="B"
            gameState={gameState}
            currentPlayer={currentPlayer}
            bothReady={bothReady}
            onSwitchPlayer={setCurrentPlayer}
            onToggleReady={handleToggleReady}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
