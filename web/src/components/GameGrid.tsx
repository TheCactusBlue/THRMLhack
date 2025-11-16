import type { GameState } from "../types";

interface GameGridProps {
  gameState: GameState;
  selectedCell: [number, number] | null;
  edgeMode: boolean;
  onCellClick: (row: number, col: number) => void;
  previewMode?: boolean;
  previewData?: any;
  isAnimating?: boolean;
}

export function GameGrid({
  gameState,
  selectedCell,
  onCellClick,
  previewMode = false,
  previewData = null,
  isAnimating = false,
}: GameGridProps) {
  const getCellColor = (row: number, col: number) => {
    // REDESIGN: Show preview probabilities when in preview mode
    if (previewMode && previewData?.probabilities) {
      const prob = previewData.probabilities[row][col]; // 0-1 probability of being Player A
      // Blend between red (0) and blue (1)
      const blueIntensity = Math.floor(prob * 255);
      const redIntensity = Math.floor((1 - prob) * 255);
      return `rgb(${redIntensity}, 0, ${blueIntensity})`;
    }

    if (gameState.last_board) {
      const spin = gameState.last_board[row][col];
      // REDESIGN: Use confidence to determine color intensity
      const confidence = gameState.spin_confidence?.[row]?.[col] ?? 1.0;
      if (spin > 0) return `rgba(59, 130, 246, ${0.4 + confidence * 0.6})`; // blue for +1
      if (spin < 0) return `rgba(239, 68, 68, ${0.4 + confidence * 0.6})`; // red for -1
    }

    // Color based on bias
    const idx = row * gameState.grid_size + col;
    const bias = gameState.biases[idx];
    if (bias > 0)
      return `rgba(59, 130, 246, ${Math.min(Math.abs(bias) / 2, 0.8)})`;
    if (bias < 0)
      return `rgba(239, 68, 68, ${Math.min(Math.abs(bias) / 2, 0.8)})`;
    return "#444";
  };

  const isSelected = (row: number, col: number) => {
    return (
      selectedCell !== null &&
      selectedCell[0] === row &&
      selectedCell[1] === col
    );
  };

  const getCellContent = (row: number, col: number) => {
    // Show preview probabilities in preview mode
    if (previewMode && previewData?.probabilities) {
      const prob = previewData.probabilities[row][col];
      return (
        <span className="text-white text-[0.7em] font-bold drop-shadow-md">
          {(prob * 100).toFixed(0)}%
        </span>
      );
    }

    // Show confidence after sampling
    if (gameState.spin_confidence) {
      return (
        <span className="text-white text-shadow-[0_2px_4px_rgba(0,0,0,0.5)] text-[0.85em] font-semibold">
          {(gameState.spin_confidence[row][col] * 100).toFixed(0)}
        </span>
      );
    }

    return null;
  };

  return (
    <div className="flex justify-center mb-3">
      <div
        className={`grid gap-[3px] p-3 bg-neutral-900 rounded-lg shadow-[0_4px_16px_rgba(0,0,0,0.4)] ${
          isAnimating ? "animate-pulse" : ""
        }`}
        style={{
          gridTemplateColumns: `repeat(${gameState.grid_size}, 55px)`,
        }}
      >
        {Array.from({ length: gameState.grid_size }).map((_, row) =>
          Array.from({ length: gameState.grid_size }).map((_, col) => (
            <div
              key={`${row}-${col}`}
              className={`w-[55px] h-[55px] flex items-center justify-center rounded-md cursor-pointer transition-all duration-300 border-2 border-transparent text-[1.3em] font-bold hover:scale-105 hover:border-gray-500 hover:shadow-[0_4px_12px_rgba(255,255,255,0.2)] ${
                isSelected(row, col)
                  ? "!border-amber-400 shadow-[0_0_20px_rgba(251,191,36,0.5)]"
                  : ""
              } ${
                previewMode ? "ring-2 ring-purple-500/50 animate-pulse" : ""
              } ${
                isAnimating ? "animate-[wiggle_0.5s_ease-in-out_infinite]" : ""
              } max-sm:w-[45px] max-sm:h-[45px] max-sm:text-[1.1em]`}
              style={{ backgroundColor: getCellColor(row, col) }}
              onClick={() => onCellClick(row, col)}
            >
              {getCellContent(row, col)}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
