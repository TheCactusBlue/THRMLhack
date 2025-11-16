import type { GameState } from "../types";

interface GameGridProps {
  gameState: GameState;
  showConfidence: boolean;
  selectedCell: [number, number] | null;
  edgeMode: boolean;
  onCellClick: (row: number, col: number) => void;
}

export function GameGrid({
  gameState,
  showConfidence,
  selectedCell,
  edgeMode,
  onCellClick,
}: GameGridProps) {
  const getCellColor = (row: number, col: number) => {
    if (gameState.last_board) {
      const spin = gameState.last_board[row][col];
      if (spin > 0) return "#3b82f6"; // blue for +1
      if (spin < 0) return "#ef4444"; // red for -1
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

  return (
    <div className="grid-wrapper">
      <div
        className="grid"
        style={{
          gridTemplateColumns: `repeat(${gameState.grid_size}, 55px)`,
        }}
      >
        {Array.from({ length: gameState.grid_size }).map((_, row) =>
          Array.from({ length: gameState.grid_size }).map((_, col) => (
            <div
              key={`${row}-${col}`}
              className={`cell ${isSelected(row, col) ? "selected" : ""}`}
              style={{ backgroundColor: getCellColor(row, col) }}
              onClick={() => onCellClick(row, col)}
            >
              {gameState.last_board && !showConfidence && (
                <span className="spin-value">
                  {gameState.last_board[row][col] > 0 ? "+" : "-"}
                </span>
              )}
              {showConfidence && gameState.spin_confidence && (
                <span className="confidence-value">
                  {(gameState.spin_confidence[row][col] * 100).toFixed(0)}%
                </span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
