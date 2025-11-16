import type { GameState } from "../types";

interface GameGridProps {
  gameState: GameState;
  selectedCell: [number, number] | null;
  edgeMode: boolean;
  onCellClick: (row: number, col: number) => void;
}

export function GameGrid({
  gameState,
  selectedCell,
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
    <div className="flex justify-center mb-3">
      <div
        className="grid gap-[3px] p-3 bg-neutral-900 rounded-lg shadow-[0_4px_16px_rgba(0,0,0,0.4)]"
        style={{
          gridTemplateColumns: `repeat(${gameState.grid_size}, 55px)`,
        }}
      >
        {Array.from({ length: gameState.grid_size }).map((_, row) =>
          Array.from({ length: gameState.grid_size }).map((_, col) => (
            <div
              key={`${row}-${col}`}
              className={`w-[55px] h-[55px] flex items-center justify-center rounded-md cursor-pointer transition-all duration-200 border-2 border-transparent text-[1.3em] font-bold hover:scale-105 hover:border-gray-500 hover:shadow-[0_4px_12px_rgba(255,255,255,0.2)] ${
                isSelected(row, col)
                  ? "!border-amber-400 shadow-[0_0_20px_rgba(251,191,36,0.5)]"
                  : ""
              } max-sm:w-[45px] max-sm:h-[45px] max-sm:text-[1.1em]`}
              style={{ backgroundColor: getCellColor(row, col) }}
              onClick={() => onCellClick(row, col)}
            >
              {gameState.spin_confidence && (
                <span className="text-white text-shadow-[0_2px_4px_rgba(0,0,0,0.5)] text-[0.85em] font-semibold">
                  {(gameState.spin_confidence[row][col] * 100).toFixed(0)}
                </span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
