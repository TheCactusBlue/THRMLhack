import type { GameState } from "../types";

interface CellTooltipProps {
  gameState: GameState;
  row: number;
  col: number;
  visible: boolean;
  position: { x: number; y: number };
}

export function CellTooltip({
  gameState,
  row,
  col,
  visible,
  position,
}: CellTooltipProps) {
  if (!visible) return null;

  const idx = row * gameState.grid_size + col;
  const bias = gameState.biases[idx];
  const spin = gameState.last_board?.[row]?.[col];
  const confidence = gameState.spin_confidence?.[row]?.[col];
  const entrenchment = gameState.entrenchment?.[row]?.[col] ?? 0;

  // Calculate predicted flip probability (simplified)
  const flipProb = confidence ? (1 - confidence) * 100 : 0;

  // Get neighbor count
  const neighbors = [
    [row - 1, col],
    [row + 1, col],
    [row, col - 1],
    [row, col + 1],
  ].filter(
    ([r, c]) =>
      r >= 0 && r < gameState.grid_size && c >= 0 && c < gameState.grid_size
  );

  const getSpinLabel = (spin: number) => {
    if (spin > 0) return "+1 (Player A - Blue)";
    if (spin < 0) return "-1 (Player B - Red)";
    return "Not sampled yet";
  };

  const getBiasLabel = (bias: number) => {
    if (bias > 0) return `+${bias.toFixed(2)} (favors Player A)`;
    if (bias < 0) return `${bias.toFixed(2)} (favors Player B)`;
    return "0.00 (neutral)";
  };

  const getEntrenchmentLabel = (entrenchment: number) => {
    if (entrenchment === 0) return "Not controlled";
    if (entrenchment === 1) return "Contested (1 round)";
    if (entrenchment === 2) return "Controlled (2 rounds, +0.3 bias)";
    return `Entrenched (${entrenchment} rounds, +0.6 bias)`;
  };

  return (
    <div
      className="fixed z-50 bg-neutral-900 border-2 border-neutral-700 rounded-lg shadow-2xl p-3 text-xs pointer-events-none max-w-xs"
      style={{
        left: `${position.x + 10}px`,
        top: `${position.y + 10}px`,
      }}
    >
      <div className="font-bold text-gray-200 mb-2 pb-2 border-b border-neutral-700">
        Cell ({row}, {col})
      </div>

      <div className="space-y-1.5">
        {spin !== undefined && (
          <div>
            <span className="text-gray-400">Current spin:</span>{" "}
            <span className="text-white font-semibold">{getSpinLabel(spin)}</span>
          </div>
        )}

        <div>
          <span className="text-gray-400">Bias:</span>{" "}
          <span className="text-white font-semibold">{getBiasLabel(bias)}</span>
        </div>

        <div>
          <span className="text-gray-400">Neighbors:</span>{" "}
          <span className="text-white">{neighbors.length} edges</span>
        </div>

        {entrenchment > 0 && (
          <div>
            <span className="text-gray-400">Entrenchment:</span>{" "}
            <span className="text-yellow-400 font-semibold">
              {getEntrenchmentLabel(entrenchment)}
            </span>
          </div>
        )}

        {confidence !== undefined && (
          <>
            <div className="pt-1.5 border-t border-neutral-700">
              <span className="text-gray-400">Confidence:</span>{" "}
              <span className="text-emerald-400 font-semibold">
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>

            <div>
              <span className="text-gray-400">Predicted flip prob:</span>{" "}
              <span
                className={`font-semibold ${
                  flipProb > 30 ? "text-red-400" : "text-green-400"
                }`}
              >
                {flipProb.toFixed(0)}%
              </span>
            </div>
          </>
        )}
      </div>

      <div className="mt-2 pt-2 border-t border-neutral-700 text-gray-500 text-[0.7em]">
        {spin !== undefined
          ? "This cell's state after sampling"
          : "Click to apply bias"}
      </div>
    </div>
  );
}
