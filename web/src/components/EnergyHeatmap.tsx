import type { GameState } from "../types";

interface EnergyHeatmapProps {
  gameState: GameState;
  visible: boolean;
}

export function EnergyHeatmap({ gameState, visible }: EnergyHeatmapProps) {
  if (!visible) return null;

  // Calculate local energy for each cell
  const calculateLocalEnergy = (row: number, col: number): number => {
    const idx = row * gameState.grid_size + col;
    const bias = gameState.biases[idx];
    const spin = gameState.last_board?.[row]?.[col] ?? 0;

    // Local energy contribution: -h_i * s_i
    let localEnergy = -bias * spin;

    // Add coupling contributions with neighbors
    // This is simplified - proper implementation would need edge mapping
    const neighbors = [
      [row - 1, col],
      [row + 1, col],
      [row, col - 1],
      [row, col + 1],
    ];

    neighbors.forEach(([nRow, nCol]) => {
      if (
        nRow >= 0 &&
        nRow < gameState.grid_size &&
        nCol >= 0 &&
        nCol < gameState.grid_size
      ) {
        const nSpin = gameState.last_board?.[nRow]?.[nCol] ?? 0;
        // Approximate coupling contribution
        const avgCoupling = 0.5; // Use average coupling
        localEnergy -= avgCoupling * spin * nSpin;
      }
    });

    return localEnergy;
  };

  // Find min/max energy for normalization
  let minEnergy = Infinity;
  let maxEnergy = -Infinity;
  const energies: number[][] = [];

  for (let row = 0; row < gameState.grid_size; row++) {
    energies[row] = [];
    for (let col = 0; col < gameState.grid_size; col++) {
      const energy = calculateLocalEnergy(row, col);
      energies[row][col] = energy;
      minEnergy = Math.min(minEnergy, energy);
      maxEnergy = Math.max(maxEnergy, energy);
    }
  }

  const getEnergyColor = (energy: number): string => {
    // Normalize to 0-1 range
    const normalized = (energy - minEnergy) / (maxEnergy - minEnergy || 1);

    // Low energy (stable) = dark blue, high energy (unstable) = bright red
    const red = Math.floor(normalized * 255);
    const blue = Math.floor((1 - normalized) * 255);

    return `rgb(${red}, 0, ${blue})`;
  };

  return (
    <div className="absolute inset-0 pointer-events-none">
      <div
        className="grid gap-[3px] p-3"
        style={{
          gridTemplateColumns: `repeat(${gameState.grid_size}, 55px)`,
        }}
      >
        {Array.from({ length: gameState.grid_size }).map((_, row) =>
          Array.from({ length: gameState.grid_size }).map((_, col) => {
            const energy = energies[row][col];
            const color = getEnergyColor(energy);

            return (
              <div
                key={`${row}-${col}`}
                className="w-[55px] h-[55px] rounded-md border border-white/20 flex items-center justify-center"
                style={{
                  backgroundColor: `${color}40`, // 25% opacity
                  boxShadow: `inset 0 0 10px ${color}80`,
                }}
                title={`Energy: ${energy.toFixed(2)}`}
              >
                <span className="text-[0.6em] text-white/80 font-mono">
                  {energy.toFixed(1)}
                </span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
