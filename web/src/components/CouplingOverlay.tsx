import type { GameState } from "../types";

interface CouplingOverlayProps {
  gameState: GameState;
  cellSize: number;
  gap: number;
  padding: number;
}

export function CouplingOverlay({
  gameState,
  cellSize = 55,
  gap = 3,
  padding = 12,
}: CouplingOverlayProps) {
  // Calculate edge index from grid coordinates
  const getEdgeIndex = (
    row1: number,
    col1: number,
    row2: number,
    col2: number
  ): number => {
    const size = gameState.grid_size;
    const i = row1 * size + col1;
    const j = row2 * size + col2;
    const [min, max] = i < j ? [i, j] : [j, i];

    // Grid edges: right edges first, then down edges
    // For a 5x5 grid: 4 right edges per row (5 rows) + 5 down edges per row (4 rows)
    const isRightEdge = max === min + 1 && Math.floor(min / size) === Math.floor(max / size);

    if (isRightEdge) {
      // Right edge: edge_idx = row * (size - 1) + col
      return Math.floor(min / size) * (size - 1) + (min % size);
    } else {
      // Down edge
      const rightEdgesCount = size * (size - 1);
      const downEdgeOffset = Math.floor(min / size) * size + (min % size);
      return rightEdgesCount + downEdgeOffset;
    }
  };

  // Get position of cell center
  const getCellCenter = (row: number, col: number) => {
    const x = padding + col * (cellSize + gap) + cellSize / 2;
    const y = padding + row * (cellSize + gap) + cellSize / 2;
    return { x, y };
  };

  // Render all edges
  const gridSize = gameState.grid_size;

  const renderEdges = () => {
    const edges = [];

    for (let row = 0; row < gridSize; row++) {
      for (let col = 0; col < gridSize; col++) {
        // Right edge
        if (col < gridSize - 1) {
          const edgeIdx = getEdgeIndex(row, col, row, col + 1);
          const coupling = gameState.couplings[edgeIdx] ?? 0.5;
          const from = getCellCenter(row, col);
          const to = getCellCenter(row, col + 1);

          edges.push(
            <g key={`h-${row}-${col}`}>
              <line
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke={coupling > 0 ? "#3b82f6" : "#ef4444"}
                strokeWidth={Math.abs(coupling) * 4 + 1}
                opacity={0.6}
                strokeLinecap="round"
              />
              <title>{`Coupling: ${coupling.toFixed(2)}`}</title>
            </g>
          );
        }

        // Down edge
        if (row < gridSize - 1) {
          const edgeIdx = getEdgeIndex(row, col, row + 1, col);
          const coupling = gameState.couplings[edgeIdx] ?? 0.5;
          const from = getCellCenter(row, col);
          const to = getCellCenter(row + 1, col);

          edges.push(
            <g key={`v-${row}-${col}`}>
              <line
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke={coupling > 0 ? "#3b82f6" : "#ef4444"}
                strokeWidth={Math.abs(coupling) * 4 + 1}
                opacity={0.6}
                strokeLinecap="round"
              />
              <title>{`Coupling: ${coupling.toFixed(2)}`}</title>
            </g>
          );
        }
      }
    }

    return edges;
  };

  const svgWidth = padding * 2 + gridSize * cellSize + (gridSize - 1) * gap;
  const svgHeight = padding * 2 + gridSize * cellSize + (gridSize - 1) * gap;

  return (
    <svg
      width={svgWidth}
      height={svgHeight}
      className="absolute top-0 left-0 pointer-events-none"
      style={{ zIndex: 1 }}
    >
      {renderEdges()}
    </svg>
  );
}
