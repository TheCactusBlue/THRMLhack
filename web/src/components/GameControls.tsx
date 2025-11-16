import type { GameState } from '../types'

interface GameControlsProps {
  gameState: GameState
  edgeMode: boolean
  loading: boolean
  onSetEdgeMode: (edgeMode: boolean) => void
  onSetSelectedCell: (cell: null) => void
  onResetGame: () => void
  onCreateGame: () => void
}

export function GameControls({
  gameState,
  edgeMode,
  loading,
  onSetEdgeMode,
  onSetSelectedCell,
  onResetGame,
  onCreateGame,
}: GameControlsProps) {
  return (
    <div className="top-bar">
      <div className="game-title">
        <h1>Energy Battle</h1>
        <div className="round-badge">Round {gameState.current_round}/{gameState.max_rounds}</div>
      </div>

      <div className="mode-indicator">
        <span className="mode-label">Mode:</span>
        <div className="mode-pills">
          <button
            className={!edgeMode ? 'mode-pill active' : 'mode-pill'}
            onClick={() => {
              onSetEdgeMode(false)
              onSetSelectedCell(null)
            }}
          >
            ⚡ Bias
          </button>
          <button
            className={edgeMode ? 'mode-pill active' : 'mode-pill'}
            onClick={() => {
              onSetEdgeMode(true)
              onSetSelectedCell(null)
            }}
          >
            🔗 Coupling
          </button>
        </div>
      </div>

      <div className="game-controls">
        <button onClick={onResetGame} disabled={loading} className="ctrl-btn reset">Reset</button>
        <button onClick={onCreateGame} disabled={loading} className="ctrl-btn new">New</button>
      </div>
    </div>
  )
}
