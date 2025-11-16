import type { GameState, PlayerType } from '../types'

interface PlayerPanelProps {
  player: PlayerType
  gameState: GameState
  currentPlayer: PlayerType
  bothReady: boolean
  onSwitchPlayer: (player: PlayerType) => void
  onToggleReady: () => void
}

export function PlayerPanel({
  player,
  gameState,
  currentPlayer,
  bothReady,
  onSwitchPlayer,
  onToggleReady,
}: PlayerPanelProps) {
  const budget = player === 'A' ? gameState.player_a_budget : gameState.player_b_budget
  const isReady = player === 'A' ? gameState.player_a_ready : gameState.player_b_ready
  const wins = player === 'A' ? gameState.player_a_wins : gameState.player_b_wins
  const isActive = currentPlayer === player
  const color = player === 'A' ? '#3b82f6' : '#ef4444'

  return (
    <div className={`player-panel ${isActive ? 'active' : ''}`} style={{ borderColor: isActive ? color : '#333' }}>
      <div className="player-header">
        <h2 style={{ color }}>Player {player}</h2>
        {isActive && <div className="turn-indicator" style={{ backgroundColor: color }}>YOUR TURN</div>}
      </div>

      <div className="score-display">
        <span className="wins-label">Wins:</span>
        <span className="wins-value" style={{ color }}>{wins}</span>
      </div>

      <div className="budget-compact">
        <div className="budget-row">
          <span className="token-label">🔗 Edge:</span>
          <span className="budget-value">
            {budget ? budget.edge_tokens - budget.edge_tokens_used : 0}/{budget?.edge_tokens || 0}
          </span>
        </div>
        <div className="budget-row">
          <span className="token-label">⚡ Bias:</span>
          <span className="budget-value">
            {budget ? budget.bias_tokens - budget.bias_tokens_used : 0}/{budget?.bias_tokens || 0}
          </span>
        </div>
      </div>

      <div className="player-actions">
        <button
          className={currentPlayer === player ? 'switch-btn active' : 'switch-btn'}
          onClick={() => onSwitchPlayer(player)}
          disabled={bothReady}
        >
          {currentPlayer === player ? '✓ Playing' : 'Switch'}
        </button>

        <button
          className={isReady ? 'ready-btn ready' : 'ready-btn'}
          onClick={onToggleReady}
          disabled={currentPlayer !== player}
          style={{ backgroundColor: isReady ? '#10b981' : '#6b7280' }}
        >
          {isReady ? '✓' : 'Ready'}
        </button>
      </div>
    </div>
  )
}
