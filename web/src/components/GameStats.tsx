import type { GameState } from '../types'

interface GameStatsProps {
  gameState: GameState
}

export function GameStats({ gameState }: GameStatsProps) {
  if (!gameState.last_board) return null

  return (
    <div className="stats-row">
      <div className="stat-mini">
        <span className="stat-label">Energy</span>
        <span className="stat-val">{gameState.energy?.toFixed(1)}</span>
      </div>
      <div className="stat-mini">
        <span className="stat-label">Mag.</span>
        <span className="stat-val">{gameState.magnetization?.toFixed(2)}</span>
      </div>
      <div className="stat-mini">
        <span className="stat-label">A</span>
        <span className="stat-val" style={{ color: '#3b82f6' }}>
          {gameState.player_a_territory}
        </span>
      </div>
      <div className="stat-mini">
        <span className="stat-label">B</span>
        <span className="stat-val" style={{ color: '#ef4444' }}>
          {gameState.player_b_territory}
        </span>
      </div>
    </div>
  )
}
