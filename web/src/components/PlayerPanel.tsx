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
    <div className="flex flex-col gap-2">
      <div className="text-center">
        <h2 className="m-0 mb-1 text-lg font-semibold" style={{ color }}>
          Player {player}
        </h2>
        {isActive && (
          <div
            className="text-[0.65rem] font-bold py-0.5 px-2 rounded text-white text-center tracking-wider animate-[pulse_1.5s_ease-in-out_infinite]"
            style={{ backgroundColor: color }}
          >
            YOUR TURN
          </div>
        )}
      </div>

      <div className="flex justify-between items-center p-1.5 bg-neutral-800 rounded text-xs">
        <span className="text-gray-400 font-semibold text-[0.7rem]">Wins:</span>
        <span className="text-2xl font-bold" style={{ color }}>
          {wins}
        </span>
      </div>

      <div className="flex flex-col gap-1">
        <div className="flex justify-between items-center px-2 py-1 bg-neutral-800 rounded text-[0.75rem]">
          <span className="font-semibold text-gray-300">🔗 Edge:</span>
          <span className="font-bold text-emerald-500 text-[0.75rem]">
            {budget ? budget.edge_tokens - budget.edge_tokens_used : 0}/{budget?.edge_tokens || 0}
          </span>
        </div>
        <div className="flex justify-between items-center px-2 py-1 bg-neutral-800 rounded text-[0.75rem]">
          <span className="font-semibold text-gray-300">⚡ Bias:</span>
          <span className="font-bold text-emerald-500 text-[0.75rem]">
            {budget ? budget.bias_tokens - budget.bias_tokens_used : 0}/{budget?.bias_tokens || 0}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-[1fr_50px] gap-1">
        <button
          className={`py-1.5 px-1.5 rounded font-semibold text-[0.7rem] border-2 cursor-pointer transition-all duration-200 ${
            currentPlayer === player
              ? 'bg-blue-500 border-blue-500 text-white'
              : 'border-neutral-700 bg-neutral-800 text-gray-300 hover:enabled:border-neutral-600 hover:enabled:bg-neutral-700'
          } disabled:opacity-50 disabled:cursor-not-allowed`}
          onClick={() => onSwitchPlayer(player)}
          disabled={bothReady}
        >
          {currentPlayer === player ? '✓ Playing' : 'Switch'}
        </button>

        <button
          className={`py-1.5 px-1 rounded font-semibold text-[0.7rem] border-2 cursor-pointer transition-all duration-200 ${
            isReady
              ? 'bg-emerald-500 border-emerald-500 text-white'
              : 'border-neutral-700 bg-neutral-800 text-gray-300 hover:enabled:border-neutral-600 hover:enabled:bg-neutral-700'
          } disabled:opacity-50 disabled:cursor-not-allowed`}
          onClick={onToggleReady}
          disabled={currentPlayer !== player}
        >
          {isReady ? '✓' : 'Ready'}
        </button>
      </div>
    </div>
  )
}
