import type { GameState } from '../types'

interface GameStatsProps {
  gameState: GameState
}

export function GameStats({ gameState }: GameStatsProps) {
  if (!gameState.last_board) return null

  return (
    <div className="flex gap-2 justify-center flex-wrap max-sm:gap-1">
      <div className="bg-neutral-900 px-3 py-1.5 rounded-md border border-neutral-800 flex flex-col items-center min-w-[70px] max-sm:min-w-[60px] max-sm:px-2 max-sm:py-1">
        <span className="text-[0.7rem] text-gray-400 uppercase tracking-[0.3px] mb-1">Energy</span>
        <span className="text-lg font-bold text-white">{gameState.energy?.toFixed(1)}</span>
      </div>
      <div className="bg-neutral-900 px-3 py-1.5 rounded-md border border-neutral-800 flex flex-col items-center min-w-[70px] max-sm:min-w-[60px] max-sm:px-2 max-sm:py-1">
        <span className="text-[0.7rem] text-gray-400 uppercase tracking-[0.3px] mb-1">Mag.</span>
        <span className="text-lg font-bold text-white">{gameState.magnetization?.toFixed(2)}</span>
      </div>
      <div className="bg-neutral-900 px-3 py-1.5 rounded-md border border-neutral-800 flex flex-col items-center min-w-[70px] max-sm:min-w-[60px] max-sm:px-2 max-sm:py-1">
        <span className="text-[0.7rem] text-gray-400 uppercase tracking-[0.3px] mb-1">A</span>
        <span className="text-lg font-bold text-blue-500">
          {gameState.player_a_territory}
        </span>
      </div>
      <div className="bg-neutral-900 px-3 py-1.5 rounded-md border border-neutral-800 flex flex-col items-center min-w-[70px] max-sm:min-w-[60px] max-sm:px-2 max-sm:py-1">
        <span className="text-[0.7rem] text-gray-400 uppercase tracking-[0.3px] mb-1">B</span>
        <span className="text-lg font-bold text-red-500">
          {gameState.player_b_territory}
        </span>
      </div>
    </div>
  )
}
