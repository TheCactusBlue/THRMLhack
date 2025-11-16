import type { GameState } from '../types'

interface GameStatsProps {
  gameState: GameState
}

export function GameStats({ gameState }: GameStatsProps) {
  if (!gameState.last_board) return null

  // PHASE 2: Calculate entrenchment stats
  const calculateEntrenchment = () => {
    if (!gameState.entrenchment || !gameState.last_board) return { a: 0, b: 0 };

    let aEntrenched = 0;
    let bEntrenched = 0;

    for (let row = 0; row < gameState.grid_size; row++) {
      for (let col = 0; col < gameState.grid_size; col++) {
        const spin = gameState.last_board[row][col];
        const entrenchment = gameState.entrenchment[row][col];

        if (entrenchment >= 2) {
          if (spin > 0) aEntrenched++;
          else if (spin < 0) bEntrenched++;
        }
      }
    }

    return { a: aEntrenched, b: bEntrenched };
  };

  const entrenchment = calculateEntrenchment();

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
      {/* PHASE 2: Show entrenchment stats */}
      {(entrenchment.a > 0 || entrenchment.b > 0) && (
        <div className="bg-neutral-900 px-3 py-1.5 rounded-md border border-yellow-800/50 flex flex-col items-center min-w-[70px] max-sm:min-w-[60px] max-sm:px-2 max-sm:py-1">
          <span className="text-[0.7rem] text-gray-400 uppercase tracking-[0.3px] mb-1">Fortified</span>
          <span className="text-sm font-bold">
            <span className="text-blue-400">{entrenchment.a}</span>
            <span className="text-gray-500 mx-1">/</span>
            <span className="text-red-400">{entrenchment.b}</span>
          </span>
        </div>
      )}
    </div>
  )
}
