import { Link } from 'react-router-dom'
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
    <div className="flex justify-between items-center px-6 py-3 bg-neutral-900 rounded-lg mb-2 border border-neutral-800 max-lg:flex-wrap max-lg:gap-3">
      <div className="flex items-center gap-6">
        <h1 className="text-2xl m-0 bg-gradient-to-br from-blue-500 to-red-500 bg-clip-text text-transparent max-sm:text-xl">
          Thermodynamic Tactics
        </h1>
        <div className="text-sm font-bold py-1.5 px-3.5 bg-neutral-800 rounded-md text-blue-400 border-2 border-neutral-700">
          Round {gameState.current_round}/{gameState.max_rounds}
        </div>
      </div>

      <div className="flex items-center gap-3 max-lg:order-3 max-lg:w-full max-lg:justify-center">
        <Link
          to="/how-to-play"
          className="py-1.5 px-3.5 rounded-md text-[0.85rem] font-semibold cursor-pointer transition-all duration-200 border-2 bg-blue-600 border-blue-600 text-white hover:bg-blue-700 max-sm:hidden"
        >
          ❓ How to Play
        </Link>

        <span className="text-[0.85rem] text-gray-400 uppercase tracking-wide">Mode:</span>
        <div className="flex gap-2">
          <button
            className={`py-1.5 px-3.5 rounded-md border-2 text-[0.85rem] font-semibold cursor-pointer transition-all duration-200 ${
              !edgeMode
                ? 'bg-blue-500 border-blue-500 text-white'
                : 'border-neutral-700 bg-neutral-800 text-gray-300 hover:border-neutral-600 hover:bg-neutral-700'
            }`}
            onClick={() => {
              onSetEdgeMode(false)
              onSetSelectedCell(null)
            }}
          >
            ⚡ Bias
          </button>
          <button
            className={`py-1.5 px-3.5 rounded-md border-2 text-[0.85rem] font-semibold cursor-pointer transition-all duration-200 ${
              edgeMode
                ? 'bg-blue-500 border-blue-500 text-white'
                : 'border-neutral-700 bg-neutral-800 text-gray-300 hover:border-neutral-600 hover:bg-neutral-700'
            }`}
            onClick={() => {
              onSetEdgeMode(true)
              onSetSelectedCell(null)
            }}
          >
            🔗 Coupling
          </button>
        </div>

        <button
          onClick={onResetGame}
          disabled={loading}
          className="py-1.5 px-3.5 rounded-md font-semibold text-[0.85rem] cursor-pointer transition-all duration-200 border-2 bg-red-500 border-red-500 text-white hover:enabled:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Reset
        </button>
        <button
          onClick={onCreateGame}
          disabled={loading}
          className="py-1.5 px-3.5 rounded-md font-semibold text-[0.85rem] cursor-pointer transition-all duration-200 border-2 bg-neutral-700 border-neutral-600 text-white hover:enabled:bg-neutral-600 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          New
        </button>
      </div>
    </div>
  )
}
