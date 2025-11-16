export function GameLegend() {
  return (
    <div className="bg-neutral-900 border-2 border-neutral-800 rounded-lg p-3 text-xs">
      <h3 className="font-bold text-gray-300 mb-2">Legend</h3>

      <div className="space-y-2">
        {/* Cell Colors */}
        <div>
          <div className="text-gray-400 font-semibold mb-1">Cell Colors:</div>
          <div className="flex gap-3 flex-wrap">
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded bg-blue-500"></div>
              <span className="text-gray-300">Player A</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded bg-red-500"></div>
              <span className="text-gray-300">Player B</span>
            </div>
          </div>
        </div>

        {/* Coupling Lines */}
        <div>
          <div className="text-gray-400 font-semibold mb-1">Coupling Lines:</div>
          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <svg width="30" height="4">
                <line x1="0" y1="2" x2="30" y2="2" stroke="#3b82f6" strokeWidth="3" />
              </svg>
              <span className="text-gray-300">Strong alignment</span>
            </div>
            <div className="flex items-center gap-1">
              <svg width="30" height="4">
                <line x1="0" y1="2" x2="30" y2="2" stroke="#3b82f6" strokeWidth="1" />
              </svg>
              <span className="text-gray-300">Weak alignment</span>
            </div>
            <div className="flex items-center gap-1">
              <svg width="30" height="4">
                <line x1="0" y1="2" x2="30" y2="2" stroke="#ef4444" strokeWidth="3" />
              </svg>
              <span className="text-gray-300">Anti-aligned</span>
            </div>
          </div>
        </div>

        {/* Entrenchment */}
        <div>
          <div className="text-gray-400 font-semibold mb-1">Entrenchment:</div>
          <div className="flex items-center gap-1">
            <div className="flex gap-0.5">
              <div className="w-1.5 h-1.5 rounded-full bg-yellow-400"></div>
              <div className="w-1.5 h-1.5 rounded-full bg-yellow-400"></div>
              <div className="w-1.5 h-1.5 rounded-full bg-yellow-400"></div>
            </div>
            <span className="text-gray-300">Fortified cells</span>
          </div>
        </div>

        {/* Cell Text */}
        <div>
          <div className="text-gray-400 font-semibold mb-1">Cell Numbers:</div>
          <div className="space-y-1 text-gray-300">
            <div>+1.5 = Bias toward player</div>
            <div>67 = Confidence %</div>
          </div>
        </div>
      </div>
    </div>
  );
}
