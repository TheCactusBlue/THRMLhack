import { useState, useEffect } from "react";
import type { PlayerClassDefinition } from "../types";

interface ClassSelectorProps {
  onSelectClasses: (
    playerAClass: string | undefined,
    playerBClass: string | undefined
  ) => void;
  getAllClasses: () => Promise<PlayerClassDefinition[]>;
}

export default function ClassSelector({
  onSelectClasses,
  getAllClasses,
}: ClassSelectorProps) {
  const [classes, setClasses] = useState<PlayerClassDefinition[]>([]);
  const [playerAClass, setPlayerAClass] = useState<string | undefined>(
    undefined
  );
  const [playerBClass, setPlayerBClass] = useState<string | undefined>(
    undefined
  );
  const [showSelector, setShowSelector] = useState(true);

  useEffect(() => {
    // Fetch available classes on mount
    getAllClasses().then(setClasses);
  }, [getAllClasses]);

  const handleStartGame = () => {
    onSelectClasses(playerAClass, playerBClass);
    setShowSelector(false);
  };

  if (!showSelector) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 border-2 border-cyan-500 rounded-lg p-8 max-w-6xl w-full max-h-[90vh] overflow-y-auto">
        <h2 className="text-3xl font-bold text-cyan-400 mb-6 text-center">
          Choose Your Classes
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Player A Selection */}
          <div className="space-y-4">
            <h3 className="text-2xl font-bold text-blue-400 mb-4">
              Player A (Blue)
            </h3>

            <div className="space-y-2">
              {/* No class option */}
              <label
                className={`block p-4 rounded border-2 cursor-pointer transition-all ${
                  playerAClass === undefined
                    ? "border-blue-400 bg-blue-900/30"
                    : "border-gray-600 hover:border-gray-500"
                }`}
              >
                <input
                  type="radio"
                  name="playerA"
                  checked={playerAClass === undefined}
                  onChange={() => setPlayerAClass(undefined)}
                  className="mr-3"
                />
                <span className="font-bold">No Class (Default)</span>
                <p className="text-sm text-gray-400 ml-6">
                  Standard tokens and random cards
                </p>
              </label>

              {/* Class options */}
              {classes.map((classInfo) => (
                <label
                  key={classInfo.type}
                  className={`block p-4 rounded border-2 cursor-pointer transition-all ${
                    playerAClass === classInfo.type
                      ? "border-blue-400 bg-blue-900/30"
                      : "border-gray-600 hover:border-gray-500"
                  }`}
                >
                  <div className="flex gap-3">
                    <input
                      type="radio"
                      name="playerA"
                      value={classInfo.type}
                      checked={playerAClass === classInfo.type}
                      onChange={(e) => setPlayerAClass(e.target.value)}
                      className="mt-1"
                    />
                    <img
                      src={`/characters/${classInfo.type}.png`}
                      alt={classInfo.name}
                      className="w-16 h-16 rounded object-contain"
                    />
                    <div className="flex-1">
                      <span className="text-xl">
                        {classInfo.icon} <strong>{classInfo.name}</strong>
                      </span>
                      <p className="text-sm text-gray-300 mt-1">
                        {classInfo.description}
                      </p>
                      <div className="mt-2 text-sm">
                        <div className="flex gap-4 text-gray-400">
                          <span>🎯 {classInfo.base_bias_tokens} Bias</span>
                          <span>🔗 {classInfo.base_edge_tokens} Edge</span>
                        </div>
                        <p className="text-xs text-cyan-400 mt-1">
                          ⚡ {classInfo.passive_ability}
                        </p>
                      </div>
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Player B Selection */}
          <div className="space-y-4">
            <h3 className="text-2xl font-bold text-red-400 mb-4">
              Player B (Red)
            </h3>

            <div className="space-y-2">
              {/* No class option */}
              <label
                className={`block p-4 rounded border-2 cursor-pointer transition-all ${
                  playerBClass === undefined
                    ? "border-red-400 bg-red-900/30"
                    : "border-gray-600 hover:border-gray-500"
                }`}
              >
                <input
                  type="radio"
                  name="playerB"
                  checked={playerBClass === undefined}
                  onChange={() => setPlayerBClass(undefined)}
                  className="mr-3"
                />
                <span className="font-bold">No Class (Default)</span>
                <p className="text-sm text-gray-400 ml-6">
                  Standard tokens and random cards
                </p>
              </label>

              {/* Class options */}
              {classes.map((classInfo) => (
                <label
                  key={classInfo.type}
                  className={`block p-4 rounded border-2 cursor-pointer transition-all ${
                    playerBClass === classInfo.type
                      ? "border-red-400 bg-red-900/30"
                      : "border-gray-600 hover:border-gray-500"
                  }`}
                >
                  <div className="flex gap-3">
                    <input
                      type="radio"
                      name="playerB"
                      value={classInfo.type}
                      checked={playerBClass === classInfo.type}
                      onChange={(e) => setPlayerBClass(e.target.value)}
                      className="mt-1"
                    />
                    <img
                      src={`/characters/${classInfo.type}.png`}
                      alt={classInfo.name}
                      className="w-16 h-16 rounded object-contain"
                    />
                    <div className="flex-1">
                      <span className="text-xl">
                        {classInfo.icon} <strong>{classInfo.name}</strong>
                      </span>
                      <p className="text-sm text-gray-300 mt-1">
                        {classInfo.description}
                      </p>
                      <div className="mt-2 text-sm">
                        <div className="flex gap-4 text-gray-400">
                          <span>🎯 {classInfo.base_bias_tokens} Bias</span>
                          <span>🔗 {classInfo.base_edge_tokens} Edge</span>
                        </div>
                        <p className="text-xs text-cyan-400 mt-1">
                          ⚡ {classInfo.passive_ability}
                        </p>
                      </div>
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="flex justify-center gap-4">
          <button
            onClick={handleStartGame}
            className="px-8 py-3 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-lg transition-colors"
          >
            Start Game
          </button>
        </div>
      </div>
    </div>
  );
}
