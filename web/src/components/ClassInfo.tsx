import { useState, useEffect } from "react";
import type { PlayerClassDefinition, PlayerType } from "../types";

interface ClassInfoProps {
  playerClass: string | undefined;
  player: PlayerType;
  getAllClasses: () => Promise<PlayerClassDefinition[]>;
}

export function ClassInfo({
  playerClass,
  player,
  getAllClasses,
}: ClassInfoProps) {
  const [classDefinition, setClassDefinition] =
    useState<PlayerClassDefinition | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (playerClass) {
      getAllClasses().then((classes) => {
        const classDef = classes.find((c) => c.type === playerClass);
        if (classDef) {
          setClassDefinition(classDef);
        }
      });
    }
  }, [playerClass, getAllClasses]);

  if (!classDefinition) {
    return null;
  }

  const color = player === "A" ? "blue" : "red";

  return (
    <div className="mt-2 border-t border-neutral-700 pt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className={`w-full text-left px-2 py-1 rounded text-xs font-semibold transition-colors ${
          expanded
            ? `bg-${color}-900/30 text-${color}-400`
            : "bg-neutral-800 text-gray-400 hover:bg-neutral-700"
        }`}
      >
        {expanded ? "▼" : "▶"} Class Info
      </button>

      {expanded && (
        <div className="mt-2 p-2 bg-neutral-800 rounded text-xs space-y-2">
          <div className="flex items-center gap-2">
            <img
              src={`/characters/${classDefinition.type}.png`}
              alt={classDefinition.name}
              className="w-12 h-12 rounded object-cover"
            />
            <div className="flex-1">
              <div className="flex items-center gap-1">
                <span className="text-lg">{classDefinition.icon}</span>
                <span className="font-bold text-white">
                  {classDefinition.name}
                </span>
              </div>
              <div className="text-gray-400 text-[0.65rem]">
                {classDefinition.description}
              </div>
            </div>
          </div>

          <div className="space-y-1">
            <div className="text-gray-400 font-semibold text-[0.65rem]">
              Starting Resources:
            </div>
            <div className="flex gap-3 text-[0.7rem]">
              <span>🎯 {classDefinition.base_bias_tokens} Bias</span>
              <span>🔗 {classDefinition.base_edge_tokens} Edge</span>
            </div>
          </div>

          <div className="space-y-1">
            <div className="text-gray-400 font-semibold text-[0.65rem]">
              Passive Ability:
            </div>
            <div className="text-cyan-400 text-[0.7rem] leading-tight">
              ⚡ {classDefinition.passive_ability}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
