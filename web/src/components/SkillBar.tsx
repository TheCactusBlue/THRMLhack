import { useState, useEffect } from 'react'
import type { Skill, SkillName, SkillCooldownStatus, PlayerType } from '../types'

interface SkillBarProps {
  playerClass: string | undefined
  player: PlayerType
  currentRound: number
  onSkillSelect: (skillName: SkillName) => void
  selectedSkill: SkillName | null
  getClassSkills: (playerClass: string) => Promise<Skill[]>
  getCooldowns: (player: PlayerType) => Promise<Record<string, SkillCooldownStatus>>
}

export function SkillBar({
  playerClass,
  player,
  currentRound,
  onSkillSelect,
  selectedSkill,
  getClassSkills,
  getCooldowns,
}: SkillBarProps) {
  const [skills, setSkills] = useState<Skill[]>([])
  const [cooldowns, setCooldowns] = useState<Record<string, SkillCooldownStatus>>({})

  useEffect(() => {
    // Fetch skills and cooldowns when player class or round changes
    if (!playerClass) return

    getClassSkills(playerClass).then(setSkills)
    getCooldowns(player).then(setCooldowns)
  }, [playerClass, player, currentRound, getClassSkills, getCooldowns])

  if (!playerClass || skills.length === 0) {
    return (
      <div className="text-center p-4 bg-gray-700 rounded-lg">
        <p className="text-gray-400">No skills available (select a class)</p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-gray-300">
        Skills (Player {player})
      </h3>
      <div className="flex flex-wrap gap-2">
        {skills.map((skill) => {
          const cooldownStatus = cooldowns[skill.name]
          const isAvailable = cooldownStatus?.available ?? true
          const roundsLeft = cooldownStatus?.rounds_until_ready ?? 0
          const isSelected = selectedSkill === skill.name

          return (
            <button
              key={skill.name}
              onClick={() => isAvailable && onSkillSelect(skill.name)}
              disabled={!isAvailable}
              className={`
                relative flex flex-col items-center justify-between
                w-28 h-36 p-2 rounded-lg border-2 transition-all
                ${isSelected
                  ? 'border-yellow-400 ring-2 ring-yellow-400 scale-105'
                  : 'border-transparent hover:border-gray-400'
                }
                ${isAvailable
                  ? 'bg-gradient-to-br from-purple-600 to-indigo-700 cursor-pointer hover:scale-105'
                  : 'bg-gray-600 opacity-50 cursor-not-allowed'
                }
              `}
              title={skill.description}
            >
              {/* Skill Icon */}
              <div className="text-3xl">
                {skill.icon}
              </div>

              {/* Skill Name */}
              <div className="text-xs font-bold text-white text-center leading-tight">
                {skill.display_name}
              </div>

              {/* Cooldown Info */}
              <div className="text-xs text-white/90">
                {isAvailable ? (
                  <span className="bg-green-500/30 px-2 py-0.5 rounded">
                    Ready!
                  </span>
                ) : (
                  <span className="bg-red-500/30 px-2 py-0.5 rounded">
                    {roundsLeft} round{roundsLeft !== 1 ? 's' : ''}
                  </span>
                )}
              </div>

              {/* Cooldown duration indicator */}
              <div className="text-[10px] text-white/60">
                CD: {skill.cooldown}
              </div>

              {/* Selected indicator */}
              {isSelected && (
                <div className="absolute -top-1 -right-1 w-5 h-5 bg-yellow-400 rounded-full flex items-center justify-center">
                  <span className="text-xs">✓</span>
                </div>
              )}

              {/* On cooldown overlay */}
              {!isAvailable && (
                <div className="absolute inset-0 bg-black/40 rounded-lg flex items-center justify-center">
                  <div className="text-2xl font-bold text-white/80">
                    {roundsLeft}
                  </div>
                </div>
              )}
            </button>
          )
        })}
      </div>

      {/* Instructions */}
      {selectedSkill && (
        <div className="text-xs text-yellow-400 bg-yellow-900/30 p-2 rounded">
          {skills.find(s => s.name === selectedSkill)?.requires_target
            ? `Click on the grid to use ${skills.find(s => s.name === selectedSkill)?.display_name}`
            : `Click the skill again to activate ${skills.find(s => s.name === selectedSkill)?.display_name}`
          }
        </div>
      )}
    </div>
  )
}
