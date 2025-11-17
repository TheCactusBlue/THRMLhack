import { useState, useEffect } from 'react'
import type { Card, CardType, PlayerType } from '../types'

interface CardHandProps {
  hand: string[]  // Card types in player's hand
  playedCards: string[]  // Card types already played
  player: PlayerType
  biasTokensAvailable: number
  edgeTokensAvailable: number
  onCardSelect: (cardType: CardType) => void
  selectedCard: CardType | null
  getAllCards: () => Promise<Card[]>
}

// Card emoji icons based on card type
const CARD_ICONS: Record<CardType, string> = {
  infiltrate: '⚔️',
  disruption: '💥',
  fortress: '🛡️',
  anchor: '⚡',
  heat_wave: '🔥',
  freeze: '❄️',
}

const CARD_COLORS: Record<CardType, string> = {
  infiltrate: 'from-red-500 to-red-700',
  disruption: 'from-orange-500 to-orange-700',
  fortress: 'from-blue-500 to-blue-700',
  anchor: 'from-purple-500 to-purple-700',
  heat_wave: 'from-yellow-500 to-red-600',
  freeze: 'from-cyan-400 to-blue-600',
}

export function CardHand({
  hand,
  playedCards,
  player,
  biasTokensAvailable,
  edgeTokensAvailable,
  onCardSelect,
  selectedCard,
  getAllCards,
}: CardHandProps) {
  const [cardDefinitions, setCardDefinitions] = useState<Record<string, Card>>({})

  useEffect(() => {
    // Fetch card definitions on mount
    getAllCards().then((cards) => {
      const defs: Record<string, Card> = {}
      cards.forEach((card) => {
        defs[card.type] = card
      })
      setCardDefinitions(defs)
    })
  }, [getAllCards])

  const canPlayCard = (cardType: string): boolean => {
    const card = cardDefinitions[cardType]
    if (!card) return false
    if (playedCards.includes(cardType)) return false

    return (
      card.bias_cost <= biasTokensAvailable &&
      card.edge_cost <= edgeTokensAvailable
    )
  }

  const availableCards = hand.filter(cardType => !playedCards.includes(cardType))

  if (availableCards.length === 0) {
    return (
      <div className="text-center p-4 bg-gray-700 rounded-lg">
        <p className="text-gray-400">All cards played!</p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-gray-300">
        Your Hand (Player {player})
      </h3>
      <div className="flex flex-wrap gap-2">
        {availableCards.map((cardType) => {
          const card = cardDefinitions[cardType]
          if (!card) return null

          const isPlayable = canPlayCard(cardType)
          const isSelected = selectedCard === cardType

          return (
            <button
              key={cardType}
              onClick={() => isPlayable && onCardSelect(cardType as CardType)}
              disabled={!isPlayable}
              className={`
                relative flex flex-col items-center justify-between
                w-24 h-32 p-2 rounded-lg border-2 transition-all
                ${isSelected
                  ? 'border-yellow-400 ring-2 ring-yellow-400 scale-105'
                  : 'border-transparent hover:border-gray-400'
                }
                ${isPlayable
                  ? `bg-gradient-to-br ${CARD_COLORS[cardType as CardType]} cursor-pointer hover:scale-105`
                  : 'bg-gray-600 opacity-50 cursor-not-allowed'
                }
              `}
              title={card.description}
            >
              {/* Card Icon */}
              <div className="text-3xl">
                {CARD_ICONS[cardType as CardType]}
              </div>

              {/* Card Name */}
              <div className="text-xs font-bold text-white text-center leading-tight">
                {card.name}
              </div>

              {/* Card Cost */}
              <div className="flex gap-1 text-xs text-white/90">
                {card.bias_cost > 0 && (
                  <span className="bg-black/30 px-1 rounded">
                    ⚡{card.bias_cost}
                  </span>
                )}
                {card.edge_cost > 0 && (
                  <span className="bg-black/30 px-1 rounded">
                    🔗{card.edge_cost}
                  </span>
                )}
              </div>

              {/* Selected indicator */}
              {isSelected && (
                <div className="absolute -top-1 -right-1 w-5 h-5 bg-yellow-400 rounded-full flex items-center justify-center">
                  <span className="text-xs">✓</span>
                </div>
              )}
            </button>
          )
        })}
      </div>

      {/* Instructions */}
      {selectedCard && (
        <div className="text-xs text-yellow-400 bg-yellow-900/30 p-2 rounded">
          Click on the grid to play {cardDefinitions[selectedCard]?.name}
        </div>
      )}
    </div>
  )
}
