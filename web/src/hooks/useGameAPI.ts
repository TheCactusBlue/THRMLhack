import { useState, useCallback } from 'react'
import type { GameState, PlayerType, Card, CardType, PlayerClassDefinition } from '../types'

const API_URL = 'http://localhost:8000'

export function useGameAPI() {
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  const showMessage = (msg: string) => {
    setMessage(msg)
    setTimeout(() => setMessage(''), 3000)
  }

  const fetchGameState = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/game/state`)
      const data = await response.json()
      setGameState(data)
    } catch (error) {
      showMessage('Error fetching game state: ' + error)
    }
  }, [])

  const createGame = useCallback(async (playerAClass?: string, playerBClass?: string) => {
    setLoading(true)
    try {
      await fetch(`${API_URL}/game/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          grid_size: 5,
          base_coupling: 0.5,
          base_beta: 3.0,  // REDESIGN: Use new higher beta for more deterministic outcomes
          bias_step: 0.5,
          coupling_step: 0.25,
          player_a_class: playerAClass,
          player_b_class: playerBClass,
        }),
      })
      await fetchGameState()
      showMessage('Game created!')
    } catch (error) {
      showMessage('Error creating game: ' + error)
    } finally {
      setLoading(false)
    }
  }, [fetchGameState])

  const updateBias = useCallback(async (row: number, col: number, direction: number, player: PlayerType) => {
    setLoading(true)
    try {
      await fetch(`${API_URL}/game/bias`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ row, col, direction, player }),
      })
      await fetchGameState()
      showMessage(`Player ${player} updated bias at (${row}, ${col})`)
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error)
      showMessage('Error: ' + errorMsg)
    } finally {
      setLoading(false)
    }
  }, [fetchGameState])

  const updateCoupling = useCallback(async (
    cell1: [number, number],
    cell2: [number, number],
    direction: number,
    player: PlayerType
  ) => {
    setLoading(true)
    try {
      await fetch(`${API_URL}/game/coupling`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cell1, cell2, direction, player }),
      })
      await fetchGameState()
      showMessage(`Player ${player} updated coupling between (${cell1}) and (${cell2})`)
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error)
      showMessage('Error: ' + errorMsg)
    } finally {
      setLoading(false)
    }
  }, [fetchGameState])

  const runSampling = useCallback(async () => {
    setLoading(true)
    try {
      await fetch(`${API_URL}/game/sample`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n_warmup: 100,
          n_samples: 50,
          steps_per_sample: 2,
        }),
      })
      await fetchGameState()
      showMessage('Sampling completed!')
    } catch (error) {
      showMessage('Error running sampling: ' + error)
    } finally {
      setLoading(false)
    }
  }, [fetchGameState])

  const resetGame = useCallback(async () => {
    setLoading(true)
    try {
      await fetch(`${API_URL}/game/reset`, { method: 'POST' })
      await fetchGameState()
      showMessage('Game reset!')
    } catch (error) {
      showMessage('Error resetting game: ' + error)
    } finally {
      setLoading(false)
    }
  }, [fetchGameState])

  const toggleReady = useCallback(async (player: PlayerType, currentReady: boolean) => {
    setLoading(true)
    try {
      await fetch(`${API_URL}/game/ready/${player}?ready=${!currentReady}`, {
        method: 'POST',
      })
      await fetchGameState()
      showMessage(`Player ${player} is ${!currentReady ? 'ready' : 'not ready'}`)
    } catch (error) {
      showMessage('Error setting ready status: ' + error)
    } finally {
      setLoading(false)
    }
  }, [fetchGameState])

  const batchActions = useCallback(async (actions: any[]) => {
    setLoading(true)
    try {
      await fetch(`${API_URL}/game/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(actions),
      })
      await fetchGameState()
      showMessage(`Executed ${actions.length} action(s)`)
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error)
      showMessage('Error: ' + errorMsg)
    } finally {
      setLoading(false)
    }
  }, [fetchGameState])

  const nextRound = useCallback(async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/game/next-round`, { method: 'POST' })
      const data = await response.json()

      if (data.game_winner) {
        showMessage(`Game Over! Winner: Player ${data.game_winner}`)
      } else {
        showMessage(`Round ${data.current_round - 1} won by Player ${data.round_winner}! Starting Round ${data.current_round}`)
      }

      await fetchGameState()
      return data
    } catch (error) {
      showMessage('Error advancing to next round: ' + error)
      return null
    } finally {
      setLoading(false)
    }
  }, [fetchGameState])

  const previewSampling = useCallback(async (n_quick_samples: number = 10) => {
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/game/preview?n_quick_samples=${n_quick_samples}`, {
        method: 'POST',
      })
      const data = await response.json()
      showMessage('Preview generated!')
      return data
    } catch (error) {
      showMessage('Error generating preview: ' + error)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  // Card system API calls
  const getAllCards = useCallback(async (): Promise<Card[]> => {
    try {
      const response = await fetch(`${API_URL}/cards/all`)
      const data = await response.json()
      return data.cards
    } catch (error) {
      showMessage('Error fetching cards: ' + error)
      return []
    }
  }, [])

  const getAllClasses = useCallback(async (): Promise<PlayerClassDefinition[]> => {
    try {
      const response = await fetch(`${API_URL}/classes/all`)
      const data = await response.json()
      return data.classes
    } catch (error) {
      showMessage('Error fetching classes: ' + error)
      return []
    }
  }, [])

  const playCard = useCallback(async (
    cardType: CardType,
    targetRow: number,
    targetCol: number,
    player: PlayerType
  ) => {
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/game/play-card`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          card_type: cardType,
          target_row: targetRow,
          target_col: targetCol,
          player,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to play card')
      }

      const data = await response.json()
      await fetchGameState()
      showMessage(`Player ${player} played ${cardType}!`)
      return data
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error)
      showMessage('Error: ' + errorMsg)
      return null
    } finally {
      setLoading(false)
    }
  }, [fetchGameState])

  return {
    gameState,
    loading,
    message,
    createGame,
    fetchGameState,
    updateBias,
    updateCoupling,
    runSampling,
    resetGame,
    toggleReady,
    nextRound,
    previewSampling,
    batchActions,
    // Card system
    getAllCards,
    playCard,
    // Class system
    getAllClasses,
  }
}
