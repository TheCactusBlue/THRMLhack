import { useState } from 'react'
import type { GameState, PlayerType } from '../types'

const API_URL = 'http://localhost:8000'

export function useGameAPI() {
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  const showMessage = (msg: string) => {
    setMessage(msg)
    setTimeout(() => setMessage(''), 3000)
  }

  const fetchGameState = async () => {
    try {
      const response = await fetch(`${API_URL}/game/state`)
      const data = await response.json()
      setGameState(data)
    } catch (error) {
      showMessage('Error fetching game state: ' + error)
    }
  }

  const createGame = async () => {
    setLoading(true)
    try {
      await fetch(`${API_URL}/game/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          grid_size: 5,
          base_coupling: 0.5,
          base_beta: 1.0,
          bias_step: 0.5,
          coupling_step: 0.25,
        }),
      })
      await fetchGameState()
      showMessage('Game created!')
    } catch (error) {
      showMessage('Error creating game: ' + error)
    } finally {
      setLoading(false)
    }
  }

  const updateBias = async (row: number, col: number, direction: number, player: PlayerType) => {
    setLoading(true)
    try {
      await fetch(`${API_URL}/game/bias`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ row, col, direction, player }),
      })
      await fetchGameState()
      showMessage(`Player ${player} updated bias at (${row}, ${col})`)
    } catch (error: any) {
      const errorMsg = error.message || error
      showMessage('Error: ' + errorMsg)
    } finally {
      setLoading(false)
    }
  }

  const updateCoupling = async (
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
    } catch (error: any) {
      const errorMsg = error.message || error
      showMessage('Error: ' + errorMsg)
    } finally {
      setLoading(false)
    }
  }

  const runSampling = async () => {
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
  }

  const resetGame = async () => {
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
  }

  const toggleReady = async (player: PlayerType, currentReady: boolean) => {
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
  }

  const nextRound = async () => {
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
  }

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
  }
}
