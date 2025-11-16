import { useState, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

interface GameState {
  grid_size: number
  biases: number[]
  couplings: number[]
  beta: number
  last_board?: number[][]
}

function App() {
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')
  const [selectedCell, setSelectedCell] = useState<[number, number] | null>(null)
  const [edgeMode, setEdgeMode] = useState(false)

  useEffect(() => {
    createGame()
  }, [])

  const showMessage = (msg: string) => {
    setMessage(msg)
    setTimeout(() => setMessage(''), 3000)
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

  const fetchGameState = async () => {
    try {
      const response = await fetch(`${API_URL}/game/state`)
      const data = await response.json()
      setGameState(data)
    } catch (error) {
      showMessage('Error fetching game state: ' + error)
    }
  }

  const updateBias = async (row: number, col: number, direction: number) => {
    setLoading(true)
    try {
      await fetch(`${API_URL}/game/bias`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ row, col, direction }),
      })
      await fetchGameState()
      showMessage(`Bias updated at (${row}, ${col})`)
    } catch (error) {
      showMessage('Error updating bias: ' + error)
    } finally {
      setLoading(false)
    }
  }

  const updateCoupling = async (
    cell1: [number, number],
    cell2: [number, number],
    direction: number
  ) => {
    setLoading(true)
    try {
      await fetch(`${API_URL}/game/coupling`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cell1, cell2, direction }),
      })
      await fetchGameState()
      showMessage(`Coupling updated between (${cell1}) and (${cell2})`)
    } catch (error) {
      showMessage('Error updating coupling: ' + error)
    } finally {
      setLoading(false)
    }
  }

  const runSampling = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/game/sample`, {
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
      setSelectedCell(null)
      setEdgeMode(false)
      showMessage('Game reset!')
    } catch (error) {
      showMessage('Error resetting game: ' + error)
    } finally {
      setLoading(false)
    }
  }

  const handleCellClick = (row: number, col: number) => {
    if (edgeMode) {
      // Edge coupling mode
      if (selectedCell === null) {
        setSelectedCell([row, col])
        showMessage(`Selected cell (${row}, ${col}). Click a neighbor to adjust coupling.`)
      } else {
        // Check if cells are neighbors
        const [r1, c1] = selectedCell
        const isNeighbor =
          (Math.abs(r1 - row) === 1 && c1 === col) ||
          (Math.abs(c1 - col) === 1 && r1 === row)

        if (isNeighbor) {
          // Ask for direction
          const direction = window.confirm('Increase coupling? (OK = increase, Cancel = decrease)')
            ? 1
            : -1
          updateCoupling(selectedCell, [row, col], direction)
        } else {
          showMessage('Cells must be neighbors!')
        }
        setSelectedCell(null)
      }
    } else {
      // Bias mode
      const direction = window.confirm(`Adjust bias at (${row}, ${col})?\n\nOK = Push to +1 (blue)\nCancel = Push to -1 (red)`)
        ? 1
        : -1
      updateBias(row, col, direction)
    }
  }

  const getCellColor = (row: number, col: number) => {
    if (!gameState) return '#444'

    if (gameState.last_board) {
      const spin = gameState.last_board[row][col]
      if (spin > 0) return '#3b82f6' // blue for +1
      if (spin < 0) return '#ef4444' // red for -1
    }

    // Color based on bias
    const idx = row * gameState.grid_size + col
    const bias = gameState.biases[idx]
    if (bias > 0) return `rgba(59, 130, 246, ${Math.min(Math.abs(bias) / 2, 0.8)})`
    if (bias < 0) return `rgba(239, 68, 68, ${Math.min(Math.abs(bias) / 2, 0.8)})`
    return '#444'
  }

  const isSelected = (row: number, col: number) => {
    return selectedCell !== null && selectedCell[0] === row && selectedCell[1] === col
  }

  if (!gameState) {
    return (
      <div className="app">
        <h1>THRMLHack Energy Battle</h1>
        <p>Loading...</p>
      </div>
    )
  }

  return (
    <div className="app">
      <h1>THRMLHack Energy Battle</h1>
      <p className="subtitle">Turn-based Ising model spin game</p>

      {message && <div className="message">{message}</div>}

      <div className="grid-container">
        <div
          className="grid"
          style={{
            gridTemplateColumns: `repeat(${gameState.grid_size}, 60px)`,
          }}
        >
          {Array.from({ length: gameState.grid_size }).map((_, row) =>
            Array.from({ length: gameState.grid_size }).map((_, col) => (
              <div
                key={`${row}-${col}`}
                className={`cell ${isSelected(row, col) ? 'selected' : ''}`}
                style={{ backgroundColor: getCellColor(row, col) }}
                onClick={() => handleCellClick(row, col)}
              >
                {gameState.last_board && (
                  <span className="spin-value">
                    {gameState.last_board[row][col] > 0 ? '+' : '-'}
                  </span>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      <div className="controls">
        <div className="mode-toggle">
          <button
            className={!edgeMode ? 'active' : ''}
            onClick={() => {
              setEdgeMode(false)
              setSelectedCell(null)
            }}
          >
            Bias Mode
          </button>
          <button
            className={edgeMode ? 'active' : ''}
            onClick={() => {
              setEdgeMode(true)
              setSelectedCell(null)
            }}
          >
            Coupling Mode
          </button>
        </div>

        <div className="action-buttons">
          <button onClick={runSampling} disabled={loading} className="primary">
            Run Sampling
          </button>
          <button onClick={resetGame} disabled={loading}>
            Reset Game
          </button>
          <button onClick={createGame} disabled={loading}>
            New Game
          </button>
        </div>
      </div>

      <div className="info">
        <p>
          <strong>Mode:</strong> {edgeMode ? 'Coupling' : 'Bias'}
        </p>
        <p>
          <strong>Blue (+1):</strong> Player A | <strong>Red (-1):</strong> Player B
        </p>
        <p>
          <strong>Beta:</strong> {gameState.beta.toFixed(2)}
        </p>
      </div>
    </div>
  )
}

export default App
