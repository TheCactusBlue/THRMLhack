import { useState, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

interface PlayerBudget {
  edge_tokens: number
  bias_tokens: number
  edge_tokens_used: number
  bias_tokens_used: number
}

interface GameState {
  grid_size: number
  biases: number[]
  couplings: number[]
  beta: number
  last_board?: number[][]
  spin_confidence?: number[][]
  energy?: number
  magnetization?: number
  player_a_territory?: number
  player_b_territory?: number
  current_round: number
  player_a_budget?: PlayerBudget
  player_b_budget?: PlayerBudget
  player_a_ready: boolean
  player_b_ready: boolean
  player_a_wins: number
  player_b_wins: number
  max_rounds: number
  game_winner?: string | null
}

function App() {
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')
  const [selectedCell, setSelectedCell] = useState<[number, number] | null>(null)
  const [edgeMode, setEdgeMode] = useState(false)
  const [showConfidence, setShowConfidence] = useState(false)
  const [currentPlayer, setCurrentPlayer] = useState<'A' | 'B'>('A')
  const [roundWinner, setRoundWinner] = useState<string | null>(null)

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
        body: JSON.stringify({ row, col, direction, player: currentPlayer }),
      })
      await fetchGameState()
      showMessage(`Player ${currentPlayer} updated bias at (${row}, ${col})`)
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
    direction: number
  ) => {
    setLoading(true)
    try {
      await fetch(`${API_URL}/game/coupling`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cell1, cell2, direction, player: currentPlayer }),
      })
      await fetchGameState()
      showMessage(`Player ${currentPlayer} updated coupling between (${cell1}) and (${cell2})`)
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
      setCurrentPlayer('A')
      setRoundWinner(null)
      showMessage('Game reset!')
    } catch (error) {
      showMessage('Error resetting game: ' + error)
    } finally {
      setLoading(false)
    }
  }

  const toggleReady = async () => {
    if (!gameState) return

    setLoading(true)
    try {
      const isReady = currentPlayer === 'A' ? gameState.player_a_ready : gameState.player_b_ready
      await fetch(`${API_URL}/game/ready/${currentPlayer}?ready=${!isReady}`, {
        method: 'POST',
      })
      await fetchGameState()
      showMessage(`Player ${currentPlayer} is ${!isReady ? 'ready' : 'not ready'}`)
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

      setRoundWinner(data.round_winner)

      if (data.game_winner) {
        showMessage(`Game Over! Winner: Player ${data.game_winner}`)
      } else {
        showMessage(`Round ${data.current_round - 1} won by Player ${data.round_winner}! Starting Round ${data.current_round}`)
      }

      await fetchGameState()
    } catch (error) {
      showMessage('Error advancing to next round: ' + error)
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

  const bothReady = gameState.player_a_ready && gameState.player_b_ready

  const renderPlayerPanel = (player: 'A' | 'B') => {
    const budget = player === 'A' ? gameState.player_a_budget : gameState.player_b_budget
    const isReady = player === 'A' ? gameState.player_a_ready : gameState.player_b_ready
    const wins = player === 'A' ? gameState.player_a_wins : gameState.player_b_wins
    const isActive = currentPlayer === player
    const color = player === 'A' ? '#3b82f6' : '#ef4444'

    return (
      <div className={`player-panel ${isActive ? 'active' : ''}`} style={{ borderColor: isActive ? color : '#333' }}>
        <div className="player-header">
          <h2 style={{ color }}>Player {player}</h2>
          {isActive && <div className="turn-indicator" style={{ backgroundColor: color }}>YOUR TURN</div>}
        </div>

        <div className="score-display">
          <span className="wins-label">Wins:</span>
          <span className="wins-value" style={{ color }}>{wins}</span>
        </div>

        <div className="budget-compact">
          <div className="budget-row">
            <span className="token-label">🔗 Edge:</span>
            <span className="budget-value">
              {budget ? budget.edge_tokens - budget.edge_tokens_used : 0}/{budget?.edge_tokens || 0}
            </span>
          </div>
          <div className="budget-row">
            <span className="token-label">⚡ Bias:</span>
            <span className="budget-value">
              {budget ? budget.bias_tokens - budget.bias_tokens_used : 0}/{budget?.bias_tokens || 0}
            </span>
          </div>
        </div>

        <div className="player-actions">
          <button
            className={currentPlayer === player ? 'switch-btn active' : 'switch-btn'}
            onClick={() => setCurrentPlayer(player)}
            disabled={bothReady}
          >
            {currentPlayer === player ? '✓ Playing' : 'Switch'}
          </button>

          <button
            className={isReady ? 'ready-btn ready' : 'ready-btn'}
            onClick={toggleReady}
            disabled={currentPlayer !== player}
            style={{ backgroundColor: isReady ? '#10b981' : '#6b7280' }}
          >
            {isReady ? '✓' : 'Ready'}
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="app-container">
      <div className="top-bar">
        <div className="game-title">
          <h1>Energy Battle</h1>
          <div className="round-badge">Round {gameState.current_round}/{gameState.max_rounds}</div>
        </div>

        <div className="mode-indicator">
          <span className="mode-label">Mode:</span>
          <div className="mode-pills">
            <button
              className={!edgeMode ? 'mode-pill active' : 'mode-pill'}
              onClick={() => { setEdgeMode(false); setSelectedCell(null); }}
            >
              ⚡ Bias
            </button>
            <button
              className={edgeMode ? 'mode-pill active' : 'mode-pill'}
              onClick={() => { setEdgeMode(true); setSelectedCell(null); }}
            >
              🔗 Coupling
            </button>
          </div>
        </div>

        <div className="game-controls">
          <button onClick={resetGame} disabled={loading} className="ctrl-btn reset">Reset</button>
          <button onClick={createGame} disabled={loading} className="ctrl-btn new">New</button>
        </div>
      </div>

      {message && <div className="message">{message}</div>}

      {gameState.game_winner && (
        <div className="game-winner">
          🏆 Game Winner: Player {gameState.game_winner}!
        </div>
      )}
      {roundWinner && !gameState.game_winner && (
        <div className="round-winner">
          Last Round: Player {roundWinner} Won
        </div>
      )}

      <div className="game-layout">
        <div className="side-panel">
          {renderPlayerPanel('A')}
        </div>

        <div className="center-panel">
          <div className="grid-wrapper">
            <div
              className="grid"
              style={{
                gridTemplateColumns: `repeat(${gameState.grid_size}, 55px)`,
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
                    {gameState.last_board && !showConfidence && (
                      <span className="spin-value">
                        {gameState.last_board[row][col] > 0 ? '+' : '-'}
                      </span>
                    )}
                    {showConfidence && gameState.spin_confidence && (
                      <span className="confidence-value">
                        {(gameState.spin_confidence[row][col] * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="action-bar">
            <button
              onClick={runSampling}
              disabled={loading || !bothReady}
              className="sample-btn"
            >
              {bothReady ? '⚡ Run Sampling' : '⏳ Waiting...'}
            </button>

            {gameState.last_board && (
              <button onClick={nextRound} disabled={loading} className="next-btn">
                Next Round →
              </button>
            )}

            <button
              className={`view-btn ${!showConfidence ? 'active' : ''}`}
              onClick={() => setShowConfidence(false)}
            >
              Spins
            </button>
            <button
              className={`view-btn ${showConfidence ? 'active' : ''}`}
              onClick={() => setShowConfidence(true)}
              disabled={!gameState.spin_confidence}
            >
              Confidence
            </button>
          </div>

          {gameState.last_board && (
            <div className="stats-row">
              <div className="stat-mini">
                <span className="stat-label">Energy</span>
                <span className="stat-val">{gameState.energy?.toFixed(1)}</span>
              </div>
              <div className="stat-mini">
                <span className="stat-label">Mag.</span>
                <span className="stat-val">{gameState.magnetization?.toFixed(2)}</span>
              </div>
              <div className="stat-mini">
                <span className="stat-label">A</span>
                <span className="stat-val" style={{ color: '#3b82f6' }}>
                  {gameState.player_a_territory}
                </span>
              </div>
              <div className="stat-mini">
                <span className="stat-label">B</span>
                <span className="stat-val" style={{ color: '#ef4444' }}>
                  {gameState.player_b_territory}
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="side-panel">
          {renderPlayerPanel('B')}
        </div>
      </div>
    </div>
  )
}

export default App
