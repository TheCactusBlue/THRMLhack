# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thermodynamic Tactics is a turn-based Ising model energy battle game built for the THRML hackathon. Two players compete by manipulating an Ising model grid through biases and couplings, then run THRML-powered Gibbs sampling to see the resulting spin configuration. The player who controls more territory (grid cells) wins each round.

The project consists of:
- **Backend**: FastAPI server (Python) that manages game state and runs THRML sampling with JAX
- **Frontend**: React + TypeScript web UI (Vite + Tailwind CSS)

## Common Development Commands

### Backend (Python)
```bash
# Run the FastAPI server (port 8000)
uv run python -m src.main

# The backend uses uv for Python dependency management
# Dependencies are defined in pyproject.toml
```

### Frontend (React)
```bash
cd web

# Install dependencies
npm install

# Run dev server (port 5173)
npm run dev

# Build for production
npm run build

# Lint
npm run lint

# Preview production build
npm run preview
```

### Full Stack
According to the README, there should be a `./run.sh` script to start both servers, but it doesn't currently exist in the repository. To run both, start them in separate terminals.

## Architecture

### Backend Architecture (src/)

The backend has two main files:

1. **src/main.py** - FastAPI application and HTTP endpoints
   - Manages global game state (`current_game`)
   - Provides REST API for game operations
   - Handles CORS for React frontend (localhost:5173)
   - Key endpoints:
     - `POST /game/create` - Initialize new game
     - `GET /game/state` - Get current game state
     - `POST /game/bias` - Update cell bias (player influences spin direction)
     - `POST /game/coupling` - Update edge coupling (player influences neighbor alignment)
     - `POST /game/sample` - Run THRML sampling to determine board state
     - `POST /game/preview` - Quick preview of predicted outcomes (doesn't modify state)
     - `POST /game/ready/{player}` - Mark player as ready
     - `POST /game/next-round` - Score current round and advance
     - `POST /game/reset` - Reset game

2. **src/game.py** - Core game logic and THRML integration
   - `GameState` dataclass: holds grid, biases, couplings, beta, player budgets, turn state
   - `GameConfig` dataclass: configuration parameters (grid_size, base_coupling, base_beta, etc.)
   - Grid representation: 5x5 grid of SpinNodes (configurable)
   - Edges: Each cell connects to right and down neighbors
   - Checkerboard blocking for efficient Gibbs sampling
   - Key functions:
     - `create_game()` - Initialize game state
     - `apply_bias()` - Modify bias at (row, col) for a player
     - `apply_edge_change()` - Modify coupling between neighboring cells
     - `run_sampling()` - Execute THRML Gibbs sampling, returns final board and samples
     - `calculate_energy()` - Compute Ising model energy: E = -Σh_i·s_i - Σ J_ij·s_i·s_j
     - `get_probability_preview()` - Fast preview sampling for player decision-making
     - `score_round()` - Determine round winner based on territory control
     - `reset_round()` - Decay biases and reset budgets for next round

### Frontend Architecture (web/src/)

1. **App.tsx** - Main application component
   - Manages game initialization and overall layout
   - Orchestrates GameGrid, GameControls, GameStats, and PlayerPanel components

2. **hooks/useGameAPI.ts** - Custom hook for backend communication
   - Wraps all API calls to the FastAPI backend
   - Handles game state management
   - Provides functions: createGame, updateBias, updateCoupling, runSampling, etc.
   - Uses fetch with `/api` prefix (proxied to localhost:8000 via Vite)

3. **components/**
   - **GameGrid.tsx** - Renders the 5x5 spin grid, handles cell interactions
   - **GameControls.tsx** - UI controls for bias mode, coupling mode, sampling, etc.
   - **GameStats.tsx** - Displays game statistics (energy, magnetization, territory)
   - **PlayerPanel.tsx** - Shows player budgets, ready status, and controls

4. **types.ts** - TypeScript interfaces
   - `GameState`, `PlayerBudget`, `PlayerType` interfaces
   - Matches backend Pydantic models

### Key Architectural Concepts

**Ising Model Physics:**
- Each grid cell is a spin that can be +1 (blue/Player A) or -1 (red/Player B)
- **Biases (h_i)**: Push individual spins toward +1 or -1
- **Couplings (J_ij)**: Control alignment between neighboring spins (positive = prefer same, negative = prefer opposite)
- **Beta (β)**: Inverse temperature; higher = more deterministic (currently 3.0)
- **Energy**: E = -Σh_i·s_i - Σ J_ij·s_i·s_j (system minimizes energy)

**Turn-Based Game Flow:**
1. Players spend tokens to adjust biases and couplings
2. Both players mark ready
3. THRML sampling runs to find low-energy spin configuration
4. Round is scored (player with more cells wins)
5. Biases decay by 50% (strategic continuity across rounds)
6. Budgets reset, advance to next round
7. First to win 3 of 5 rounds wins the game

**THRML Integration:**
- Uses THRML library with JAX for efficient sampling
- Checkerboard blocking strategy for Gibbs updates (even/odd cells)
- Sampling uses `IsingEBM`, `IsingSamplingProgram`, and `hinton_init`
- Warmup, samples, and steps_per_sample are configurable

**Player Budgets:**
- Each player has limited tokens per round:
  - 2 bias tokens (modify individual cell preferences)
  - 3 edge tokens (modify coupling between neighbors)
- Strategy involves choosing where to spend tokens for maximum territorial advantage

**State Management:**
- Backend maintains single global `current_game` state
- Frontend polls `/game/state` for updates
- Preview endpoint allows "what-if" exploration without committing

## Technology Stack

- **Backend**: Python 3.14+, FastAPI, JAX, THRML, Uvicorn, Pydantic
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS 4
- **Dependency Management**: uv (Python), npm (JavaScript)

## Configuration

- Backend port: 8000
- Frontend dev port: 5173
- Vite proxy: `/api` -> `http://localhost:8000`
- CORS: Backend allows requests from `http://localhost:5173`

## Important Implementation Notes

- JAX arrays are immutable; use `.at[idx].add(delta)` syntax for updates
- THRML sampling returns boolean arrays that must be converted to spin values (+1/-1)
- Sample aggregation uses majority voting with tie-breaking
- Biases decay between rounds (0.5x multiplier) to create strategic continuity
- Preview sampling uses reduced warmup (20 vs 100) and fewer samples for speed
