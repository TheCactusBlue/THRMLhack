# THRMLHack Web UI

Interactive web interface for the THRMLHack Energy Battle game - a turn-based Ising model spin game.

## Setup

Install dependencies:

```bash
npm install
```

## Development

Start the development server:

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

**Important:** Make sure the FastAPI backend is running on port 8000 before using the web UI.

## How to Play

1. **Bias Mode**: Click cells to adjust their bias toward +1 (blue) or -1 (red)
2. **Coupling Mode**: Click two neighboring cells to adjust the coupling strength between them
3. **Run Sampling**: Execute THRML sampling to see the final spin configuration
4. **Reset/New Game**: Start fresh or create a new game

## Game Mechanics

- The game uses an Ising model with JAX and THRML for sampling
- Each cell represents a spin that can be +1 or -1
- Biases push individual spins in a particular direction
- Couplings control how neighboring spins interact
- Sampling reveals the energy-minimized configuration

## Build

Build for production:

```bash
npm run build
```

Preview the production build:

```bash
npm run preview
```
