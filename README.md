# Thermodynamic Tactics

A turn-based Ising model energy battle game built for the THRML hackathon.

## Overview

THRMLHack is an interactive game where players manipulate an Ising model by adjusting biases and couplings, then run THRML sampling to see the resulting spin configuration. The game features:

- 5x5 grid of spins (configurable)
- Bias adjustment to push spins toward +1 (blue) or -1 (red)
- Coupling adjustment between neighboring cells
- THRML-powered Gibbs sampling using JAX
- Interactive web UI built with React and FastAPI

## Running the Game

### Quick Start (both servers)

```bash
./run.sh
```

This will start both the backend (port 8000) and frontend (port 5173) servers.

### Manual Start

**Backend:**

```bash
uv run python -m src.main
```

The FastAPI server will start on `http://localhost:8000`

**Frontend** (in a separate terminal):

```bash
cd web
npm install
npm run dev
```

The React app will be available at `http://localhost:5173`

## Game Mechanics

- **Bias Mode**: Click cells to adjust their energy bias (OK = push to +1/blue, Cancel = push to -1/red)
- **Coupling Mode**: Click two neighboring cells to adjust the coupling strength between them
- **Run Sampling**: Execute THRML sampling to find the energy-minimized spin configuration
- **Reset/New Game**: Clear the board or start fresh

The game uses the THRML library with JAX for efficient sampling of the Ising model energy landscape.
