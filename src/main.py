from random import randint
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from src import game

app = FastAPI(title="THRMLHack Energy Battle Game")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global game state
current_game: Optional[game.GameState] = None
rng_key = jax.random.key(randint(1, 9999999999))


# Request/Response models
class CreateGameRequest(BaseModel):
    grid_size: int = 5
    base_coupling: float = 0.5
    base_beta: float = 1.0
    bias_step: float = 0.5
    coupling_step: float = 0.25


class BiasUpdateRequest(BaseModel):
    row: int
    col: int
    direction: int  # +1 or -1


class CouplingUpdateRequest(BaseModel):
    cell1: Tuple[int, int]
    cell2: Tuple[int, int]
    direction: int  # +1 or -1


class SamplingRequest(BaseModel):
    n_warmup: int = 100
    n_samples: int = 50
    steps_per_sample: int = 2


class GameStateResponse(BaseModel):
    grid_size: int
    biases: list[float]
    couplings: list[float]
    beta: float
    last_board: Optional[list[list[float]]] = None
    spin_confidence: Optional[list[list[float]]] = None  # Confidence of each spin (0-1)
    energy: Optional[float] = None  # Total energy of the system
    magnetization: Optional[float] = None  # Overall magnetization (-1 to 1)
    player_a_territory: Optional[int] = None  # Number of +1 spins
    player_b_territory: Optional[int] = None  # Number of -1 spins


@app.get("/")
def read_root():
    return {"message": "THRMLHack Energy Battle API", "status": "running"}


@app.post("/game/create")
def create_game(request: CreateGameRequest):
    global current_game

    config = game.GameConfig(
        grid_size=request.grid_size,
        base_coupling=request.base_coupling,
        base_beta=request.base_beta,
        bias_step=request.bias_step,
        coupling_step=request.coupling_step,
    )

    current_game = game.create_game(config)

    return {
        "message": "Game created successfully",
        "grid_size": current_game.config.grid_size,
        "num_nodes": len(current_game.nodes),
        "num_edges": len(current_game.edges),
    }


@app.get("/game/state")
def get_game_state():
    if current_game is None:
        raise HTTPException(status_code=404, detail="No active game. Create a game first.")

    response = GameStateResponse(
        grid_size=current_game.config.grid_size,
        biases=current_game.biases.tolist(),
        couplings=current_game.couplings.tolist(),
        beta=float(current_game.beta),
    )

    if current_game.last_final_spins is not None:
        board = current_game.last_final_spins.reshape(
            (current_game.config.grid_size, current_game.config.grid_size)
        )
        response.last_board = board.tolist()

        # Calculate spin confidence from samples
        if current_game.last_samples is not None:
            # Calculate mean spin value per node (ranges from -1 to 1)
            mean_spins = jnp.mean(current_game.last_samples, axis=0)
            # Confidence is how far from 0 (uncertain) to 1 (certain)
            confidence = jnp.abs(mean_spins).reshape(
                (current_game.config.grid_size, current_game.config.grid_size)
            )
            response.spin_confidence = confidence.tolist()

        # Calculate energy using the Ising model formula
        energy = game.calculate_energy(current_game, current_game.last_final_spins)
        response.energy = float(energy)

        # Calculate magnetization (average spin)
        magnetization = jnp.mean(current_game.last_final_spins)
        response.magnetization = float(magnetization)

        # Count territory for each player
        player_a_count = int(jnp.sum(current_game.last_final_spins == 1))
        player_b_count = int(jnp.sum(current_game.last_final_spins == -1))
        response.player_a_territory = player_a_count
        response.player_b_territory = player_b_count

    return response


@app.post("/game/bias")
def update_bias(request: BiasUpdateRequest):
    if current_game is None:
        raise HTTPException(status_code=404, detail="No active game. Create a game first.")

    if request.direction not in [-1, 1]:
        raise HTTPException(status_code=400, detail="Direction must be +1 or -1")

    try:
        game.apply_bias(current_game, request.row, request.col, request.direction)
        return {
            "message": "Bias updated",
            "row": request.row,
            "col": request.col,
            "new_bias": float(current_game.biases[request.row * current_game.config.grid_size + request.col]),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/game/coupling")
def update_coupling(request: CouplingUpdateRequest):
    if current_game is None:
        raise HTTPException(status_code=404, detail="No active game. Create a game first.")

    if request.direction not in [-1, 1]:
        raise HTTPException(status_code=400, detail="Direction must be +1 or -1")

    try:
        game.apply_edge_change(
            current_game,
            request.cell1,
            request.cell2,
            request.direction,
        )
        return {
            "message": "Coupling updated",
            "cell1": request.cell1,
            "cell2": request.cell2,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/game/sample")
def run_sampling(request: SamplingRequest):
    global rng_key

    if current_game is None:
        raise HTTPException(status_code=404, detail="No active game. Create a game first.")

    try:
        rng_key, subkey = jax.random.split(rng_key)

        final_board, samples = game.run_sampling(
            current_game,
            subkey,
            n_warmup=request.n_warmup,
            n_samples=request.n_samples,
            steps_per_sample=request.steps_per_sample,
        )

        # Calculate energy of the final configuration
        final_spins = final_board.flatten()
        energy = game.calculate_energy(current_game, final_spins)
        magnetization = jnp.mean(final_spins)

        # Calculate spin confidence
        mean_spins = jnp.mean(samples, axis=0)
        confidence = jnp.abs(mean_spins).reshape(final_board.shape)

        # Count territory
        player_a_count = int(jnp.sum(final_board == 1))
        player_b_count = int(jnp.sum(final_board == -1))

        return {
            "message": "Sampling completed",
            "board": final_board.tolist(),
            "num_samples": samples.shape[0],
            "energy": float(energy),
            "magnetization": float(magnetization),
            "spin_confidence": confidence.tolist(),
            "player_a_territory": player_a_count,
            "player_b_territory": player_b_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/reset")
def reset_game():
    global current_game, rng_key

    if current_game is None:
        raise HTTPException(status_code=404, detail="No active game. Create a game first.")

    config = current_game.config
    current_game = game.create_game(config)
    rng_key = jax.random.key(randint(1, 9999999999))

    return {"message": "Game reset successfully"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
