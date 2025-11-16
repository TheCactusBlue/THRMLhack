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
    base_beta: float = 3.0  # REDESIGN: Default to new higher beta
    bias_step: float = 0.5
    coupling_step: float = 0.25


class BiasUpdateRequest(BaseModel):
    row: int
    col: int
    direction: int  # +1 or -1
    player: str = 'A'  # 'A' or 'B'


class CouplingUpdateRequest(BaseModel):
    cell1: Tuple[int, int]
    cell2: Tuple[int, int]
    direction: int  # +1 or -1
    player: str = 'A'  # 'A' or 'B'


class SamplingRequest(BaseModel):
    n_warmup: int = 100
    n_samples: int = 50
    steps_per_sample: int = 2


class PlayerBudgetResponse(BaseModel):
    edge_tokens: int
    bias_tokens: int
    edge_tokens_used: int
    bias_tokens_used: int


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

    # Turn-based game state
    current_round: int = 1
    player_a_budget: Optional[PlayerBudgetResponse] = None
    player_b_budget: Optional[PlayerBudgetResponse] = None
    player_a_ready: bool = False
    player_b_ready: bool = False
    player_a_wins: int = 0
    player_b_wins: int = 0
    max_rounds: int = 5
    game_winner: Optional[str] = None


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
        current_round=current_game.current_round,
        player_a_budget=PlayerBudgetResponse(
            edge_tokens=current_game.player_a_budget.edge_tokens,
            bias_tokens=current_game.player_a_budget.bias_tokens,
            edge_tokens_used=current_game.player_a_budget.edge_tokens_used,
            bias_tokens_used=current_game.player_a_budget.bias_tokens_used,
        ),
        player_b_budget=PlayerBudgetResponse(
            edge_tokens=current_game.player_b_budget.edge_tokens,
            bias_tokens=current_game.player_b_budget.bias_tokens,
            edge_tokens_used=current_game.player_b_budget.edge_tokens_used,
            bias_tokens_used=current_game.player_b_budget.bias_tokens_used,
        ),
        player_a_ready=current_game.player_a_ready,
        player_b_ready=current_game.player_b_ready,
        player_a_wins=current_game.player_a_wins,
        player_b_wins=current_game.player_b_wins,
        max_rounds=current_game.max_rounds,
        game_winner=game.check_game_winner(current_game),
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

    if request.player not in ['A', 'B']:
        raise HTTPException(status_code=400, detail="Player must be 'A' or 'B'")

    try:
        game.apply_bias(current_game, request.row, request.col, request.direction, request.player)
        return {
            "message": "Bias updated",
            "row": request.row,
            "col": request.col,
            "player": request.player,
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

    if request.player not in ['A', 'B']:
        raise HTTPException(status_code=400, detail="Player must be 'A' or 'B'")

    try:
        game.apply_edge_change(
            current_game,
            request.cell1,
            request.cell2,
            request.direction,
            request.player,
        )
        return {
            "message": "Coupling updated",
            "cell1": request.cell1,
            "cell2": request.cell2,
            "player": request.player,
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


@app.post("/game/ready/{player}")
def set_ready(player: str, ready: bool = True):
    if current_game is None:
        raise HTTPException(status_code=404, detail="No active game. Create a game first.")

    if player not in ['A', 'B']:
        raise HTTPException(status_code=400, detail="Player must be 'A' or 'B'")

    try:
        game.set_player_ready(current_game, player, ready)
        both_ready = game.both_players_ready(current_game)

        return {
            "message": f"Player {player} ready status set to {ready}",
            "player": player,
            "ready": ready,
            "both_ready": both_ready,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/game/next-round")
def next_round():
    if current_game is None:
        raise HTTPException(status_code=404, detail="No active game. Create a game first.")

    try:
        # First score the current round
        if current_game.last_final_spins is None:
            raise HTTPException(status_code=400, detail="Run sampling before starting next round")

        round_winner = game.score_round(current_game)

        # Check if game is over
        game_winner = game.check_game_winner(current_game)

        if game_winner:
            return {
                "message": f"Game over! Winner: Player {game_winner}",
                "round_winner": round_winner,
                "game_winner": game_winner,
                "player_a_wins": current_game.player_a_wins,
                "player_b_wins": current_game.player_b_wins,
            }

        # Reset for next round
        game.reset_round(current_game)

        return {
            "message": "Round scored and new round started",
            "round_winner": round_winner,
            "current_round": current_game.current_round,
            "player_a_wins": current_game.player_a_wins,
            "player_b_wins": current_game.player_b_wins,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/game/preview")
def preview_sampling(n_quick_samples: int = 10):
    """
    REDESIGN: Preview endpoint - run quick sampling to show predicted outcomes.

    This helps players make informed decisions by showing them:
    - Probability heatmap of cell states
    - Predicted territory counts
    - Confidence levels

    Returns preview data without modifying game state.
    """
    global rng_key

    if current_game is None:
        raise HTTPException(status_code=404, detail="No active game. Create a game first.")

    try:
        rng_key, subkey = jax.random.split(rng_key)

        preview_data = game.get_probability_preview(current_game, subkey, n_quick_samples)

        # Reshape probabilities to grid
        grid_size = current_game.config.grid_size
        probabilities_grid = preview_data['probabilities'].reshape((grid_size, grid_size))
        mean_spins_grid = preview_data['mean_spins'].reshape((grid_size, grid_size))
        std_spins_grid = preview_data['std_spins'].reshape((grid_size, grid_size))

        return {
            "message": "Preview completed",
            "probabilities": probabilities_grid.tolist(),  # 0-1 probability of being Player A
            "mean_spins": mean_spins_grid.tolist(),  # -1 to +1 average spin
            "std_spins": std_spins_grid.tolist(),  # Standard deviation
            "predicted_a_count": preview_data['predicted_a_count'],
            "predicted_b_count": preview_data['predicted_b_count'],
            "confidence": preview_data['confidence'],  # Overall confidence 0-1
            "n_samples": n_quick_samples,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
