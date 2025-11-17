from random import randint
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Tuple, List
import jax
import jax.numpy as jnp
from src import game
from src.cards import CardType, Card
from src.player_classes import PlayerClass, get_all_classes, get_class_definition
from src.skills import SkillName, SKILLS, get_class_skills, use_skill, get_cooldown_status
from src.skill_effects import execute_skill

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
    player_a_class: Optional[str] = None  # CLASS SYSTEM: Player A's chosen class
    player_b_class: Optional[str] = None  # CLASS SYSTEM: Player B's chosen class


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
    hand: List[str] = []  # List of card types in hand
    played_cards: List[str] = []  # List of card types played this round
    player_class: Optional[str] = None  # CLASS SYSTEM: Player's chosen class
    cards_redrawn: int = 0  # CLASS SYSTEM: Track card redraws (for Hybrid class)


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

    # PHASE 2: Entrenchment data
    entrenchment: Optional[list[list[int]]] = None  # Rounds of consecutive control per cell


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

    # CLASS SYSTEM: Parse player classes from request
    player_a_class = None
    player_b_class = None
    if request.player_a_class:
        try:
            player_a_class = PlayerClass(request.player_a_class)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid player class: {request.player_a_class}")
    if request.player_b_class:
        try:
            player_b_class = PlayerClass(request.player_b_class)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid player class: {request.player_b_class}")

    current_game = game.create_game(config, player_a_class, player_b_class)

    return {
        "message": "Game created successfully",
        "grid_size": current_game.config.grid_size,
        "num_nodes": len(current_game.nodes),
        "num_edges": len(current_game.edges),
        "player_a_class": player_a_class.value if player_a_class else None,
        "player_b_class": player_b_class.value if player_b_class else None,
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
            hand=[str(card.value) for card in current_game.player_a_budget.hand],
            played_cards=[str(card.value) for card in current_game.player_a_budget.played_cards],
            player_class=current_game.player_a_budget.player_class.value if current_game.player_a_budget.player_class else None,
            cards_redrawn=current_game.player_a_budget.cards_redrawn,
        ),
        player_b_budget=PlayerBudgetResponse(
            edge_tokens=current_game.player_b_budget.edge_tokens,
            bias_tokens=current_game.player_b_budget.bias_tokens,
            edge_tokens_used=current_game.player_b_budget.edge_tokens_used,
            bias_tokens_used=current_game.player_b_budget.bias_tokens_used,
            hand=[str(card.value) for card in current_game.player_b_budget.hand],
            played_cards=[str(card.value) for card in current_game.player_b_budget.played_cards],
            player_class=current_game.player_b_budget.player_class.value if current_game.player_b_budget.player_class else None,
            cards_redrawn=current_game.player_b_budget.cards_redrawn,
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

    # PHASE 2: Include entrenchment data
    entrenchment_grid = current_game.entrenchment.reshape(
        (current_game.config.grid_size, current_game.config.grid_size)
    )
    response.entrenchment = [[int(val) for val in row] for row in entrenchment_grid.tolist()]

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


@app.post("/game/batch")
def batch_actions(actions: list[dict]):
    """
    Execute multiple actions in a single request.
    Each action should have:
    - type: 'bias' or 'coupling'
    - params: parameters for the action
    """
    if current_game is None:
        raise HTTPException(status_code=404, detail="No active game. Create a game first.")

    try:
        for action in actions:
            action_type = action.get('type')
            params = action.get('params', {})

            if action_type == 'bias':
                game.apply_bias(
                    current_game,
                    params['row'],
                    params['col'],
                    params['direction'],
                    params.get('player', 'A')
                )
            elif action_type == 'coupling':
                game.apply_edge_change(
                    current_game,
                    tuple(params['cell1']),
                    tuple(params['cell2']),
                    params['direction'],
                    params.get('player', 'A')
                )
            else:
                raise ValueError(f"Unknown action type: {action_type}")

        return {
            "message": f"Executed {len(actions)} actions successfully",
            "actions_count": len(actions),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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


class PlayCardRequest(BaseModel):
    card_type: str  # CardType enum value
    target_row: int
    target_col: int
    player: str = 'A'  # 'A' or 'B'


class UseSkillRequest(BaseModel):
    skill_name: str  # SkillName enum value
    player: str = 'A'  # 'A' or 'B'
    target_row: Optional[int] = None
    target_col: Optional[int] = None


@app.get("/cards/all")
def get_all_cards():
    """
    Get information about all available card types.
    """
    cards_info = []
    for card_type in CardType:
        card = Card.get_card_definition(card_type)
        cards_info.append({
            "type": card.card_type.value,
            "name": card.name,
            "description": card.description,
            "bias_cost": card.bias_cost,
            "edge_cost": card.edge_cost,
        })
    return {"cards": cards_info}


@app.get("/classes/all")
def get_all_classes_endpoint():
    """
    Get information about all available player classes.
    """
    classes_list = get_all_classes()
    classes_info = []
    for class_def in classes_list:
        classes_info.append({
            "type": class_def.class_type.value,
            "name": class_def.name,
            "description": class_def.description,
            "base_bias_tokens": class_def.base_bias_tokens,
            "base_edge_tokens": class_def.base_edge_tokens,
            "passive_ability": class_def.passive_ability,
            "icon": class_def.icon,
            "color_scheme": class_def.color_scheme,
        })
    return {"classes": classes_info}


@app.get("/skills/all")
def get_all_skills():
    """
    Get information about all available skills.
    """
    skills_info = []
    for skill_name, skill in SKILLS.items():
        skills_info.append({
            "name": skill.name.value,
            "display_name": skill.display_name,
            "description": skill.description,
            "cooldown": skill.cooldown,
            "requires_target": skill.requires_target,
            "icon": skill.icon,
        })
    return {"skills": skills_info}


@app.get("/skills/class/{player_class}")
def get_class_skills_endpoint(player_class: str):
    """
    Get the 3 skills available to a specific player class.
    """
    skills = get_class_skills(player_class)
    skills_info = []
    for skill in skills:
        skills_info.append({
            "name": skill.name.value,
            "display_name": skill.display_name,
            "description": skill.description,
            "cooldown": skill.cooldown,
            "requires_target": skill.requires_target,
            "icon": skill.icon,
        })
    return {"skills": skills_info}


@app.get("/game/cooldowns/{player}")
def get_cooldowns(player: str):
    """
    Get cooldown status for all skills for a player.
    """
    if current_game is None:
        raise HTTPException(status_code=404, detail="No active game")

    if player not in ['A', 'B']:
        raise HTTPException(status_code=400, detail="Player must be 'A' or 'B'")

    budget = current_game.player_a_budget if player == 'A' else current_game.player_b_budget

    if budget.player_class is None:
        return {"cooldowns": {}}

    status = get_cooldown_status(
        budget.player_class.value,
        current_game.current_round,
        budget.skill_cooldowns
    )

    return {"cooldowns": status}


@app.post("/game/use-skill")
def use_skill_endpoint(request: UseSkillRequest):
    """
    Use a skill from the player's class.
    """
    if current_game is None:
        raise HTTPException(status_code=404, detail="No active game")

    if request.player not in ['A', 'B']:
        raise HTTPException(status_code=400, detail="Player must be 'A' or 'B'")

    try:
        skill_name = SkillName(request.skill_name)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid skill name: {request.skill_name}")

    budget = current_game.player_a_budget if request.player == 'A' else current_game.player_b_budget

    # Check if player has this skill
    player_skills = get_class_skills(budget.player_class.value if budget.player_class else None)
    if not any(s.name == skill_name for s in player_skills):
        # Check if it's a morphed skill
        if budget.morphed_skill != skill_name:
            raise HTTPException(status_code=400, detail=f"Skill {skill_name.value} not available to this class")

    # Check if skill is on cooldown
    skill = SKILLS[skill_name]
    if not use_skill(skill_name, current_game.current_round, budget.skill_cooldowns):
        state = budget.skill_cooldowns.get(skill_name)
        rounds_left = state.rounds_until_ready(current_game.current_round, skill.cooldown) if state else 0
        raise HTTPException(
            status_code=400,
            detail=f"Skill {skill_name.value} is on cooldown ({rounds_left} rounds remaining)"
        )

    # Check if skill requires target
    if skill.requires_target and (request.target_row is None or request.target_col is None):
        raise HTTPException(status_code=400, detail=f"Skill {skill_name.value} requires a target location")

    try:
        # Execute the skill
        updated_game, message = execute_skill(
            current_game,
            skill_name,
            request.player,
            request.target_row,
            request.target_col
        )

        return {
            "message": message,
            "player": request.player,
            "skill": skill_name.value,
            "cooldown_status": get_cooldown_status(
                budget.player_class.value if budget.player_class else None,
                current_game.current_round,
                budget.skill_cooldowns
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/game/play-card")
def play_card_endpoint(request: PlayCardRequest):
    """
    Play a card from the player's hand at a target location.

    The card will apply its effect to the game board based on the card type.
    """
    if current_game is None:
        raise HTTPException(status_code=404, detail="No active game. Create a game first.")

    if request.player not in ['A', 'B']:
        raise HTTPException(status_code=400, detail="Player must be 'A' or 'B'")

    try:
        # Convert string to CardType enum
        card_type = CardType(request.card_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid card type: {request.card_type}")

    try:
        # Play the card
        game.play_card(current_game, request.player, card_type, request.target_row, request.target_col)

        budget = current_game.player_a_budget if request.player == 'A' else current_game.player_b_budget

        return {
            "message": f"Card {card_type.value} played successfully",
            "player": request.player,
            "card_type": card_type.value,
            "target": (request.target_row, request.target_col),
            "remaining_hand": [str(c.value) for c in budget.hand if c not in budget.played_cards],
            "played_cards": [str(c.value) for c in budget.played_cards],
            "tokens_remaining": {
                "bias": budget.bias_tokens - budget.bias_tokens_used,
                "edge": budget.edge_tokens - budget.edge_tokens_used,
            },
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
