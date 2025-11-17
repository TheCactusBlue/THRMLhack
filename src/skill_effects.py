"""
Skill effect implementations for Thermodynamic Tactics.

Each skill has a function that executes its effect on the game state.
"""

import random
from typing import Tuple, Optional
import jax.numpy as jnp

from .skills import SkillName
from .game import GameState


def execute_skill(
    game: GameState,
    skill_name: SkillName,
    player: str,
    target_row: Optional[int] = None,
    target_col: Optional[int] = None
) -> Tuple[GameState, str]:
    """
    Execute a skill effect and return updated game state and a message.

    Args:
        game: Current game state
        skill_name: Which skill to execute
        player: 'A' or 'B'
        target_row: Target row (if skill requires target)
        target_col: Target column (if skill requires target)

    Returns:
        (updated_game, message)
    """
    budget = game.player_a_budget if player == 'A' else game.player_b_budget
    opponent_budget = game.player_b_budget if player == 'A' else game.player_a_budget
    player_dir = 1 if player == 'A' else -1

    # Execute skill based on type
    if skill_name == SkillName.DEEP_STRIKE:
        return _deep_strike(game, target_row, target_col, player_dir)

    elif skill_name == SkillName.SURGICAL_CUT:
        return _surgical_cut(game, target_row, target_col)

    elif skill_name == SkillName.MOMENTUM:
        return _momentum(game, budget, player)

    elif skill_name == SkillName.REINFORCE:
        return _reinforce(game, target_row, target_col)

    elif skill_name == SkillName.BASTION:
        return _bastion(game, target_row, target_col, budget)

    elif skill_name == SkillName.ENTRENCH:
        return _entrench(game, budget, player)

    elif skill_name == SkillName.CASCADE:
        return _cascade(game, target_row, target_col, player_dir)

    elif skill_name == SkillName.FEEDBACK_LOOP:
        return _feedback_loop(game, target_row, target_col)

    elif skill_name == SkillName.EXPLOIT:
        return _exploit(game, budget, opponent_budget, player)

    elif skill_name == SkillName.HEAT_WAVE:
        return _heat_wave(game, target_row, target_col)

    elif skill_name == SkillName.POLARITY_FLIP:
        return _polarity_flip(game, target_row, target_col)

    elif skill_name == SkillName.GAMBIT:
        return _gambit(game, budget, player)

    elif skill_name == SkillName.MORPH:
        return _morph(game, budget, opponent_budget, player)

    elif skill_name == SkillName.BALANCE_SHIFT:
        return _balance_shift(game, budget, player)

    elif skill_name == SkillName.ADAPT:
        return _adapt(game, budget, player)

    return game, "Unknown skill"


# ========== INFILTRATOR SKILLS ==========

def _deep_strike(game: GameState, row: int, col: int, player_dir: int) -> Tuple[GameState, str]:
    """Apply massive bias (+2.0) to a single cell."""
    idx = row * game.config.grid_size + col
    game.biases = game.biases.at[idx].add(2.0 * player_dir)
    return game, f"Deep Strike applied massive bias to ({row}, {col})"


def _surgical_cut(game: GameState, row: int, col: int) -> Tuple[GameState, str]:
    """Sever all edges around a target cell (set J=0)."""
    grid_size = game.config.grid_size
    idx = row * grid_size + col
    edges_cut = 0

    # Check all 4 neighbors (up, down, left, right)
    neighbors = []
    if row > 0:  # up
        neighbors.append((idx - grid_size, idx))
    if row < grid_size - 1:  # down
        neighbors.append((idx, idx + grid_size))
    if col > 0:  # left
        neighbors.append((idx - 1, idx))
    if col < grid_size - 1:  # right
        neighbors.append((idx, idx + 1))

    for n1, n2 in neighbors:
        key = (min(n1, n2), max(n1, n2))
        if key in game.edge_index:
            edge_idx = game.edge_index[key]
            game.couplings = game.couplings.at[edge_idx].set(0.0)
            edges_cut += 1

    return game, f"Surgical Cut severed {edges_cut} edges around ({row}, {col})"


def _momentum(game: GameState, budget, player: str) -> Tuple[GameState, str]:
    """Gain +1 bias token if you control >50% of the board."""
    if game.last_final_spins is None:
        return game, "Momentum: No board state yet to check control"

    player_dir = 1 if player == 'A' else -1
    controlled = jnp.sum(game.last_final_spins == player_dir)
    total = len(game.last_final_spins)

    if controlled > total / 2:
        budget.bias_tokens += 1
        return game, f"Momentum: Gained +1 bias token (controlling {controlled}/{total} cells)"
    else:
        return game, f"Momentum: No bonus (only controlling {controlled}/{total} cells)"


# ========== FORTRESS SKILLS ==========

def _reinforce(game: GameState, row: int, col: int) -> Tuple[GameState, str]:
    """Strengthen all edges in a 3x3 region (+0.5 coupling)."""
    grid_size = game.config.grid_size
    edges_reinforced = 0

    # Iterate over 3x3 region centered on (row, col)
    for r in range(max(0, row - 1), min(grid_size, row + 2)):
        for c in range(max(0, col - 1), min(grid_size, col + 2)):
            idx = r * grid_size + c

            # Strengthen edge to the right
            if c < grid_size - 1:
                key = (idx, idx + 1)
                if key in game.edge_index:
                    edge_idx = game.edge_index[key]
                    game.couplings = game.couplings.at[edge_idx].add(0.5)
                    edges_reinforced += 1

            # Strengthen edge downward
            if r < grid_size - 1:
                key = (idx, idx + grid_size)
                if key in game.edge_index:
                    edge_idx = game.edge_index[key]
                    game.couplings = game.couplings.at[edge_idx].add(0.5)
                    edges_reinforced += 1

    return game, f"Reinforce: Strengthened {edges_reinforced} edges in 3x3 region"


def _bastion(game: GameState, row: int, col: int, budget) -> Tuple[GameState, str]:
    """Lock a cell's spin for 2 rounds."""
    idx = row * game.config.grid_size + col
    budget.locked_cells[idx] = 2
    return game, f"Bastion: Locked cell ({row}, {col}) for 2 rounds"


def _entrench(game: GameState, budget, player: str) -> Tuple[GameState, str]:
    """Gain +2 edge tokens if you have 3+ entrenched cells."""
    entrenched_count = jnp.sum(game.entrenchment >= 3)

    if entrenched_count >= 3:
        budget.edge_tokens += 2
        return game, f"Entrench: Gained +2 edge tokens ({entrenched_count} entrenched cells)"
    else:
        return game, f"Entrench: No bonus (only {entrenched_count} entrenched cells, need 3+)"


# ========== MANIPULATOR SKILLS ==========

def _cascade(game: GameState, row: int, col: int, player_dir: int) -> Tuple[GameState, str]:
    """Apply bias to a line of 5 cells (diminishing: 1.0, 0.8, 0.6, 0.4, 0.2)."""
    grid_size = game.config.grid_size
    multipliers = [1.0, 0.8, 0.6, 0.4, 0.2]
    cells_affected = 0

    # Apply cascade horizontally to the right
    for i, mult in enumerate(multipliers):
        c = col + i
        if c < grid_size:
            idx = row * grid_size + c
            game.biases = game.biases.at[idx].add(mult * player_dir)
            cells_affected += 1

    return game, f"Cascade: Applied diminishing bias to {cells_affected} cells"


def _feedback_loop(game: GameState, row: int, col: int) -> Tuple[GameState, str]:
    """Create a 2x2 region with all edges set to +1.5 (strong alignment)."""
    grid_size = game.config.grid_size

    if row >= grid_size - 1 or col >= grid_size - 1:
        return game, "Feedback Loop: Invalid target (need room for 2x2)"

    edges_set = 0

    # Set all edges in the 2x2 grid
    for r in range(row, row + 2):
        for c in range(col, col + 2):
            idx = r * grid_size + c

            # Edge to the right
            if c < col + 1:
                key = (idx, idx + 1)
                if key in game.edge_index:
                    edge_idx = game.edge_index[key]
                    game.couplings = game.couplings.at[edge_idx].set(1.5)
                    edges_set += 1

            # Edge downward
            if r < row + 1:
                key = (idx, idx + grid_size)
                if key in game.edge_index:
                    edge_idx = game.edge_index[key]
                    game.couplings = game.couplings.at[edge_idx].set(1.5)
                    edges_set += 1

    return game, f"Feedback Loop: Created strong alignment in 2x2 region ({edges_set} edges)"


def _exploit(game: GameState, budget, opponent_budget, player: str) -> Tuple[GameState, str]:
    """Steal 1 edge token from opponent if they have >3 tokens."""
    if opponent_budget.edge_tokens > 3:
        opponent_budget.edge_tokens -= 1
        budget.edge_tokens += 1
        return game, "Exploit: Stole 1 edge token from opponent"
    else:
        return game, f"Exploit: Opponent has only {opponent_budget.edge_tokens} edge tokens (need >3)"


# ========== WILDCARD SKILLS ==========

def _heat_wave(game: GameState, row: int, col: int) -> Tuple[GameState, str]:
    """Reduce beta by 1.5 in a 3x3 region (more randomness)."""
    # Note: In the current implementation, beta is a single scalar for the whole grid.
    # To implement regional beta, we'd need to modify the sampling logic.
    # For now, reduce global beta.
    game.beta = jnp.maximum(0.1, game.beta - 1.5)
    return game, f"Heat Wave: Reduced beta to {float(game.beta):.2f} (global effect)"


def _polarity_flip(game: GameState, row: int, col: int) -> Tuple[GameState, str]:
    """Invert all couplings in a 2x2 region (J → -J)."""
    grid_size = game.config.grid_size

    if row >= grid_size - 1 or col >= grid_size - 1:
        return game, "Polarity Flip: Invalid target (need room for 2x2)"

    edges_flipped = 0

    for r in range(row, row + 2):
        for c in range(col, col + 2):
            idx = r * grid_size + c

            # Flip edge to the right
            if c < col + 1:
                key = (idx, idx + 1)
                if key in game.edge_index:
                    edge_idx = game.edge_index[key]
                    game.couplings = game.couplings.at[edge_idx].multiply(-1.0)
                    edges_flipped += 1

            # Flip edge downward
            if r < row + 1:
                key = (idx, idx + grid_size)
                if key in game.edge_index:
                    edge_idx = game.edge_index[key]
                    game.couplings = game.couplings.at[edge_idx].multiply(-1.0)
                    edges_flipped += 1

    return game, f"Polarity Flip: Inverted {edges_flipped} couplings in 2x2 region"


def _gambit(game: GameState, budget, player: str) -> Tuple[GameState, str]:
    """Randomly gain either +2 bias tokens OR +2 edge tokens."""
    if random.choice([True, False]):
        budget.bias_tokens += 2
        return game, "Gambit: Gained +2 bias tokens"
    else:
        budget.edge_tokens += 2
        return game, "Gambit: Gained +2 edge tokens"


# ========== HYBRID SKILLS ==========

def _morph(game: GameState, budget, opponent_budget, player: str) -> Tuple[GameState, str]:
    """Copy one skill from opponent's class."""
    if opponent_budget.player_class is None:
        return game, "Morph: Opponent has no class to copy from"

    # Get opponent's skills
    from .skills import CLASS_SKILLS
    opponent_skills = CLASS_SKILLS.get(opponent_budget.player_class.value, [])

    if not opponent_skills:
        return game, "Morph: Opponent has no skills"

    # Copy a random skill
    copied_skill = random.choice(opponent_skills)
    budget.morphed_skill = copied_skill

    return game, f"Morph: Copied {copied_skill.value} from opponent (usable this round)"


def _balance_shift(game: GameState, budget, player: str) -> Tuple[GameState, str]:
    """Convert up to 2 bias tokens ↔ edge tokens."""
    # For simplicity, let's convert bias → edge if we have more bias, else edge → bias
    if budget.bias_tokens > budget.edge_tokens:
        # Convert bias to edge
        amount = min(2, budget.bias_tokens)
        budget.bias_tokens -= amount
        budget.edge_tokens += amount
        return game, f"Balance Shift: Converted {amount} bias tokens → edge tokens"
    else:
        # Convert edge to bias
        amount = min(2, budget.edge_tokens)
        budget.edge_tokens -= amount
        budget.bias_tokens += amount
        return game, f"Balance Shift: Converted {amount} edge tokens → bias tokens"


def _adapt(game: GameState, budget, player: str) -> Tuple[GameState, str]:
    """Reduce cooldown of your other skills by 1 round each."""
    skills_adapted = 0

    for skill_name, skill_state in budget.skill_cooldowns.items():
        if skill_name != SkillName.ADAPT and skill_state.last_used_round > -1:
            # Artificially advance the "last used" round by 1 to reduce effective cooldown
            skill_state.last_used_round -= 1
            skills_adapted += 1

    return game, f"Adapt: Reduced cooldown of {skills_adapted} skills by 1 round"
