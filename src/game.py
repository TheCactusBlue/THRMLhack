# energy_battle_backend.py

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import jax
import jax.numpy as jnp

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

# Import card system from separate module
from .cards import CardType, Card, deal_cards_to_player, can_play_card, play_card
# Import player class system
from .player_classes import PlayerClass, deal_class_cards, get_starting_budget, apply_class_passive
# Import skill system
from .skills import SkillName, SkillState, initialize_skill_cooldowns


# -------------------------
# Config + State containers
# -------------------------
@dataclass
class GameConfig:
    grid_size: int = 5
    base_coupling: float = 0.5
    base_beta: float = 3.0  # REDESIGN: Increased from 1.0 for more deterministic outcomes
    bias_step: float = 0.5       # how much each bias click changes h_i
    coupling_step: float = 0.25  # how much each edge click changes J_ij
    bias_decay_rate: float = 0.5  # REDESIGN: Biases decay by this factor each round


@dataclass
class PlayerBudget:
    edge_tokens: int = 3
    bias_tokens: int = 2
    edge_tokens_used: int = 0
    bias_tokens_used: int = 0
    # Card system
    hand: List[CardType] = field(default_factory=list)  # Cards in player's hand
    played_cards: List[CardType] = field(default_factory=list)  # Cards played this round
    # Class system
    player_class: Optional[PlayerClass] = None  # Player's chosen class
    cards_redrawn: int = 0  # Track card redraws (for Hybrid class)
    # Skill system
    skill_cooldowns: Dict[SkillName, SkillState] = field(default_factory=dict)  # Track skill cooldowns
    locked_cells: Dict[int, int] = field(default_factory=dict)  # For Bastion skill: cell_idx -> rounds_remaining
    morphed_skill: Optional[SkillName] = None  # For Morph skill: temporarily copied skill


@dataclass
class GameState:
    config: GameConfig
    nodes: List[SpinNode]                    # length = N
    edges: List[Tuple[SpinNode, SpinNode]]   # length = M
    edge_index: Dict[Tuple[int, int], int]   # map (i, j) -> edge idx in edges/J
    biases: jnp.ndarray                      # shape (N,)
    couplings: jnp.ndarray                   # shape (M,)
    beta: jnp.ndarray                        # scalar
    last_samples: Optional[jnp.ndarray] = None   # (n_samples, N) if collected
    last_final_spins: Optional[jnp.ndarray] = None  # (N,)

    # Turn-based game state
    current_turn: int = 0  # 0 = planning phase, 1 = sampling phase, 2 = scoring phase
    current_round: int = 1
    player_a_budget: PlayerBudget = None  # type: ignore
    player_b_budget: PlayerBudget = None  # type: ignore
    player_a_ready: bool = False
    player_b_ready: bool = False
    player_a_wins: int = 0
    player_b_wins: int = 0
    max_rounds: int = 5

    # PHASE 2: Entrenchment mechanic - track consecutive rounds of control
    entrenchment: jnp.ndarray = None  # type: ignore  # shape (N,) - rounds of consecutive control
    previous_round_spins: Optional[jnp.ndarray] = None  # (N,) - spins from last round

    def __post_init__(self):
        if self.player_a_budget is None:
            self.player_a_budget = PlayerBudget()
        if self.player_b_budget is None:
            self.player_b_budget = PlayerBudget()
        if self.entrenchment is None:
            n_nodes = len(self.nodes)
            self.entrenchment = jnp.zeros((n_nodes,), dtype=jnp.int32)


# -------------------------
# Helpers to build the grid
# -------------------------

def _grid_idx(row: int, col: int, grid_size: int) -> int:
    """Map (row, col) -> flat index."""
    return row * grid_size + col


def _build_nodes(grid_size: int) -> List[SpinNode]:
    """Create one SpinNode per grid cell."""
    n = grid_size * grid_size
    return [SpinNode() for _ in range(n)]


def _build_grid_edges(nodes: List[SpinNode], grid_size: int):
    """
    Build edges for a 2D grid: connect each cell to right and down neighbors.
    Returns:
        edges: list[(node_i, node_j)]
        edge_index: dict[(i, j)] -> edge_idx (i < j for consistency)
    """
    edges: List[Tuple[SpinNode, SpinNode]] = []
    edge_index: Dict[Tuple[int, int], int] = {}

    def add_edge(i: int, j: int):
        if i > j:
            i, j = j, i
        idx = len(edges)
        edges.append((nodes[i], nodes[j]))
        edge_index[(i, j)] = idx

    for r in range(grid_size):
        for c in range(grid_size):
            i = _grid_idx(r, c, grid_size)
            # right neighbor
            if c + 1 < grid_size:
                j = _grid_idx(r, c + 1, grid_size)
                add_edge(i, j)
            # down neighbor
            if r + 1 < grid_size:
                j = _grid_idx(r + 1, c, grid_size)
                add_edge(i, j)

    return edges, edge_index


def _build_checkerboard_blocks(nodes: List[SpinNode], grid_size: int):
    """
    Split nodes into two blocks (checkerboard) for block Gibbs updates.
    """
    even_nodes = []
    odd_nodes = []

    for r in range(grid_size):
        for c in range(grid_size):
            idx = _grid_idx(r, c, grid_size)
            if (r + c) % 2 == 0:
                even_nodes.append(nodes[idx])
            else:
                odd_nodes.append(nodes[idx])

    free_blocks = [Block(even_nodes), Block(odd_nodes)]
    all_nodes_block = Block(nodes)
    return free_blocks, all_nodes_block


# -------------------------
# Public API
# -------------------------
def create_game(
    config: Optional[GameConfig] = None,
    player_a_class: Optional[PlayerClass] = None,
    player_b_class: Optional[PlayerClass] = None
) -> GameState:
    """
    Initialize a new game with a grid of SpinNodes and default couplings/biases.

    Args:
        config: Game configuration
        player_a_class: Player A's chosen class (None for default/no class)
        player_b_class: Player B's chosen class (None for default/no class)
    """
    if config is None:
        config = GameConfig()

    nodes = _build_nodes(config.grid_size)
    edges, edge_index = _build_grid_edges(nodes, config.grid_size)

    n_nodes = len(nodes)
    n_edges = len(edges)

    biases = jnp.zeros((n_nodes,), dtype=jnp.float32)
    couplings = jnp.ones((n_edges,), dtype=jnp.float32) * config.base_coupling
    beta = jnp.array(config.base_beta, dtype=jnp.float32)

    # CLASS SYSTEM: Initialize player budgets with class-specific resources
    if player_a_class:
        a_bias, a_edge = get_starting_budget(player_a_class)
        player_a_budget = PlayerBudget(
            bias_tokens=a_bias,
            edge_tokens=a_edge,
            player_class=player_a_class,
            skill_cooldowns=initialize_skill_cooldowns(player_a_class.value if player_a_class else None)
        )
    else:
        player_a_budget = PlayerBudget()

    if player_b_class:
        b_bias, b_edge = get_starting_budget(player_b_class)
        player_b_budget = PlayerBudget(
            bias_tokens=b_bias,
            edge_tokens=b_edge,
            player_class=player_b_class,
            skill_cooldowns=initialize_skill_cooldowns(player_b_class.value if player_b_class else None)
        )
    else:
        player_b_budget = PlayerBudget()

    game = GameState(
        config=config,
        nodes=nodes,
        edges=edges,
        edge_index=edge_index,
        biases=biases,
        couplings=couplings,
        beta=beta,
        player_a_budget=player_a_budget,
        player_b_budget=player_b_budget,
    )

    # CARD SYSTEM: Deal initial cards to both players (class-weighted if class is chosen)
    if player_a_class:
        deal_class_cards(game.player_a_budget, player_a_class, num_cards=5)
    else:
        deal_cards_to_player(game.player_a_budget, num_cards=5)

    if player_b_class:
        deal_class_cards(game.player_b_budget, player_b_class, num_cards=5)
    else:
        deal_cards_to_player(game.player_b_budget, num_cards=5)

    return game


def apply_bias(game: GameState, row: int, col: int, direction: int, player: str = 'A'):
    """
    Modify bias h_i at (row, col).
    direction = +1  -> push toward Player A (+1 spin)
    direction = -1  -> push toward Player B (-1 spin)
    player: 'A' or 'B' - which player is making the move
    """
    # Check if player has tokens
    budget = game.player_a_budget if player == 'A' else game.player_b_budget
    if budget.bias_tokens_used >= budget.bias_tokens:
        raise ValueError(f"Player {player} has no bias tokens remaining")

    idx = _grid_idx(row, col, game.config.grid_size)
    delta = direction * game.config.bias_step
    # JAX arrays are immutable, so we create a new one
    game.biases = game.biases.at[idx].add(delta)

    # Consume a token
    budget.bias_tokens_used += 1


def apply_edge_change(game: GameState,
                      cell1: Tuple[int, int],
                      cell2: Tuple[int, int],
                      direction: int,
                      player: str = 'A'):
    """
    Modify coupling J_ij between two neighboring cells (row1, col1) and (row2, col2).
    direction = +1 -> strengthen alignment (cooperation)
    direction = -1 -> weaken alignment (or even anti-align if you go negative)
    player: 'A' or 'B' - which player is making the move
    """
    # Check if player has tokens
    budget = game.player_a_budget if player == 'A' else game.player_b_budget
    if budget.edge_tokens_used >= budget.edge_tokens:
        raise ValueError(f"Player {player} has no edge tokens remaining")

    r1, c1 = cell1
    r2, c2 = cell2
    i = _grid_idx(r1, c1, game.config.grid_size)
    j = _grid_idx(r2, c2, game.config.grid_size)
    if i > j:
        i, j = j, i

    key = (i, j)
    edge_idx = game.edge_index.get(key)
    if edge_idx is None:
        raise ValueError(f"No edge between {cell1} and {cell2} (must be neighbors).")

    delta = direction * game.config.coupling_step
    game.couplings = game.couplings.at[edge_idx].add(delta)

    # Consume a token
    budget.edge_tokens_used += 1


def set_beta(game: GameState, beta_value: float):
    """Set the inverse temperature β (higher => less randomness)."""
    game.beta = jnp.array(beta_value, dtype=jnp.float32)


def calculate_energy(game: GameState, spins: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the energy of the Ising model for a given spin configuration.

    Energy E = -sum_i(h_i * s_i) - sum_{i<j}(J_ij * s_i * s_j)

    where:
        h_i are the biases
        J_ij are the couplings
        s_i are the spins (+1 or -1)
    """
    # Bias term: -sum_i(h_i * s_i)
    bias_energy = -jnp.sum(game.biases * spins)

    # Coupling term: -sum_{i<j}(J_ij * s_i * s_j)
    coupling_energy = 0.0
    for edge_idx, (node_i, node_j) in enumerate(game.edges):
        # Get node indices from the nodes list
        i = game.nodes.index(node_i)
        j = game.nodes.index(node_j)
        coupling_energy -= game.couplings[edge_idx] * spins[i] * spins[j]

    total_energy = bias_energy + coupling_energy
    return total_energy


def run_sampling(game: GameState,
                 rng_key: jax.Array,
                 n_warmup: int = 100,
                 n_samples: int = 50,
                 steps_per_sample: int = 2):
    """
    Run THRML sampling for the current game parameters and return:

        final_board: jnp.ndarray, shape (grid_size, grid_size), values in {+1, -1}
        samples: jnp.ndarray, shape (n_samples, N) or similar (depending on THRML API)

    NOTE: Exact sample structure may vary with THRML versions; you may need
    to adapt the unpacking logic based on what `sample_states` returns.
    """
    grid_size = game.config.grid_size
    nodes = game.nodes
    edges = game.edges

    # (Re)build Ising model with current biases, couplings, beta
    model = IsingEBM(
        nodes=nodes,
        edges=edges,
        biases=game.biases,
        weights=game.couplings,
        beta=game.beta,
    )

    # Build blocks
    free_blocks, all_nodes_block = _build_checkerboard_blocks(nodes, grid_size)

    # Build sampling program
    program = IsingSamplingProgram(
        ebm=model,
        free_blocks=free_blocks,
        clamped_blocks=[],
    )

    # Initialize state
    key_init, key_samp = jax.random.split(rng_key, 2)
    init_state = hinton_init(key_init, model, free_blocks, ())

    # Schedule
    schedule = SamplingSchedule(
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=steps_per_sample,
    )

    # Run sampling
    # sample_states(key, program, schedule, init_state_free, state_clamp, nodes_to_sample)
    samples = sample_states(
        key_samp,
        program,
        schedule,
        init_state,
        [],  # state_clamp - empty list for clamped blocks
        [all_nodes_block],  # nodes_to_sample
    )

    # For many examples, `samples` is an array of shape (n_samples, N),
    # with spins as booleans. If it's nested, you may need to index into it.
    # Here we assume it's a simple array or the first element is.
    if isinstance(samples, (list, tuple)):
        samples_array = samples[0]
    else:
        samples_array = samples

    # The samples come back as shape (n_samples, N) where the first index is for
    # each block being sampled. Since we're sampling [all_nodes_block], we need samples_array[0]
    if samples_array.ndim == 3:
        # If shape is (1, n_samples, N), take the first element
        samples_array = samples_array[0]

    # Convert boolean samples to spin values: True -> +1, False -> -1
    spin_samples = jnp.where(samples_array, 1.0, -1.0)  # shape (n_samples, N)

    # Aggregate samples: majority vote per node
    # Count positive votes (spins = +1) for each node
    positive_votes = jnp.sum(spin_samples > 0, axis=0)  # shape (N,)
    negative_votes = jnp.sum(spin_samples < 0, axis=0)  # shape (N,)

    # Majority vote: if more positive votes, assign +1; otherwise -1
    final_spins = jnp.where(positive_votes > negative_votes, 1.0, -1.0)

    # For ties, use the last sample's value to break the tie
    ties = (positive_votes == negative_votes)
    final_spins = jnp.where(ties, spin_samples[-1], final_spins)

    final_board = final_spins.reshape((grid_size, grid_size))

    # Store diagnostics on game state (store as spin values, not booleans)
    game.last_samples = spin_samples
    game.last_final_spins = final_spins

    return final_board, spin_samples


def set_player_ready(game: GameState, player: str, ready: bool = True):
    """Mark a player as ready or not ready."""
    if player == 'A':
        game.player_a_ready = ready
    elif player == 'B':
        game.player_b_ready = ready
    else:
        raise ValueError(f"Invalid player: {player}")


def both_players_ready(game: GameState) -> bool:
    """Check if both players are ready."""
    return game.player_a_ready and game.player_b_ready


def update_entrenchment(game: GameState):
    """
    PHASE 2: Update entrenchment levels based on consecutive rounds of control.

    Entrenchment rewards territorial stability:
    - 0 rounds: no bonus
    - 1 round: "Contested" (no bonus yet)
    - 2 rounds: "Controlled" (+0.3 bias bonus)
    - 3+ rounds: "Entrenched" (+0.6 bias bonus, capped)
    """
    if game.last_final_spins is None:
        return

    # Update entrenchment counters
    if game.previous_round_spins is not None:
        # Check which cells maintained their spin from previous round
        maintained = (game.last_final_spins == game.previous_round_spins)

        # Increment entrenchment for maintained cells, reset for flipped cells
        game.entrenchment = jnp.where(
            maintained,
            game.entrenchment + 1,
            0
        )

    # Apply automatic bias bonuses based on entrenchment
    # Bonus = min(entrenchment * 0.3, 0.6)
    auto_bias = jnp.minimum(game.entrenchment * 0.3, 0.6)

    # Apply bonus in the direction of the current spin
    # Player A (+1) gets positive bias, Player B (-1) gets negative bias
    entrenchment_bias = auto_bias * game.last_final_spins

    # Add entrenchment bias to existing biases
    game.biases = game.biases + entrenchment_bias

    # Store current spins for next round's comparison
    game.previous_round_spins = game.last_final_spins.copy()


def reset_round(game: GameState):
    """Reset for a new round: decay biases and reset budgets, but keep game progress."""
    # PHASE 2: Update entrenchment BEFORE decaying biases
    update_entrenchment(game)

    # REDESIGN: Decay biases instead of resetting to zero
    # This creates strategic continuity - your previous work still matters!
    game.biases = game.biases * game.config.bias_decay_rate

    # PHASE 2: Dynamic token income based on territory control
    # CLASS SYSTEM: Store previous classes before resetting budgets
    prev_a_class = game.player_a_budget.player_class if game.player_a_budget else None
    prev_b_class = game.player_b_budget.player_class if game.player_b_budget else None

    if game.last_final_spins is not None:
        player_a_cells = int(jnp.sum(game.last_final_spins == 1))
        player_b_cells = int(jnp.sum(game.last_final_spins == -1))

        # Base tokens (use class-specific if available)
        if prev_a_class:
            a_bias_base, a_edge_base = get_starting_budget(prev_a_class)
        else:
            a_bias_base, a_edge_base = 2, 2

        if prev_b_class:
            b_bias_base, b_edge_base = get_starting_budget(prev_b_class)
        else:
            b_bias_base, b_edge_base = 2, 2

        # Bonus: +1 token per 5 cells controlled
        a_territory_bonus = player_a_cells // 5
        b_territory_bonus = player_b_cells // 5

        # Entrenchment bonus: +1 token if 3+ entrenched cells
        a_entrenched = int(jnp.sum((game.entrenchment >= 3) & (game.last_final_spins == 1)))
        b_entrenched = int(jnp.sum((game.entrenchment >= 3) & (game.last_final_spins == -1)))
        a_entrenchment_bonus = 1 if a_entrenched >= 3 else 0
        b_entrenchment_bonus = 1 if b_entrenched >= 3 else 0

        # Set new budgets
        game.player_a_budget = PlayerBudget(
            edge_tokens=a_edge_base + a_territory_bonus + a_entrenchment_bonus,
            bias_tokens=a_bias_base,
            player_class=prev_a_class
        )
        game.player_b_budget = PlayerBudget(
            edge_tokens=b_edge_base + b_territory_bonus + b_entrenchment_bonus,
            bias_tokens=b_bias_base,
            player_class=prev_b_class
        )
    else:
        # First round - use class-specific budgets if available
        if prev_a_class:
            a_bias, a_edge = get_starting_budget(prev_a_class)
            game.player_a_budget = PlayerBudget(
                bias_tokens=a_bias, edge_tokens=a_edge, player_class=prev_a_class
            )
        else:
            game.player_a_budget = PlayerBudget()

        if prev_b_class:
            b_bias, b_edge = get_starting_budget(prev_b_class)
            game.player_b_budget = PlayerBudget(
                bias_tokens=b_bias, edge_tokens=b_edge, player_class=prev_b_class
            )
        else:
            game.player_b_budget = PlayerBudget()

    # CARD SYSTEM: Deal cards to both players (class-weighted if class is chosen)
    if prev_a_class:
        deal_class_cards(game.player_a_budget, prev_a_class, num_cards=5)
    else:
        deal_cards_to_player(game.player_a_budget, num_cards=5)

    if prev_b_class:
        deal_class_cards(game.player_b_budget, prev_b_class, num_cards=5)
    else:
        deal_cards_to_player(game.player_b_budget, num_cards=5)

    # Reset ready status
    game.player_a_ready = False
    game.player_b_ready = False

    # Clear last board state
    game.last_samples = None
    game.last_final_spins = None

    # Advance round
    game.current_round += 1


def score_round(game: GameState) -> str:
    """
    Score the current round based on territory control.
    Returns the winner: 'A', 'B', or 'tie'
    """
    if game.last_final_spins is None:
        raise ValueError("No sampling results to score. Run sampling first.")

    player_a_count = int(jnp.sum(game.last_final_spins == 1))
    player_b_count = int(jnp.sum(game.last_final_spins == -1))

    if player_a_count > player_b_count:
        game.player_a_wins += 1
        return 'A'
    elif player_b_count > player_a_count:
        game.player_b_wins += 1
        return 'B'
    else:
        return 'tie'


def check_game_winner(game: GameState) -> Optional[str]:
    """
    Check if the game is over and return the winner.
    Returns 'A', 'B', or None if game is not over.
    """
    if game.player_a_wins > game.max_rounds // 2:
        return 'A'
    elif game.player_b_wins > game.max_rounds // 2:
        return 'B'
    elif game.current_round > game.max_rounds:
        # All rounds played, highest score wins
        if game.player_a_wins > game.player_b_wins:
            return 'A'
        elif game.player_b_wins > game.player_a_wins:
            return 'B'
        else:
            return 'tie'
    return None


# Card functions are now imported from cards.py


def get_probability_preview(game: GameState, rng_key: jax.Array, n_quick_samples: int = 10):
    """
    REDESIGN: Run a quick sampling preview to show players predicted outcomes.

    This helps players:
    - Learn cause-effect relationships
    - Make informed strategic decisions
    - Build intuition for the physics
    - Reduce frustration from unexpected outcomes

    Returns:
        probabilities: jnp.ndarray, shape (N,), values 0-1 (probability of being +1/Player A)
        predicted_counts: dict with 'A' and 'B' territory predictions
    """
    grid_size = game.config.grid_size
    nodes = game.nodes
    edges = game.edges

    # Build model with current state
    model = IsingEBM(
        nodes=nodes,
        edges=edges,
        biases=game.biases,
        weights=game.couplings,
        beta=game.beta,
    )

    # Build blocks
    free_blocks, all_nodes_block = _build_checkerboard_blocks(nodes, grid_size)

    # Build sampling program
    program = IsingSamplingProgram(
        ebm=model,
        free_blocks=free_blocks,
        clamped_blocks=[],
    )

    # Initialize state
    key_init, key_samp = jax.random.split(rng_key, 2)
    init_state = hinton_init(key_init, model, free_blocks, ())

    # Quick schedule with minimal warmup
    schedule = SamplingSchedule(
        n_warmup=20,  # Reduced warmup for speed
        n_samples=n_quick_samples,
        steps_per_sample=1,  # Minimal gap for speed
    )

    # Run quick sampling
    samples = sample_states(
        key_samp,
        program,
        schedule,
        init_state,
        [],
        [all_nodes_block],
    )

    # Process samples
    if isinstance(samples, (list, tuple)):
        samples_array = samples[0]
    else:
        samples_array = samples

    if samples_array.ndim == 3:
        samples_array = samples_array[0]

    # Convert to spins
    spin_samples = jnp.where(samples_array, 1.0, -1.0)

    # Calculate per-cell probabilities (mean spin value, normalized to 0-1)
    mean_spins = jnp.mean(spin_samples, axis=0)  # -1 to +1
    probabilities = (mean_spins + 1.0) / 2.0  # 0 to 1 (P of being +1/Player A)

    # Calculate predicted territory counts
    predicted_a = jnp.sum(mean_spins > 0)
    predicted_b = jnp.sum(mean_spins < 0)

    # Calculate standard deviation for confidence intervals
    std_spins = jnp.std(spin_samples, axis=0)

    return {
        'probabilities': probabilities,  # Per-cell probability of being Player A
        'mean_spins': mean_spins,  # Average spin value per cell
        'std_spins': std_spins,  # Standard deviation per cell
        'predicted_a_count': float(predicted_a),
        'predicted_b_count': float(predicted_b),
        'confidence': float(jnp.mean(jnp.abs(mean_spins))),  # Overall confidence (0-1)
    }
