# energy_battle_backend.py

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import jax
import jax.numpy as jnp

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


# -------------------------
# Config + State containers
# -------------------------
@dataclass
class GameConfig:
    grid_size: int = 5
    base_coupling: float = 0.5
    base_beta: float = 1.0
    bias_step: float = 0.5       # how much each bias click changes h_i
    coupling_step: float = 0.25  # how much each edge click changes J_ij


@dataclass
class PlayerBudget:
    edge_tokens: int = 3
    bias_tokens: int = 2
    edge_tokens_used: int = 0
    bias_tokens_used: int = 0


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

    def __post_init__(self):
        if self.player_a_budget is None:
            self.player_a_budget = PlayerBudget()
        if self.player_b_budget is None:
            self.player_b_budget = PlayerBudget()


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
def create_game(config: Optional[GameConfig] = None) -> GameState:
    """
    Initialize a new game with a grid of SpinNodes and default couplings/biases.
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

    return GameState(
        config=config,
        nodes=nodes,
        edges=edges,
        edge_index=edge_index,
        biases=biases,
        couplings=couplings,
        beta=beta,
    )


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


def reset_round(game: GameState):
    """Reset for a new round: reset biases and budgets, but keep game progress."""
    # Reset biases to zero
    game.biases = jnp.zeros_like(game.biases)

    # Reset budgets
    game.player_a_budget = PlayerBudget()
    game.player_b_budget = PlayerBudget()

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
