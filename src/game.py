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


def apply_bias(game: GameState, row: int, col: int, direction: int):
    """
    Modify bias h_i at (row, col).
    direction = +1  -> push toward Player A (+1 spin)
    direction = -1  -> push toward Player B (-1 spin)
    """
    idx = _grid_idx(row, col, game.config.grid_size)
    delta = direction * game.config.bias_step
    # JAX arrays are immutable, so we create a new one
    game.biases = game.biases.at[idx].add(delta)


def apply_edge_change(game: GameState,
                      cell1: Tuple[int, int],
                      cell2: Tuple[int, int],
                      direction: int):
    """
    Modify coupling J_ij between two neighboring cells (row1, col1) and (row2, col2).
    direction = +1 -> strengthen alignment (cooperation)
    direction = -1 -> weaken alignment (or even anti-align if you go negative)
    """
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


def set_beta(game: GameState, beta_value: float):
    """Set the inverse temperature β (higher => less randomness)."""
    game.beta = jnp.array(beta_value, dtype=jnp.float32)


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
        model=model,
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
    # Depending on THRML version, sample_states may return a PyTree; you can
    # inspect it once and then adjust this unpacking.
    samples = sample_states(
        key_samp,
        program,
        schedule,
        init_state,
        observers=[],
        observed_blocks=[all_nodes_block],
    )

    # For many examples, `samples` is an array of shape (n_samples, N),
    # with spins in {+1, -1}. If it’s nested, you may need to index into it.
    # Here we assume it's a simple array or the first element is.
    if isinstance(samples, (list, tuple)):
        samples_array = samples[0]
    else:
        samples_array = samples

    # Aggregate samples: majority vote per node
    mean_spins = jnp.mean(samples_array, axis=0)  # shape (N,)
    final_spins = jnp.sign(mean_spins)
    # If any are exactly 0 (rare), treat them as +1 by default
    final_spins = jnp.where(final_spins == 0, 1.0, final_spins)

    final_board = final_spins.reshape((grid_size, grid_size))

    # Store diagnostics on game state
    game.last_samples = samples_array
    game.last_final_spins = final_spins

    return final_board, samples_array
