# cards.py
# Card system for Thermodynamic Tactics

from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING
from enum import Enum
import random

if TYPE_CHECKING:
    from .game import GameState, PlayerBudget


# -------------------------
# Card System
# -------------------------
class CardType(str, Enum):
    """Card types from the redesign proposal"""
    # Offensive cards
    INFILTRATE = "infiltrate"  # Apply strong bias to 3 adjacent cells in enemy territory
    DISRUPTION = "disruption"  # Weaken 4 edges in a 2x2 region

    # Defensive cards
    FORTRESS = "fortress"      # Strengthen edges in a 3x3 region
    ANCHOR = "anchor"          # Apply strong bias to 1 cell + 4 neighbors

    # Special cards
    HEAT_WAVE = "heat_wave"    # Reduce beta in a region (more chaos)
    FREEZE = "freeze"          # Increase beta in a region (lock state)


@dataclass
class Card:
    """Represents a playable action card"""
    card_type: CardType
    name: str
    description: str
    bias_cost: int  # Number of bias tokens consumed
    edge_cost: int  # Number of edge tokens consumed

    @staticmethod
    def get_card_definition(card_type: CardType) -> 'Card':
        """Factory method to create card instances from card type"""
        definitions = {
            CardType.INFILTRATE: Card(
                card_type=CardType.INFILTRATE,
                name="Infiltrate",
                description="Apply strong bias to 3 adjacent cells in enemy territory",
                bias_cost=2,
                edge_cost=0,
            ),
            CardType.DISRUPTION: Card(
                card_type=CardType.DISRUPTION,
                name="Disruption",
                description="Weaken 4 edges in a 2x2 region",
                bias_cost=0,
                edge_cost=3,
            ),
            CardType.FORTRESS: Card(
                card_type=CardType.FORTRESS,
                name="Fortress",
                description="Strengthen edges in a 3x3 region",
                bias_cost=0,
                edge_cost=3,
            ),
            CardType.ANCHOR: Card(
                card_type=CardType.ANCHOR,
                name="Anchor",
                description="Apply strong bias to 1 cell + 4 neighbors",
                bias_cost=2,
                edge_cost=0,
            ),
            CardType.HEAT_WAVE: Card(
                card_type=CardType.HEAT_WAVE,
                name="Heat Wave",
                description="Reduce beta in a region (more chaos)",
                bias_cost=0,
                edge_cost=0,
            ),
            CardType.FREEZE: Card(
                card_type=CardType.FREEZE,
                name="Freeze",
                description="Increase beta in a region (lock state)",
                bias_cost=0,
                edge_cost=0,
            ),
        }
        return definitions[card_type]


# -------------------------
# Helper function
# -------------------------
def _grid_idx(row: int, col: int, grid_size: int) -> int:
    """Map (row, col) -> flat index."""
    return row * grid_size + col


# -------------------------
# Card management functions
# -------------------------
def deal_cards_to_player(budget: 'PlayerBudget', num_cards: int = 5):
    """
    Deal random cards to a player's hand at the start of a round.

    According to the redesign:
    - Each player draws 5 random cards at round start
    - Players pick 2-3 cards to play (constrained by token budget)
    """
    all_card_types = list(CardType)
    budget.hand = random.sample(all_card_types, min(num_cards, len(all_card_types)))
    budget.played_cards = []


def can_play_card(game: 'GameState', player: str, card_type: CardType) -> Tuple[bool, str]:
    """
    Check if a player can play a card based on their budget and hand.

    Returns:
        (can_play, error_message)
    """
    budget = game.player_a_budget if player == 'A' else game.player_b_budget

    # Check if card is in hand
    if card_type not in budget.hand:
        return False, f"Card {card_type} not in hand"

    # Check if already played
    if card_type in budget.played_cards:
        return False, f"Card {card_type} already played this round"

    # Get card definition
    card = Card.get_card_definition(card_type)

    # Check if player has enough tokens
    bias_available = budget.bias_tokens - budget.bias_tokens_used
    edge_available = budget.edge_tokens - budget.edge_tokens_used

    if card.bias_cost > bias_available:
        return False, f"Not enough bias tokens (need {card.bias_cost}, have {bias_available})"

    if card.edge_cost > edge_available:
        return False, f"Not enough edge tokens (need {card.edge_cost}, have {edge_available})"

    return True, ""


def play_card(game: 'GameState', player: str, card_type: CardType, target_row: int, target_col: int):
    """
    Play a card at a target location on the grid.

    The card effects will be applied in the apply_card_effect function.
    This function just validates and tracks the card play.

    Args:
        game: GameState
        player: 'A' or 'B'
        card_type: Type of card to play
        target_row: Row coordinate of the target region center
        target_col: Column coordinate of the target region center
    """
    # Validate card can be played
    can_play, error = can_play_card(game, player, card_type)
    if not can_play:
        raise ValueError(error)

    # Validate target is in bounds
    if not (0 <= target_row < game.config.grid_size and 0 <= target_col < game.config.grid_size):
        raise ValueError(f"Target ({target_row}, {target_col}) out of bounds")

    budget = game.player_a_budget if player == 'A' else game.player_b_budget
    card = Card.get_card_definition(card_type)

    # Apply the card effect
    apply_card_effect(game, player, card_type, target_row, target_col)

    # Consume tokens
    budget.bias_tokens_used += card.bias_cost
    budget.edge_tokens_used += card.edge_cost

    # Mark card as played
    budget.played_cards.append(card_type)


def apply_card_effect(game: 'GameState', player: str, card_type: CardType, target_row: int, target_col: int):
    """
    Apply the effect of a card to the game state.

    Each card modifies biases and/or couplings in specific patterns:
    - INFILTRATE: Apply strong bias to 3 adjacent cells
    - DISRUPTION: Weaken 4 edges in a 2x2 region
    - FORTRESS: Strengthen edges in a 3x3 region
    - ANCHOR: Apply strong bias to 1 cell + 4 neighbors
    - HEAT_WAVE: Reduce beta in a region (more chaos) - NOT IMPLEMENTED YET (need regional beta)
    - FREEZE: Increase beta in a region (lock state) - NOT IMPLEMENTED YET (need regional beta)
    """
    grid_size = game.config.grid_size

    # Player direction: A = +1, B = -1
    direction = 1 if player == 'A' else -1

    if card_type == CardType.INFILTRATE:
        # Apply strong bias to 3 adjacent cells in a line
        # Pattern: apply to target and 2 neighbors (horizontal or vertical)
        cells_to_bias = []

        # Add target cell
        cells_to_bias.append((target_row, target_col))

        # Add horizontal neighbors if available
        if target_col + 1 < grid_size:
            cells_to_bias.append((target_row, target_col + 1))
        if target_col - 1 >= 0 and len(cells_to_bias) < 3:
            cells_to_bias.append((target_row, target_col - 1))

        # If we still need more cells, add vertical neighbors
        if len(cells_to_bias) < 3 and target_row + 1 < grid_size:
            cells_to_bias.append((target_row + 1, target_col))
        if len(cells_to_bias) < 3 and target_row - 1 >= 0:
            cells_to_bias.append((target_row - 1, target_col))

        # Apply strong bias (1.0) to each cell
        for row, col in cells_to_bias[:3]:
            idx = _grid_idx(row, col, grid_size)
            game.biases = game.biases.at[idx].add(direction * 1.0)

    elif card_type == CardType.DISRUPTION:
        # Weaken 4 edges in a 2x2 region
        # Pattern: affect edges in a 2x2 square centered at target
        for dr in range(2):
            for dc in range(2):
                r = target_row + dr
                c = target_col + dc
                if r < grid_size and c < grid_size:
                    i = _grid_idx(r, c, grid_size)

                    # Weaken right edge
                    if c + 1 < grid_size:
                        j = _grid_idx(r, c + 1, grid_size)
                        key = (min(i, j), max(i, j))
                        if key in game.edge_index:
                            edge_idx = game.edge_index[key]
                            game.couplings = game.couplings.at[edge_idx].add(-0.3)

                    # Weaken down edge
                    if r + 1 < grid_size:
                        j = _grid_idx(r + 1, c, grid_size)
                        key = (min(i, j), max(i, j))
                        if key in game.edge_index:
                            edge_idx = game.edge_index[key]
                            game.couplings = game.couplings.at[edge_idx].add(-0.3)

    elif card_type == CardType.FORTRESS:
        # Strengthen edges in a 3x3 region
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r = target_row + dr
                c = target_col + dc
                if 0 <= r < grid_size and 0 <= c < grid_size:
                    i = _grid_idx(r, c, grid_size)

                    # Strengthen right edge
                    if c + 1 < grid_size:
                        j = _grid_idx(r, c + 1, grid_size)
                        key = (min(i, j), max(i, j))
                        if key in game.edge_index:
                            edge_idx = game.edge_index[key]
                            game.couplings = game.couplings.at[edge_idx].add(0.3)

                    # Strengthen down edge
                    if r + 1 < grid_size:
                        j = _grid_idx(r + 1, c, grid_size)
                        key = (min(i, j), max(i, j))
                        if key in game.edge_index:
                            edge_idx = game.edge_index[key]
                            game.couplings = game.couplings.at[edge_idx].add(0.3)

    elif card_type == CardType.ANCHOR:
        # Apply strong bias to center cell + 4 orthogonal neighbors
        cells_to_bias = [(target_row, target_col)]

        # Add neighbors (up, down, left, right)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = target_row + dr, target_col + dc
            if 0 <= r < grid_size and 0 <= c < grid_size:
                cells_to_bias.append((r, c))

        # Apply strong bias (1.2) to center, moderate (0.6) to neighbors
        for i, (row, col) in enumerate(cells_to_bias):
            idx = _grid_idx(row, col, grid_size)
            bias_strength = 1.2 if i == 0 else 0.6
            game.biases = game.biases.at[idx].add(direction * bias_strength)

    elif card_type == CardType.HEAT_WAVE:
        # TODO: Implement regional beta variation
        # For now, slightly reduce global beta
        game.beta = game.beta * 0.9

    elif card_type == CardType.FREEZE:
        # TODO: Implement regional beta variation
        # For now, slightly increase global beta
        game.beta = game.beta * 1.1

    else:
        raise ValueError(f"Unknown card type: {card_type}")
