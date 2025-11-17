# player_classes.py
# Player class system for Thermodynamic Tactics

from dataclasses import dataclass
from enum import Enum
from typing import List, TYPE_CHECKING

from .cards import CardType

if TYPE_CHECKING:
    from .game import PlayerBudget


# -------------------------
# Player Class System
# -------------------------
class PlayerClass(str, Enum):
    """Player classes with unique card loadouts and starting resources"""
    INFILTRATOR = "infiltrator"    # Aggressive - bias-focused, penetrate enemy lines
    FORTRESS = "fortress"          # Defensive - edge-focused, hold territory
    MANIPULATOR = "manipulator"    # Controller - balanced, area control
    WILDCARD = "wildcard"          # Chaos - special cards, unpredictable
    HYBRID = "hybrid"              # Balanced - versatile, adaptable


@dataclass
class ClassDefinition:
    """Defines a player class with its unique properties"""
    class_type: PlayerClass
    name: str
    description: str
    # Starting resources
    base_bias_tokens: int
    base_edge_tokens: int
    # Card loadout weights (higher = more likely to draw)
    card_weights: dict[CardType, float]
    # Passive ability description
    passive_ability: str
    # Visual/flavor
    icon: str  # Emoji or symbol
    color_scheme: str  # For UI theming


# -------------------------
# Class Definitions
# -------------------------
CLASS_DEFINITIONS = {
    PlayerClass.INFILTRATOR: ClassDefinition(
        class_type=PlayerClass.INFILTRATOR,
        name="Infiltrator",
        description="Aggressive offensive specialist. Excels at penetrating enemy territory with powerful bias attacks.",
        base_bias_tokens=4,  # +2 bias tokens
        base_edge_tokens=2,  # -1 edge tokens
        card_weights={
            CardType.INFILTRATE: 3.0,    # 3x more likely
            CardType.ANCHOR: 2.0,         # 2x more likely
            CardType.DISRUPTION: 1.5,     # 1.5x more likely
            CardType.FORTRESS: 0.5,       # Half as likely
            CardType.HEAT_WAVE: 1.2,      # Slightly more
            CardType.FREEZE: 0.8,         # Slightly less
        },
        passive_ability="Bias Mastery: Bias actions cost 25% less tokens (rounded down, min 1)",
        icon="⚔️",
        color_scheme="#e63946"  # Red
    ),

    PlayerClass.FORTRESS: ClassDefinition(
        class_type=PlayerClass.FORTRESS,
        name="Fortress",
        description="Defensive specialist. Masters edge manipulation to create impenetrable walls and hold territory.",
        base_bias_tokens=1,  # -1 bias tokens
        base_edge_tokens=5,  # +2 edge tokens
        card_weights={
            CardType.FORTRESS: 3.0,       # 3x more likely
            CardType.ANCHOR: 2.0,         # 2x more likely
            CardType.FREEZE: 2.0,         # 2x more likely (lock down)
            CardType.INFILTRATE: 0.5,     # Half as likely
            CardType.DISRUPTION: 1.5,     # Moderate
            CardType.HEAT_WAVE: 0.5,      # Half as likely
        },
        passive_ability="Fortification: Edge coupling changes are 25% stronger (+0.3125 instead of +0.25)",
        icon="🛡️",
        color_scheme="#457b9d"  # Blue
    ),

    PlayerClass.MANIPULATOR: ClassDefinition(
        class_type=PlayerClass.MANIPULATOR,
        name="Manipulator",
        description="Control specialist. Balances bias and edges to dominate large areas of the battlefield.",
        base_bias_tokens=3,  # +1 bias token
        base_edge_tokens=3,  # Standard edge tokens
        card_weights={
            CardType.ANCHOR: 2.5,         # Strongly favored
            CardType.FORTRESS: 1.8,       # Favored
            CardType.INFILTRATE: 1.8,     # Favored
            CardType.DISRUPTION: 1.5,     # Moderate
            CardType.HEAT_WAVE: 1.2,      # Slightly more
            CardType.FREEZE: 1.2,         # Slightly more
        },
        passive_ability="Territorial Control: +1 bonus token when controlling 10+ cells (can be bias OR edge)",
        icon="🎯",
        color_scheme="#2a9d8f"  # Teal
    ),

    PlayerClass.WILDCARD: ClassDefinition(
        class_type=PlayerClass.WILDCARD,
        name="Wildcard",
        description="Chaos specialist. Unpredictable and dangerous, manipulates temperature to create explosive plays.",
        base_bias_tokens=2,  # Standard bias
        base_edge_tokens=3,  # Standard edge
        card_weights={
            CardType.HEAT_WAVE: 3.0,      # 3x more likely
            CardType.FREEZE: 3.0,         # 3x more likely
            CardType.DISRUPTION: 2.0,     # 2x more likely
            CardType.INFILTRATE: 1.5,     # Moderate
            CardType.FORTRESS: 0.8,       # Slightly less
            CardType.ANCHOR: 0.8,         # Slightly less
        },
        passive_ability="Chaos Theory: Heat Wave and Freeze cards cost 50% less tokens and have 50% stronger effects",
        icon="🎲",
        color_scheme="#f4a261"  # Orange
    ),

    PlayerClass.HYBRID: ClassDefinition(
        class_type=PlayerClass.HYBRID,
        name="Hybrid",
        description="Versatile all-rounder. Balanced resources and card access allow adaptation to any situation.",
        base_bias_tokens=3,  # +1 bias
        base_edge_tokens=3,  # Standard edge
        card_weights={
            CardType.INFILTRATE: 1.0,     # Equal chance
            CardType.DISRUPTION: 1.0,     # Equal chance
            CardType.FORTRESS: 1.0,       # Equal chance
            CardType.ANCHOR: 1.0,         # Equal chance
            CardType.HEAT_WAVE: 1.0,      # Equal chance
            CardType.FREEZE: 1.0,         # Equal chance
        },
        passive_ability="Adaptability: Can redraw 1 card per round. Gains +1 token of choice when behind by 5+ cells",
        icon="⚖️",
        color_scheme="#9d4edd"  # Purple
    ),
}


# -------------------------
# Helper Functions
# -------------------------
def get_class_definition(class_type: PlayerClass) -> ClassDefinition:
    """Get the definition for a player class"""
    return CLASS_DEFINITIONS[class_type]


def get_starting_budget(class_type: PlayerClass) -> tuple[int, int]:
    """
    Get starting bias and edge tokens for a class.

    Returns:
        (bias_tokens, edge_tokens)
    """
    class_def = get_class_definition(class_type)
    return class_def.base_bias_tokens, class_def.base_edge_tokens


def deal_class_cards(budget: 'PlayerBudget', class_type: PlayerClass, num_cards: int = 5):
    """
    Deal cards to a player based on their class weights.

    Uses weighted random selection to favor cards that match the class's playstyle.
    """
    import random

    class_def = get_class_definition(class_type)

    # Create weighted card pool
    card_pool = []
    for card_type in CardType:
        weight = class_def.card_weights.get(card_type, 1.0)
        # Add card multiple times based on weight (convert float to int counts)
        count = max(1, int(weight * 10))  # Scale up for better granularity
        card_pool.extend([card_type] * count)

    # Draw cards without replacement from the weighted pool
    # Note: We allow duplicates in hand for strategic variety
    drawn_cards = random.choices(card_pool, k=num_cards)

    budget.hand = drawn_cards
    budget.played_cards = []


def apply_class_passive(
    class_type: PlayerClass,
    budget: 'PlayerBudget',
    action_type: str = None,
    territory_count: int = 0,
    cells_behind: int = 0
) -> dict:
    """
    Apply class passive ability effects.

    Args:
        class_type: The player's class
        budget: The player's budget to modify
        action_type: Type of action being performed ('bias', 'edge', 'heat_wave', 'freeze')
        territory_count: Number of cells controlled (for Manipulator)
        cells_behind: How many cells behind the player is (for Hybrid)

    Returns:
        dict with modification values to apply
    """
    modifications = {
        'bias_cost_multiplier': 1.0,
        'edge_cost_multiplier': 1.0,
        'coupling_strength_multiplier': 1.0,
        'beta_effect_multiplier': 1.0,
        'bonus_tokens': {'bias': 0, 'edge': 0},
    }

    if class_type == PlayerClass.INFILTRATOR:
        # Bias Mastery: Bias actions cost 25% less
        modifications['bias_cost_multiplier'] = 0.75

    elif class_type == PlayerClass.FORTRESS:
        # Fortification: Edge coupling changes are 25% stronger
        modifications['coupling_strength_multiplier'] = 1.25

    elif class_type == PlayerClass.MANIPULATOR:
        # Territorial Control: +1 token when controlling 10+ cells
        if territory_count >= 10:
            # Player can choose which type - for now, default to edge
            modifications['bonus_tokens']['edge'] = 1

    elif class_type == PlayerClass.WILDCARD:
        # Chaos Theory: Heat/Freeze cost 50% less and 50% stronger
        if action_type in ['heat_wave', 'freeze']:
            modifications['beta_effect_multiplier'] = 1.5
            # Note: Cost reduction handled in card cost calculation

    elif class_type == PlayerClass.HYBRID:
        # Adaptability: +1 token when behind by 5+ cells
        if cells_behind >= 5:
            # Player can choose - default to bias
            modifications['bonus_tokens']['bias'] = 1
        # Note: Card redraw handled separately in game logic

    return modifications


def get_all_classes() -> List[ClassDefinition]:
    """Get all available player classes"""
    return [CLASS_DEFINITIONS[cls] for cls in PlayerClass]
