"""
Skill system for Thermodynamic Tactics.

Each player class has 3 unique skills with round-based cooldowns.
Skills are special abilities that complement the token-based bias/edge system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Optional, List
import jax.numpy as jnp


class SkillName(str, Enum):
    """All available skills in the game."""
    # Infiltrator skills
    DEEP_STRIKE = "deep_strike"
    SURGICAL_CUT = "surgical_cut"
    MOMENTUM = "momentum"

    # Fortress skills
    REINFORCE = "reinforce"
    BASTION = "bastion"
    ENTRENCH = "entrench"

    # Manipulator skills
    CASCADE = "cascade"
    FEEDBACK_LOOP = "feedback_loop"
    EXPLOIT = "exploit"

    # Wildcard skills
    HEAT_WAVE = "heat_wave"
    POLARITY_FLIP = "polarity_flip"
    GAMBIT = "gambit"

    # Hybrid skills
    MORPH = "morph"
    BALANCE_SHIFT = "balance_shift"
    ADAPT = "adapt"


@dataclass
class Skill:
    """Definition of a skill."""
    name: SkillName
    display_name: str
    description: str
    cooldown: int  # Number of rounds before skill can be used again
    requires_target: bool  # Whether skill needs a grid location
    icon: str  # Emoji or icon identifier


@dataclass
class SkillState:
    """Tracks when a skill was last used and when it's available again."""
    last_used_round: int  # -1 if never used, otherwise round number when used

    def is_available(self, current_round: int, cooldown: int) -> bool:
        """Check if skill is off cooldown."""
        if self.last_used_round == -1:
            return True
        return (current_round - self.last_used_round) >= cooldown

    def rounds_until_ready(self, current_round: int, cooldown: int) -> int:
        """Returns number of rounds until skill is ready (0 if ready now)."""
        if self.is_available(current_round, cooldown):
            return 0
        return cooldown - (current_round - self.last_used_round)


# Skill definitions
SKILLS: Dict[SkillName, Skill] = {
    # Infiltrator
    SkillName.DEEP_STRIKE: Skill(
        name=SkillName.DEEP_STRIKE,
        display_name="Deep Strike",
        description="Apply massive bias (+2.0) to a single cell deep in enemy territory",
        cooldown=3,
        requires_target=True,
        icon="🗡️"
    ),
    SkillName.SURGICAL_CUT: Skill(
        name=SkillName.SURGICAL_CUT,
        display_name="Surgical Cut",
        description="Sever all edges around a target cell (set J=0), isolating it",
        cooldown=2,
        requires_target=True,
        icon="✂️"
    ),
    SkillName.MOMENTUM: Skill(
        name=SkillName.MOMENTUM,
        display_name="Momentum",
        description="Gain +1 bias token if you control >50% of the board",
        cooldown=1,
        requires_target=False,
        icon="⚡"
    ),

    # Fortress
    SkillName.REINFORCE: Skill(
        name=SkillName.REINFORCE,
        display_name="Reinforce",
        description="Strengthen all edges in a 3x3 region (+0.5 coupling)",
        cooldown=3,
        requires_target=True,
        icon="🛡️"
    ),
    SkillName.BASTION: Skill(
        name=SkillName.BASTION,
        display_name="Bastion",
        description="Lock a cell's spin for 2 rounds (immune to sampling changes)",
        cooldown=2,
        requires_target=True,
        icon="🏰"
    ),
    SkillName.ENTRENCH: Skill(
        name=SkillName.ENTRENCH,
        display_name="Entrench",
        description="Gain +2 edge tokens if you have 3+ entrenched cells",
        cooldown=1,
        requires_target=False,
        icon="⚓"
    ),

    # Manipulator
    SkillName.CASCADE: Skill(
        name=SkillName.CASCADE,
        display_name="Cascade",
        description="Apply bias to a line of 5 cells (diminishing: 1.0→0.2)",
        cooldown=3,
        requires_target=True,
        icon="🎭"
    ),
    SkillName.FEEDBACK_LOOP: Skill(
        name=SkillName.FEEDBACK_LOOP,
        display_name="Feedback Loop",
        description="Create a 2x2 region with all edges set to +1.5 (strong alignment)",
        cooldown=2,
        requires_target=True,
        icon="🔄"
    ),
    SkillName.EXPLOIT: Skill(
        name=SkillName.EXPLOIT,
        display_name="Exploit",
        description="Steal 1 edge token from opponent if they have >3 tokens",
        cooldown=1,
        requires_target=False,
        icon="🎯"
    ),

    # Wildcard
    SkillName.HEAT_WAVE: Skill(
        name=SkillName.HEAT_WAVE,
        display_name="Heat Wave",
        description="Reduce beta by 1.5 in a 3x3 region (more randomness)",
        cooldown=3,
        requires_target=True,
        icon="🔥"
    ),
    SkillName.POLARITY_FLIP: Skill(
        name=SkillName.POLARITY_FLIP,
        display_name="Polarity Flip",
        description="Invert all couplings in a 2x2 region (J → -J)",
        cooldown=2,
        requires_target=True,
        icon="🔀"
    ),
    SkillName.GAMBIT: Skill(
        name=SkillName.GAMBIT,
        display_name="Gambit",
        description="Randomly gain either +2 bias tokens OR +2 edge tokens",
        cooldown=1,
        requires_target=False,
        icon="🎲"
    ),

    # Hybrid
    SkillName.MORPH: Skill(
        name=SkillName.MORPH,
        display_name="Morph",
        description="Copy one skill from opponent's class (one-time use this round)",
        cooldown=3,
        requires_target=False,
        icon="🔄"
    ),
    SkillName.BALANCE_SHIFT: Skill(
        name=SkillName.BALANCE_SHIFT,
        display_name="Balance Shift",
        description="Convert up to 2 bias tokens ↔ edge tokens",
        cooldown=2,
        requires_target=False,
        icon="⚖️"
    ),
    SkillName.ADAPT: Skill(
        name=SkillName.ADAPT,
        display_name="Adapt",
        description="Reduce cooldown of your other skills by 1 round each",
        cooldown=1,
        requires_target=False,
        icon="🌀"
    ),
}


# Class-to-skills mapping
CLASS_SKILLS: Dict[str, List[SkillName]] = {
    "infiltrator": [SkillName.DEEP_STRIKE, SkillName.SURGICAL_CUT, SkillName.MOMENTUM],
    "fortress": [SkillName.REINFORCE, SkillName.BASTION, SkillName.ENTRENCH],
    "manipulator": [SkillName.CASCADE, SkillName.FEEDBACK_LOOP, SkillName.EXPLOIT],
    "wildcard": [SkillName.HEAT_WAVE, SkillName.POLARITY_FLIP, SkillName.GAMBIT],
    "hybrid": [SkillName.MORPH, SkillName.BALANCE_SHIFT, SkillName.ADAPT],
}


def get_class_skills(player_class: Optional[str]) -> List[Skill]:
    """Get the 3 skills for a given player class."""
    if player_class is None or player_class not in CLASS_SKILLS:
        return []
    return [SKILLS[skill_name] for skill_name in CLASS_SKILLS[player_class]]


def initialize_skill_cooldowns(player_class: Optional[str]) -> Dict[SkillName, SkillState]:
    """Initialize cooldown tracking for a player's skills."""
    cooldowns = {}
    for skill_name in CLASS_SKILLS.get(player_class, []):
        cooldowns[skill_name] = SkillState(last_used_round=-1)
    return cooldowns


def use_skill(skill_name: SkillName, current_round: int, cooldowns: Dict[SkillName, SkillState]) -> bool:
    """
    Mark a skill as used. Returns True if skill was available, False if on cooldown.
    """
    if skill_name not in cooldowns:
        return False

    skill = SKILLS[skill_name]
    state = cooldowns[skill_name]

    if not state.is_available(current_round, skill.cooldown):
        return False

    # Mark as used
    cooldowns[skill_name] = SkillState(last_used_round=current_round)
    return True


def get_cooldown_status(player_class: Optional[str], current_round: int,
                       cooldowns: Dict[SkillName, SkillState]) -> Dict[str, dict]:
    """
    Get status of all skills for a player.
    Returns dict mapping skill name to {available: bool, rounds_until_ready: int}
    """
    status = {}
    for skill_name in CLASS_SKILLS.get(player_class, []):
        skill = SKILLS[skill_name]
        state = cooldowns.get(skill_name, SkillState(last_used_round=-1))
        status[skill_name.value] = {
            "available": state.is_available(current_round, skill.cooldown),
            "rounds_until_ready": state.rounds_until_ready(current_round, skill.cooldown),
            "cooldown": skill.cooldown,
        }
    return status
