from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import random
import math
 
 
# Types / data structures
 
 
Vec2 = Tuple[int, int]
RobotId = str
TeamId = str
 
@dataclass
class LocalObservation:
    robot_id: RobotId
    position: Vec2
    nearby: List[str] = field(default_factory=list)
    teammates: List[Tuple[RobotId, Vec2]] = field(default_factory=list)
    hazards: List[str] = field(default_factory=list)
    messages_in: List[str] = field(default_factory=list)
 
@dataclass
class GlobalState:
    t: int
    robots: Dict[RobotId, Dict[str, Any]]  # {position: (x,y), hp: int, team: optional}
    map_summary: str
    history: Dict[str, Any] = field(default_factory=lambda: {
        "actions": [],           # [{t, robot_id, action}]
        "rewards": [],           # [{t, value, reason}]
        "human_constraints": [], # [str]
        "llm_summaries": []      # [str]
    })
 
@dataclass
class StrategyCandidate:
    text: str
    risks: List[str]
    uncertainties: List[str]
 
@dataclass
class HumanPlan:
    priorities: List[str]
    safety_constraints: List[str]
    mission_goals: List[str]
    approved_strategies: List[str]
 
@dataclass
class Subgoal:
    subgoal_id: str
    description: str
    vector: List[float]  # placeholder embedding/encoding
 
@dataclass
class HRLAction:
    teams: Dict[TeamId, List[RobotId]]
    subgoals: Dict[TeamId, Subgoal]
    request_replan: bool
 
@dataclass
class Reward:
    value: float
    breakdown: List[str]
 
@dataclass
class Config:
    num_robots: int = 12
    max_steps_per_episode: int = 50
    enable_llm: bool = True
    enable_human_intervention: bool = True
    log_every: int = 5
    seed: Optional[int] = 42
 
 
# Helpers
 
 
def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))
 
def encode_subgoal(desc: str) -> List[float]:
    # placeholder "vector" encoder (replace with embedding model)
    s = sum(ord(c) for c in desc)
    return [float(s % 97), float((s * 7) % 101), float((s * 13) % 103)]
 
def pick_direction_from_vector(v: List[float]) -> str:
    # deterministic primitive action from vector
    d = int(v[0]) % 4
    return ["MOVE_UP", "MOVE_RIGHT", "MOVE_DOWN", "MOVE_LEFT"][d]
 
 
# ==========================================================
# BLOCK 1: ENVIRONMENT INITIALIZATION
# ==========================================================
def block1_environment_initialization(cfg: Config) -> GlobalState:
    """
    Loads environment, spawns N robots, resets map, initializes buffers/logging.
    """
    if cfg.seed is not None:
        random.seed(cfg.seed)
 
    robots: Dict[RobotId, Dict[str, Any]] = {}
    for i in range(1, cfg.num_robots + 1):
        rid = f"R{i}"
        robots[rid] = {"position": (random.randint(0, 9), random.randint(0, 9)), "hp": 100}
 
    gs = GlobalState(
        t=0,
        robots=robots,
        map_summary="10x10 grid world; obstacles randomized; objective unknown",
    )
    return gs