from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
import math
import heapq
import os
import json
import re

try:
    import openai  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    openai = None

# -----------------------------
# Types / data structures
# -----------------------------
Cell = Tuple[int, int]
Vec2 = Tuple[int, int]
RobotId = str
TeamId = str

DISCRETE_ACTIONS = {
    "W": 0,
    "NW": 1,
    "N": 2,
    "NE": 3,
    "E": 4,
    "SE": 5,
    "S": 6,
    "SW": 7,
    "HOLD": 8,
}

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
class ParsedCommand:
    raw_text: str
    action: str                     # hold, attack, avoid, protect, regroup, line, spread
    scope: str                      # all, team, robot
    target: str                     # all / team id / robot id
    params: Dict[str, Any] = field(default_factory=dict)
 
@dataclass
class HumanPlan:
    priorities: List[str]
    safety_constraints: List[str]
    mission_goals: List[str]
    approved_strategies: List[str]
    parsed_commands: List[ParsedCommand] = field(default_factory=list)
    strategy_index: Optional[int] = None  # legacy: LLM candidate index when using old block5

@dataclass
class Subgoal:
    subgoal_id: str
    description: str
    vector: List[float]  # placeholder embedding/encoding
    role: str = "attack"  # "attack" | "defend" | "hold" — used by block8 for CTF goal

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
    max_steps_per_episode: int = 5000
    enable_llm: bool = True
    enable_human_intervention: bool = True
    log_every: int = 5
    seed: Optional[int] = 42


# -----------------------------
# Helpers
# -----------------------------

def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

def encode_subgoal(desc: str) -> List[float]:
    # placeholder "vector" encoder (replace with embedding model)
    s = sum(ord(c) for c in desc)
    return [float(s % 97), float((s * 7) % 101), float((s * 13) % 103)]

def pick_direction_from_vector(v: List[float]) -> str:
    # deterministic primitive action from vector
    d = int(v[0]) % 8
    return ["W", "NW", "N", "NE", "E", "SE", "S", "SW"][d]


# Grid dimensions matching pyquaticus world_size [120, 60] -> rows=60, cols=120 (row=y, col=x)
# Field: 120 m x 60 m. Agent (from config): radius 2.0 m, diameter 4.0 m. 1 grid cell = 1 m.
GRID_ROWS = 60
GRID_COLS = 120
# Boundary buffer in grid cells: keep paths at least this many cells from walls.
# Use 6 so defenders (and all agents) don't path along the bottom/top and hit walls (agent radius 2m).
PATH_BOUNDARY_BUFFER_CELLS = 6
AGENT_RADIUS_CELLS = 2  # kept for any other use; grid uses PATH_BOUNDARY_BUFFER_CELLS

# (drow, dcol) for 8 directions: N, NE, E, SE, S, SW, W, NW
_DIRECTION_TO_OFFSET = {
    "N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
    "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1),
}
_OFFSET_TO_DIRECTION = {v: k for k, v in _DIRECTION_TO_OFFSET.items()}


def _world_to_grid(x: float, y: float) -> Tuple[int, int]:
    """Convert world (x, y) to grid (row, col)."""
    row = clamp(int(round(y)), 0, GRID_ROWS - 1)
    col = clamp(int(round(x)), 0, GRID_COLS - 1)
    return (row, col)


def _path_step_to_direction(dr: int, dc: int) -> str:
    """Map (drow, dcol) from one path cell to next to direction string.
    Grid has row=y; pyquaticus uses heading 0 (N) = +y. So we flip dr so that
    increasing row (dr=1 = +y) maps to N (env +y), not S (env -y which sent defenders to bottom).
    """
    return _OFFSET_TO_DIRECTION.get((-dr, dc), "HOLD")


# ==========================================================
# BLOCK 1: ENVIRONMENT INITIALIZATION
# ==========================================================
def block1_environment_initialization(cfg: Config) -> GlobalState:
    """
    Load environment; spawn N robots; reset map, goals, flags; init buffers/logging.
    """
    if cfg.seed is not None:
        random.seed(cfg.seed)

    robots: Dict[RobotId, Dict[str, Any]] = {}
    # Random start positions for BLUE Team
    for i in range(1, (cfg.num_robots // 2) + 1):
        rid = f"R{i}"
        robots[rid] = {"position": (random.randint(60, 119), random.randint(0, 59)), "hp": 100}
    
    # Random start positions for RED Team
    for i in range((cfg.num_robots // 2) + 1, cfg.num_robots + 1):
        rid = f"R{i}"
        robots[rid] = {"position": (random.randint(0, 59), random.randint(0, 59)), "hp": 100}

    gs = GlobalState(
        t=0,
        robots=robots,
        map_summary="120x60 grid world; obstacles randomized; objective unknown",
    )
    return gs


# Thresholds for env-driven nearby/hazards (meters). PyQuaticus obs: wall_*_distance, (opponent_i, "distance").
WALL_NEAR_M = 20.0
WALL_HAZARD_M = 8.0
ENEMY_SPOTTED_M = 40.0


def block2_state_collection(
    gs: GlobalState,
    env_obs: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[LocalObservation]:
    """
    Each robot Ri collects local observation oi: nearby map/objects/enemies, teammate positions,
    local hazards. No full observability; all observations gathered centrally.
    If env_obs is provided (agent_id -> raw obs dict from env.state_to_obs(agent_id, normalize=False)),
    nearby and hazards are derived from env: walls (wall_*_distance), enemies (opponent_* distance),
    and red_zone when on_side is False (on enemy territory).
    """
    ids = list(gs.robots.keys())

    local_obs: List[LocalObservation] = []
    for rid in ids:
        pos = gs.robots[rid]["position"]
        teammates = [(tid, gs.robots[tid]["position"]) for tid in ids if tid != rid][:3]

        nearby: List[str] = []
        hazards: List[str] = []
        if env_obs is not None:
            agent_id: Optional[int] = None
            if rid.startswith("R") and rid[1:].isdigit():
                agent_id = int(rid[1:])
            if agent_id is not None and agent_id in env_obs:
                obs = env_obs[agent_id]
                wall_near_any = False
                wall_hazard_any = False
                for i in range(4):
                    d = obs.get(f"wall_{i}_distance", 1e9)
                    if isinstance(d, (int, float)):
                        if d < WALL_HAZARD_M:
                            wall_hazard_any = True
                        if d < WALL_NEAR_M:
                            wall_near_any = True
                if wall_hazard_any:
                    hazards.append("wall_close")
                if wall_near_any:
                    nearby.append("wall_near")
                for i in range(6):
                    d = obs.get((f"opponent_{i}", "distance"), 1e9)
                    if isinstance(d, (int, float)) and d < ENEMY_SPOTTED_M:
                        nearby.append("enemy_spotted")
                        break
                if not obs.get("on_side", True):
                    hazards.append("red_zone")
        if not nearby and not hazards:
            nearby = [x for x in ["wall", "crate", "enemy?"] if random.random() < 0.4]
            hazards = [x for x in ["red_zone", "minefield"] if random.random() < 0.2]

        local_obs.append(LocalObservation(
            robot_id=rid,
            position=pos,
            nearby=nearby,
            teammates=teammates,
            hazards=hazards,
            messages_in=[],
        ))

    return local_obs


# ==========================================================
# BLOCK 3: GLOBAL STATE ENCODING (CTDE)
# ==========================================================
def block3_global_state_encoding(gs: GlobalState, local_obs: List[LocalObservation]) -> GlobalState:
    """
    Combines all robot observations into global representation S_t.
    Includes: robot states 1..N, environment map summary, historical actions & rewards,
    previous human constraints, previous LLM suggestions. Used as input for strategy generation.
    """
    hazard_count = sum(len(o.hazards) for o in local_obs)
    enemy_signals = sum(1 for o in local_obs if "enemy_spotted" in o.nearby or "enemy?" in o.nearby)
    n_act = len(gs.history["actions"])
    n_rew = len(gs.history["rewards"])
    n_hc = len(gs.history["human_constraints"])
    n_llm = len(gs.history["llm_summaries"])
    # Single snapshot for this step (do not append to previous; avoids unbounded "history of summaries")
    gs.map_summary = (
        f"Step t={gs.t}. Hazards reported={hazard_count}, enemy_signals={enemy_signals}. "
        f"History: {n_act} actions, {n_rew} rewards, {n_hc} human_constraints, {n_llm} LLM summaries."
    )
    return gs


# ==========================================================
# BLOCK 4: LLM STATE SUMMARIZATION
# ==========================================================
# NOTE: Not used by sim_main.py or run_episode() right now — user types plain-text
# instructions instead (Block 5 from text → Block 6/7). Kept for future LLM integration.
def block4_llm_state_summarization(
    gs: GlobalState, cfg: Config,
    ctf_state: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[StrategyCandidate]]:
    """
    Convert global encoded state S_t into natural-language summary; produce candidate
    strategies with reasoning; generate uncertainties, warnings, risk predictions.
    If an external LLM is available (OPENAI_API_KEY and openai installed), use it to
    generate the summary and strategies. Otherwise fall back to rule-based stubs.
    """
    # Assume blue side = high x (x >= 60), red side = low x (x < 60); 120x60 world
    SCRIMMAGE_X = 60.0
    ids = list(gs.robots.keys())
    n_blue = max(1, len(ids)) // 2
    blue_ids = ids[:n_blue]
    # Positions and roles from gs.robots (blue agents only for "our" summary)
    on_our_side: List[RobotId] = []
    on_their_side: List[RobotId] = []
    carriers: List[RobotId] = []
    for rid in blue_ids:
        data = gs.robots.get(rid, {})
        pos = data.get("position", (0.0, 0.0))
        x = float(pos[0])
        if data.get("has_flag"):
            carriers.append(rid)
        if x >= SCRIMMAGE_X:
            on_our_side.append(rid)
        else:
            on_their_side.append(rid)

    sb = ctf_state.get("score_blue", 0) if ctf_state else 0
    sr = ctf_state.get("score_red", 0) if ctf_state else 0
    red_has_blue = ctf_state.get("red_has_blue_flag", False) if ctf_state else False
    blue_has_red = ctf_state.get("blue_has_red_flag", False) if ctf_state else False

    # Rule-based natural-language summary as a fallback / context for the LLM
    summary_parts: List[str] = []
    summary_parts.append(f"Step t={gs.t}. Score Blue {sb} – Red {sr}.")
    if len(on_their_side) > len(on_our_side):
        summary_parts.append("Frontline weak: more of our robots on their side than holding ours.")
    elif len(on_our_side) >= n_blue - 1:
        summary_parts.append("Most of our team on our side; consider pushing into their sector.")
    if carriers:
        summary_parts.append(f"Carriers {', '.join(carriers)} returning with flag.")
    if red_has_blue:
        summary_parts.append("Threat: Red has our flag; defend and reclaim.")
    if blue_has_red and not carriers:
        summary_parts.append("We have their flag (carrier not in list—check state).")
    if len(on_their_side) >= 2 and len([r for r in on_their_side if r not in carriers]) >= 2:
        summary_parts.append("Multiple attackers in enemy sector.")
    if n_blue >= 4 and len(on_our_side) >= 3:
        summary_parts.append("Robots on our side somewhat congested; could spread or send more to attack.")
    summary_parts.append(gs.map_summary)

    rule_based_summary = " ".join(summary_parts)

    # Default stub strategies (used if LLM is disabled or unavailable)
    def _stub_strategies() -> Tuple[str, List[StrategyCandidate]]:
        strategies = [
            StrategyCandidate(
                text="Team A defend; Team B flank right; Team C explore north.",
                risks=["Possible hazard exposure", "Higher communication overhead"],
                uncertainties=["Enemy locations partially observed", "Objective location unknown"],
            ),
            StrategyCandidate(
                text="Scout corners with 2 robots; others hold and communicate.",
                risks=["Scouts could isolate", "Defense might weaken"],
                uncertainties=["Obstacle density unknown"],
            ),
        ]
        return rule_based_summary, strategies

    # If LLM is disabled, just use the stub logic
    if not cfg.enable_llm:
        llm_summary, strategies = _stub_strategies()
        gs.history["llm_summaries"].append(llm_summary)
        return llm_summary, strategies

    # Try to call external LLM (OpenAI) if configured; otherwise fall back to stub
    api_key = os.getenv("OPENAI_API_KEY")
    if openai is None:
        llm_summary, strategies = _stub_strategies()
        gs.history["llm_summaries"].append(llm_summary)
        print("[Block 4] Using stub strategies (openai package not installed).")
        return llm_summary, strategies
    if not api_key:
        llm_summary, strategies = _stub_strategies()
        gs.history["llm_summaries"].append(llm_summary)
        print("[Block 4] Using stub strategies (OPENAI_API_KEY not set).")
        return llm_summary, strategies

    # Build a compact machine-readable state description for the LLM
    state_payload = {
        "time_step": gs.t,
        "score": {"blue": sb, "red": sr},
        "blue_has_red_flag": bool(blue_has_red),
        "red_has_blue_flag": bool(red_has_blue),
        "blue_robots_on_our_side": on_our_side,
        "blue_robots_on_their_side": on_their_side,
        "carriers": carriers,
        "rule_based_summary": rule_based_summary,
    }

    system_msg = (
        "You are the high-level strategist for the BLUE team in a capture-the-flag "
        "simulation. Propose 2 or 3 concise strategies for the BLUE team, each with "
        "risks and uncertainties, given the current game state. Respond with valid JSON only."
    )
    user_msg = (
        "Game state JSON:\n"
        + json.dumps(state_payload, sort_keys=True)
        + "\n\n"
        "Respond ONLY with valid JSON in this exact form (no markdown, no code fences):\n"
        "{\"summary\": \"one paragraph summary\", \"strategies\": [{\"text\": \"...\", \"risks\": [\"...\"], \"uncertainties\": [\"...\"]}, ...]}\n"
    )

    def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, stripping optional markdown code fences."""
        s = raw.strip()
        if s.startswith("```"):
            # Remove ```json or ``` and trailing ```
            lines = s.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    try:
        client = openai.OpenAI(api_key=api_key)  # type: ignore[attr-defined]
        completion = client.chat.completions.create(  # type: ignore[attr-defined]
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        msg = completion.choices[0].message
        content = (msg.content or "").strip()
        if not content:
            raise ValueError("Empty LLM response")
        parsed = _extract_json(content)
        if not parsed or not isinstance(parsed.get("strategies"), list):
            raise ValueError("LLM response missing 'strategies' array")
        llm_summary = parsed.get("summary") or rule_based_summary
        if isinstance(llm_summary, str):
            pass
        else:
            llm_summary = rule_based_summary
        strat_objs: List[StrategyCandidate] = []
        for s in parsed["strategies"]:
            if not isinstance(s, dict):
                continue
            text = s.get("text") or ""
            if not text:
                continue
            strat_objs.append(
                StrategyCandidate(
                    text=text,
                    risks=list(s.get("risks") or []) if isinstance(s.get("risks"), list) else [],
                    uncertainties=list(s.get("uncertainties") or []) if isinstance(s.get("uncertainties"), list) else [],
                )
            )
        if not strat_objs:
            llm_summary, strat_objs = _stub_strategies()
        gs.history["llm_summaries"].append(llm_summary)
        print(f"[Block 4] LLM returned {len(strat_objs)} strategy(ies).")
        return llm_summary, strat_objs
    except Exception as e:
        # Any API / parsing failure: fall back to stub strategies so the sim still runs
        llm_summary, strategies = _stub_strategies()
        gs.history["llm_summaries"].append(llm_summary)
        print(f"[Block 4] LLM failed ({e!r}), using stub strategies.")
        return llm_summary, strategies


# -----------------------------
# Human text commands (Block 5 / 6)
# -----------------------------

def parse_human_command(text: str) -> Optional[ParsedCommand]:
    """
    Parse a single plain-text command into ParsedCommand. Returns None if unrecognized.
    Supported patterns (case-insensitive):
      team A hold region R1
      all avoid red_zone
      robot R3 protect R1 radius 3
      team B attack target T2
      all regroup at 4 4
      team C line at 5 8
      all spread
    """
    t = text.strip()
    if not t:
        return None
    raw = t
    t_low = t.lower()

    m = re.match(
        r"^team\s+([abc])\s+hold\s+region\s+(\w+)",
        t_low,
        re.I,
    )
    if m:
        team, region = m.group(1).upper(), m.group(2)
        return ParsedCommand(
            raw_text=raw,
            action="hold",
            scope="team",
            target=team,
            params={"region": region},
        )

    m = re.match(r"^all\s+avoid\s+(\w+)", t_low)
    if m:
        return ParsedCommand(
            raw_text=raw,
            action="avoid",
            scope="all",
            target="all",
            params={"zone": m.group(1)},
        )

    m = re.match(
        r"^robot\s+(r\d+)\s+protect\s+(r\d+)\s+radius\s+(\d+)",
        t_low,
        re.I,
    )
    if m:
        rid_self, prot, rad = m.group(1).upper(), m.group(2).upper(), int(m.group(3))
        return ParsedCommand(
            raw_text=raw,
            action="protect",
            scope="robot",
            target=rid_self,
            params={"protect_target": prot, "radius": rad},
        )

    m = re.match(r"^team\s+([abc])\s+attack\s+target\s+(\w+)", t_low, re.I)
    if m:
        team, tgt = m.group(1).upper(), m.group(2)
        return ParsedCommand(
            raw_text=raw,
            action="attack",
            scope="team",
            target=team,
            params={"attack_target": tgt},
        )

    m = re.match(r"^all\s+regroup\s+at\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)", t_low)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        return ParsedCommand(
            raw_text=raw,
            action="regroup",
            scope="all",
            target="all",
            params={"cell": (x, y)},
        )

    m = re.match(r"^team\s+([abc])\s+line\s+at\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)", t_low, re.I)
    if m:
        team = m.group(1).upper()
        ax, ay = float(m.group(2)), float(m.group(3))
        return ParsedCommand(
            raw_text=raw,
            action="line",
            scope="team",
            target=team,
            params={"anchor": (ax, ay)},
        )

    if re.match(r"^all\s+spread\b", t_low):
        return ParsedCommand(
            raw_text=raw,
            action="spread",
            scope="all",
            target="all",
            params={},
        )

    return None


def command_applies_to_robot(cmd: ParsedCommand, rid: RobotId, team_of: Dict[str, str]) -> bool:
    """Whether command cmd applies to robot rid given team membership (A/B/C)."""
    scope = cmd.scope
    target = cmd.target.upper() if isinstance(cmd.target, str) else str(cmd.target)
    rid_u = rid.upper()
    team = team_of.get(rid, "A").upper()

    if scope == "all":
        return True
    if scope == "team":
        return target == team
    if scope == "robot":
        # protect: only the named robot
        if cmd.action == "protect":
            return rid_u == target
        return rid_u == target
    return False


def default_team_of(robot_ids: List[RobotId]) -> Dict[str, str]:
    """Assign teams A/B/C in round-robin (same idea as llmagent stub)."""
    out: Dict[str, str] = {}
    for i, rid in enumerate(robot_ids):
        out[rid] = ["A", "B", "C"][i % 3]
    return out


def block5_human_intervention_from_text(
    human_text_commands: List[str],
    enable_human_intervention: bool = True,
) -> HumanPlan:
    """
    Human types commands in plain text (one string per line).
    Example commands:
      - team A hold region R1
      - all avoid red_zone
      - robot R3 protect R1 radius 3
      - team B attack target T2
      - all regroup at 4 4
      - team C line at 5 8
      - all spread
    """
    if not enable_human_intervention:
        return HumanPlan([], [], [], [], [], None)

    parsed_commands: List[ParsedCommand] = []
    priorities: List[str] = []
    safety_constraints: List[str] = []
    mission_goals: List[str] = []
    approved_strategies: List[str] = []

    for text in human_text_commands:
        cmd = parse_human_command(text)
        if cmd is not None:
            parsed_commands.append(cmd)
            approved_strategies.append(text)

            if cmd.action == "avoid":
                safety_constraints.append(text)
            elif cmd.action in ["hold", "protect", "regroup", "line", "spread"]:
                priorities.append(text)
            elif cmd.action == "attack":
                mission_goals.append(text)

    return HumanPlan(
        priorities=priorities,
        safety_constraints=safety_constraints or ["Avoid red_zone"],
        mission_goals=mission_goals,
        approved_strategies=approved_strategies,
        parsed_commands=parsed_commands,
        strategy_index=None,
    )


def block6_high_level_rl_manager_with_commands(
    robot_ids: List[str],
    human_plan: HumanPlan,
    team_of: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """
    Returns high-level assignments per robot from parsed plain-text commands.
    """
    assignments: Dict[str, Dict[str, Any]] = {}

    for rid in robot_ids:
        assignments[rid] = {"subgoal": "IDLE", "target_cell": None}

    for cmd in human_plan.parsed_commands:
        for rid in robot_ids:
            if not command_applies_to_robot(cmd, rid, team_of):
                continue

            if cmd.action == "hold":
                assignments[rid]["subgoal"] = f"HOLD_{cmd.params['region']}"
                assignments[rid]["target_region"] = cmd.params["region"]

            elif cmd.action == "avoid":
                assignments[rid].setdefault("avoid_zones", [])
                assignments[rid]["avoid_zones"].append(cmd.params["zone"])

            elif cmd.action == "protect":
                assignments[rid]["subgoal"] = f"PROTECT_{cmd.params['protect_target']}"
                assignments[rid]["protect_target"] = cmd.params["protect_target"]
                assignments[rid]["radius"] = cmd.params["radius"]

            elif cmd.action == "attack":
                assignments[rid]["subgoal"] = f"ATTACK_{cmd.params['attack_target']}"
                assignments[rid]["attack_target"] = cmd.params["attack_target"]

            elif cmd.action == "regroup":
                assignments[rid]["subgoal"] = "REGROUP"
                assignments[rid]["target_cell"] = cmd.params["cell"]

            elif cmd.action == "line":
                assignments[rid]["subgoal"] = "LINE_FORMATION"
                assignments[rid]["anchor"] = cmd.params["anchor"]

            elif cmd.action == "spread":
                assignments[rid]["subgoal"] = "SPREAD"

    return assignments


def block7_dispatch_assignments(gs: GlobalState, assignments: Dict[str, Dict[str, Any]]) -> None:
    """Attach per-robot high-level assignment for Block 8 (replaces team-only dispatch)."""
    for rid, a in assignments.items():
        if rid not in gs.robots:
            continue
        gs.robots[rid]["hl_assignment"] = dict(a)


# ==========================================================
# BLOCK 5 (legacy): LLM strategy menu — kept for all_blocks.run_episode
# ==========================================================
def block5_human_intervention(
    strategies: List[StrategyCandidate], cfg: Config,
    llm_summary: str = "",
) -> HumanPlan:
    """
    Human reads LLM summary; approves, modifies, merges, or rejects strategies; adds priorities,
    safety constraints, mission goals. Output = final HUMAN+LLM strategic plan.
    Caller should invoke once per episode (or when replanning) and reuse the returned plan.
    """
    if not cfg.enable_human_intervention:
        return HumanPlan([], [], [], [], [], None)

    if not strategies:
        return HumanPlan(
            priorities=[], safety_constraints=[], mission_goals=[],
            approved_strategies=["No strategy available"],
            parsed_commands=[],
            strategy_index=None,
        )

    print("\n--- Blue team: strategy approval ---")
    print("Summary:", llm_summary)
    print("\nCandidate strategies (with risks and uncertainties):")
    for i, s in enumerate(strategies, 1):
        print(f"  {i}. {s.text}")
        if s.risks:
            print(f"     Risks: {', '.join(s.risks)}")
        if s.uncertainties:
            print(f"     Uncertainties: {', '.join(s.uncertainties)}")
    print(f"\nChoose a strategy (1–{len(strategies)}), or type your own (e.g. 'Two defend, rest attack'):")
    choice = input("Your choice [1]: ").strip() or "1"

    strategy_index = None
    if choice.isdigit() and 1 <= int(choice) <= len(strategies):
        strategy_index = int(choice)
        approved = strategies[strategy_index - 1].text
    else:
        approved = choice if choice else strategies[0].text

    return HumanPlan(
        priorities=["Defense > Capture"],
        safety_constraints=["Avoid red_zone"],
        mission_goals=["Secure perimeter", "Capture opponent flag"],
        approved_strategies=[approved],
        parsed_commands=[],
        strategy_index=strategy_index,
    )


# ==========================================================
# BLOCK 6: HIGH-LEVEL RL MANAGER (HRL META-POLICY)
# ==========================================================
def block6_high_level_rl_manager(gs: GlobalState, human_plan: HumanPlan) -> HRLAction:
    """
    HRL meta-policy: inputs = Human+LLM plan, encoded global state S_t.
    Decides: (A) which robot groups/teams to form; (B) which subgoals to assign (e.g. hold region,
    capture flag, patrol, scout); (C) when to replan. Output = subgoal assignment a_HRL.
    Maps approved strategy text to Defend/Attack teams and role-based subgoals for block8.
    """
    ids = list(gs.robots.keys())
    if not ids:
        return HRLAction(teams={}, subgoals={}, request_replan=False)

    approved = (human_plan.approved_strategies or [""])[0].lower()
    n = len(ids)

    # When user picked strategy 1/2/3, use that to set team split. Strategy 1 = Defend + Flank + Attack.
    idx = getattr(human_plan, "strategy_index", None)
    num_flank = 0
    if idx is not None and 1 <= idx <= 3:
        # 1 = balanced with flank (2 defend, 2 flank, rest attack), 2 = aggressive (1 defend), 3 = cautious (3 defend)
        num_defend = {1: min(2, n), 2: min(1, n), 3: min(3, max(1, n - 1))}[idx]
        if idx == 1 and n >= 4:
            num_flank = min(2, n - num_defend - 1)  # at least 1 attacker
    else:
        # Parse custom text
        num_defend = 0
        if "split" in approved or "1–2 defenders" in approved or "1-2 defenders" in approved:
            num_defend = min(2, max(1, n // 3))
        elif "team a defend" in approved or ("defend" in approved and ("flank" in approved or "push" in approved or "attack" in approved or "scout" in approved)):
            num_defend = min(2, max(1, n // 3))
            if "flank" in approved and n >= 4:
                num_flank = min(2, n - num_defend - 1)
        elif "defend first" in approved or "majority defend" in approved:
            num_defend = max(1, n - 2)
        if "defend" in approved and num_defend == 0 and n >= 2:
            num_defend = min(2, max(1, n // 3))

    defend_ids = ids[:num_defend]
    flank_ids = ids[num_defend : num_defend + num_flank] if num_flank else []
    attack_ids = ids[num_defend + num_flank :]

    teams: Dict[TeamId, List[RobotId]] = {}
    subgoals: Dict[TeamId, Subgoal] = {}

    if defend_ids:
        teams["Defend"] = defend_ids
        subgoals["Defend"] = Subgoal(
            "defend_flag", "Defend our flag", encode_subgoal("Defend our flag"), role="defend"
        )
    if flank_ids:
        teams["Flank"] = flank_ids
        subgoals["Flank"] = Subgoal(
            "flank_right", "Flank right toward enemy", encode_subgoal("Flank right"), role="flank"
        )
    if attack_ids:
        teams["Attack"] = attack_ids
        subgoals["Attack"] = Subgoal(
            "attack_red", "Go to red zone / capture flag", encode_subgoal("Attack red zone"), role="attack"
        )

    request_replan = False
    return HRLAction(teams=teams, subgoals=subgoals, request_replan=request_replan)


# ==========================================================
# BLOCK 7: SUBGOAL DISPATCHING TO ROBOT TEAMS
# ==========================================================
def block7_subgoal_dispatching(gs: GlobalState, hrl: HRLAction) -> None:
    """
    Subgoal g_k assigned to team Tk; each subgoal encoded into vector; each robot receives
    its team's subgoal with local meaning.
    """
    for team_id, members in hrl.teams.items():
        for rid in members:
            gs.robots[rid]["team"] = team_id


# ==========================================================
# BLOCK 8: ASTAR (pathfinding utility)
# ==========================================================
def block8_astar(
    grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int], eight_connected: bool = True
) -> List[Tuple[int, int]]:
    """
    A* pathfinding on a 2D grid. grid[row][col] with 1 = obstacle.
    Returns path from start to goal (list of (row, col)), or [] if no path.
    If eight_connected, uses 8 neighbors (N, NE, E, SE, S, SW, W, NW).
    """
    rows = len(grid)
    cols = len(grid[0])

    def heuristic(a: Vec2, b: Vec2) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        if eight_connected:
            neighbor_list = [
                (current[0] + dr, current[1] + dc)
                for dr, dc in _DIRECTION_TO_OFFSET.values()
            ]
        else:
            neighbor_list = [
                (current[0] + 1, current[1]), (current[0] - 1, current[1]),
                (current[0], current[1] + 1), (current[0], current[1] - 1),
            ]

        for n in neighbor_list:
            if 0 <= n[0] < rows and 0 <= n[1] < cols:
                if grid[n[0]][n[1]] == 1:
                    continue
                cost = 1.414 if (eight_connected and n[0] != current[0] and n[1] != current[1]) else 1.0
                tentative = g_score[current] + cost
                if n not in g_score or tentative < g_score[n]:
                    came_from[n] = current
                    g_score[n] = tentative
                    f = tentative + heuristic(n, goal)
                    heapq.heappush(open_set, (f, n))

    return []


def _build_grid(
    rows: int, cols: int,
    obstacles: Optional[List[Tuple[int, int]]] = None,
    boundary_buffer: int = 0,
) -> List[List[int]]:
    """Build a grid (0=free, 1=obstacle). Optional obstacles list. boundary_buffer: cells
    from each edge to mark as obstacles (keeps paths inside play area per pyquaticus)."""
    grid = [[0] * cols for _ in range(rows)]
    if boundary_buffer > 0:
        for r in range(rows):
            for c in range(cols):
                if r < boundary_buffer or r >= rows - boundary_buffer:
                    grid[r][c] = 1
                elif c < boundary_buffer or c >= cols - boundary_buffer:
                    grid[r][c] = 1
    if obstacles:
        for r, c in obstacles:
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = 1
    return grid


def _snap_goal_to_nearest_free_cell(
    grid: List[List[int]], goal: Vec2
) -> Optional[Vec2]:
    """
    If goal sits on a blocked cell (e.g. inside boundary buffer), return the closest
    free cell using 8-connected BFS from `goal`. If `goal` is already free, return it.
    """
    rows = len(grid)
    if rows == 0:
        return None
    cols = len(grid[0])
    gr, gc = goal
    if 0 <= gr < rows and 0 <= gc < cols and grid[gr][gc] == 0:
        return goal

    q: deque[Vec2] = deque()
    seen: set[Vec2] = set()
    q.append((gr, gc))
    seen.add((gr, gc))
    while q:
        r, c = q.popleft()
        if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0:
            return (r, c)
        for dr, dc in _DIRECTION_TO_OFFSET.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return None


def _goal_world_from_hl_assignment(
    assignment: Dict[str, Any],
    opponent_flag_world: Tuple[float, float],
    own_flag_world: Optional[Tuple[float, float]],
    defender_hold_world: Optional[Tuple[float, float]],
    flank_hold_world: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """Map Block 6 command assignment to a world (x, y) goal. None => HOLD."""
    sg = str(assignment.get("subgoal", "IDLE"))
    tc = assignment.get("target_cell")
    if tc and hasattr(tc, "__len__") and len(tc) >= 2:
        tc_world = (float(tc[0]), float(tc[1]))
    else:
        tc_world = None
    if sg == "IDLE" or sg.startswith("IDLE"):
        return None
    if sg.startswith("ATTACK"):
        return tc_world or opponent_flag_world
    if sg.startswith("HOLD") or sg.startswith("PROTECT"):
        return tc_world or defender_hold_world or own_flag_world
    if sg == "REGROUP":
        return tc_world or defender_hold_world or own_flag_world
    if sg == "LINE_FORMATION":
        an = assignment.get("anchor")
        if an and len(an) >= 2:
            return (float(an[0]), float(an[1]))
        return flank_hold_world
    if sg == "SPREAD":
        return tc_world or flank_hold_world
    return opponent_flag_world


# ==========================================================
# BLOCK 8: LOW-LEVEL MULTI-AGENT EXECUTION 
# ==========================================================
def block8_low_level_execution(
    gs: GlobalState, local_obs: List[LocalObservation], hrl: HRLAction,
    grid_rows: int = GRID_ROWS, grid_cols: int = GRID_COLS,
    opponent_flag_world: Optional[Tuple[float, float]] = None,
    own_flag_world: Optional[Tuple[float, float]] = None,
    defender_hold_world: Optional[Tuple[float, float]] = None,
    flank_hold_world: Optional[Tuple[float, float]] = None,
    robot_assignments: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Each robot Ri uses policy (here A* placeholder for MAPPO/MASAC) to select primitive action
    (movement). Inputs: local obs oi, encoded subgoal g_k. Outputs: action ai.
    CTF: carrier → own zone; Defend → defender_hold_world; Flank → flank_hold_world (flank right);
    Attack → opponent zone. If robot_assignments is set (Block 6 command mode), goals follow
    per-robot hl_assignment / assignment dict instead of team_id. Boundary buffer keeps paths inside play area.
    """
    use_flag_goal = opponent_flag_world is not None
    # boundary_buffer = agent radius in cells so agents can get as close to walls as allowed
    # Keep paths well away from walls so defenders don't run into bottom/top boundaries
    boundary_buffer = PATH_BOUNDARY_BUFFER_CELLS if use_flag_goal else 0
    grid = _build_grid(grid_rows, grid_cols, boundary_buffer=boundary_buffer)
    robot_actions: List[Dict[str, Any]] = []
    goal_offset_cells = 15

    for o in local_obs:
        rid = o.robot_id
        pos = gs.robots.get(rid, {}).get("position", (0, 0))
        x, y = pos
        start = _world_to_grid(x, y)

        # --- Per-robot command assignments (Block 5/6 text commands) ---
        hl: Optional[Dict[str, Any]] = None
        if robot_assignments is not None and rid in robot_assignments:
            hl = robot_assignments[rid]
        elif gs.robots.get(rid, {}).get("hl_assignment") is not None:
            hl = gs.robots[rid]["hl_assignment"]

        if hl is not None:
            subgoal_vec = encode_subgoal(str(hl.get("subgoal", "cmd")))
            fallback_direction = pick_direction_from_vector(subgoal_vec)
            if use_flag_goal:
                if own_flag_world is not None and gs.robots.get(rid, {}).get("has_flag", False):
                    goal_world = own_flag_world
                else:
                    gwl = _goal_world_from_hl_assignment(
                        hl, opponent_flag_world, own_flag_world, defender_hold_world, flank_hold_world
                    )
                    if gwl is None:
                        robot_actions.append({"robot_id": rid, "action": "HOLD"})
                        continue
                    goal_world = gwl
                goal = _world_to_grid(float(goal_world[0]), float(goal_world[1]))
            else:
                drow, dcol = _DIRECTION_TO_OFFSET.get(fallback_direction, (0, 0))
                goal_row = clamp(start[0] + goal_offset_cells * drow, 0, grid_rows - 1)
                goal_col = clamp(start[1] + goal_offset_cells * dcol, 0, grid_cols - 1)
                goal = (goal_row, goal_col)
        else:
            team_id = gs.robots.get(rid, {}).get("team")
            if team_id is None or team_id not in hrl.subgoals:
                robot_actions.append({"robot_id": rid, "action": "HOLD"})
                continue

            subgoal = hrl.subgoals[team_id]
            fallback_direction = pick_direction_from_vector(subgoal.vector)

            if use_flag_goal:
                # Carrier → own zone; Defend → defender_hold_world; Flank → flank_hold_world; Attack → opponent zone
                if own_flag_world is not None and gs.robots.get(rid, {}).get("has_flag", False):
                    goal_world = own_flag_world
                elif team_id == "Defend" and (defender_hold_world is not None or own_flag_world is not None):
                    goal_world = defender_hold_world if defender_hold_world is not None else own_flag_world
                elif team_id == "Flank" and flank_hold_world is not None:
                    goal_world = flank_hold_world
                else:
                    goal_world = opponent_flag_world
                goal = _world_to_grid(float(goal_world[0]), float(goal_world[1]))
            else:
                drow, dcol = _DIRECTION_TO_OFFSET.get(fallback_direction, (0, 0))
                goal_row = clamp(start[0] + goal_offset_cells * drow, 0, grid_rows - 1)
                goal_col = clamp(start[1] + goal_offset_cells * dcol, 0, grid_cols - 1)
                goal = (goal_row, goal_col)

        snapped = _snap_goal_to_nearest_free_cell(grid, goal)
        if snapped is None:
            robot_actions.append({"robot_id": rid, "action": "HOLD"})
            continue
        goal = snapped

        if start == goal:
            robot_actions.append({"robot_id": rid, "action": "HOLD"})
            continue

        path = block8_astar(grid, start, goal, eight_connected=True)
        if len(path) >= 2:
            dr = path[1][0] - path[0][0]
            dc = path[1][1] - path[0][1]
            direction = _path_step_to_direction(dr, dc)
        else:
            direction = fallback_direction

        # Boundary nudge: avoid driving into walls. (After dr flip, N = +y = away from bottom.)
        row, col = start
        margin = (boundary_buffer + 1) if use_flag_goal else 1
        # Near bottom (low row): don't command S/SW/SE (env -y); nudge to N
        if row < margin and direction in ("S", "SW", "SE"):
            direction = "N"
        elif row > grid_rows - margin - 1 and direction in ("N", "NW", "NE"):
            direction = "S"
        if col < margin and direction in ("W", "NW", "SW"):
            direction = "E"
        elif col > grid_cols - margin - 1 and direction in ("E", "NE", "SE"):
            direction = "W"

        robot_actions.append({"robot_id": rid, "action": direction})

    return robot_actions


# ==========================================================
# BLOCK 9: ENVIRONMENT TRANSITION
# ==========================================================
def block9_environment_transition(gs: GlobalState, actions: List[Dict[str, Any]]) -> Reward:
    """
    Apply all robot actions; world → S_(t+1). Compute reward (task progress, goal completion,
    damage/collisions, safety violations); update logs. Standalone use; sim_main
    uses PyQuaticus env for transition.
    """
    reward_val = 0.0
    breakdown: List[str] = []

    for a in actions:
        rid = a["robot_id"]
        act = a["action"]
        x, y = gs.robots[rid]["position"]
        before = (x, y)

        if act == "MOVE_UP":
            y = clamp(y - 1, 0, 9)
        elif act == "MOVE_DOWN":
            y = clamp(y + 1, 0, 9)
        elif act == "MOVE_LEFT":
            x = clamp(x - 1, 0, 9)
        elif act == "MOVE_RIGHT":
            x = clamp(x + 1, 0, 9)
        elif act == "HOLD":
            pass
        # ATTACK/COLLECT/COMMUNICATE can be added later

        gs.robots[rid]["position"] = (x, y)

        # simple shaping reward: +0.1 if moved
        if (x, y) != before:
            reward_val += 0.1

        gs.history["actions"].append({"t": gs.t, "robot_id": rid, "action": act})

    breakdown.append(f"task_progress={reward_val:.2f}")
    gs.history["rewards"].append({"t": gs.t, "value": reward_val, "reason": "progress"})

    gs.t += 1
    return Reward(value=reward_val, breakdown=breakdown)


# ==========================================================
# BLOCK 10: CHECK FOR STRATEGIC CHANGES
# ==========================================================
def block10_check_strategic_changes(
    gs: GlobalState,
    ctf_state: Optional[Dict[str, Any]] = None,
    last_replan_step: int = -1,
    last_replan_ctf_state: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Returns True if we should return to Human+LLM (steps 4–5). Caller should set human_plan = None.

    Checks:
    1) Plan failing: score_red increased since last replan (they just scored).
    2) New threat: red_has_blue_flag just became True since last replan.

    ctf_state: score_blue, score_red, blue_has_red_flag, red_has_blue_flag.
    Caller must pass last_replan_step and last_replan_ctf_state and update them when this returns True.
    """
    if ctf_state is None:
        ctf_state = {}
    sr = ctf_state.get("score_red", 0)
    red_has_blue = ctf_state.get("red_has_blue_flag", False)

    # 1) Plan failing: they just scored since last replan
    if last_replan_ctf_state is not None:
        prev_sr = last_replan_ctf_state.get("score_red", -1)
        if sr > prev_sr:
            return True
        prev_red_had = last_replan_ctf_state.get("red_has_blue_flag", False)
        if red_has_blue and not prev_red_had:
            return True  # they just took our flag (new threat; see also 2b below)

    # 2b) New threat: red just took our flag since last replan
    if red_has_blue and last_replan_step >= 0 and last_replan_ctf_state is not None:
        prev_red_had = last_replan_ctf_state.get("red_has_blue_flag", False)
        if not prev_red_had:
            return True

    return False


# ==========================================================
# BLOCK 11: LEARNING & POLICY UPDATES
# ==========================================================
def block11_learning_and_updates(gs: GlobalState) -> None:
    """
     Placeholder for PPO/A2C updates (high-level) and MAPPO/MASAC updates (low-level).
    """
    # In a real implementation:
    # - collect trajectories
    # - compute advantages/returns
    # - update policy networks
    gs.history.setdefault("learning_notes", [])
    gs.history["learning_notes"].append(f"Learning update executed at t={gs.t} (stub)")


# ==========================================================
# BLOCK 12: TERMINATION CHECK
# ==========================================================
def block12_termination_check(
    gs: GlobalState, cfg: Config, env: Optional[Any] = None
) -> Tuple[bool, str]:
    """
    Win/loss achieved? Task finished? Max steps? Human manually ends (handled by caller).
    Returns (done, reason_string). If env is provided and env.dones["__all__"], returns
    (True, env.message) for win/loss or time limit. Else if gs.t >= max_steps, returns
    (True, "Max steps reached"). Otherwise (False, "").
    """
    if env is not None and getattr(env, "dones", {}).get("__all__", False):
        return True, getattr(env, "message", "Game over.")
    if gs.t >= cfg.max_steps_per_episode:
        return True, "Max steps reached."
    return False, ""


# -----------------------------
# Example runner (optional)
# -----------------------------
def run_episode(cfg: Config) -> GlobalState:
    gs = block1_environment_initialization(cfg)

    while True:
        local_obs = block2_state_collection(gs)
        gs = block3_global_state_encoding(gs, local_obs)
        # Block 4 (LLM summarization) disabled — use parsed user-style commands instead.
        # llm_summary, strategies = block4_llm_state_summarization(gs, cfg)
        # human_plan = block5_human_intervention(strategies, cfg)
        blue_rids = list(gs.robots.keys())
        human_plan = block5_human_intervention_from_text(
            ["team A attack target T1"], enable_human_intervention=cfg.enable_human_intervention
        )
        team_of = default_team_of(blue_rids)
        robot_assignments = block6_high_level_rl_manager_with_commands(
            blue_rids, human_plan, team_of
        )
        block7_dispatch_assignments(gs, robot_assignments)
        hrl = HRLAction(teams={}, subgoals={}, request_replan=False)
        gs.history["human_constraints"].extend(human_plan.safety_constraints)

        robot_actions = block8_low_level_execution(
            gs, local_obs, hrl, robot_assignments=robot_assignments
        )
        reward = block9_environment_transition(gs, robot_actions)

        should_replan = block10_check_strategic_changes(gs) or hrl.request_replan

        # periodic learning
        if gs.t > 0 and gs.t % 20 == 0:
            block11_learning_and_updates(gs)

        if cfg.log_every and gs.t % cfg.log_every == 0:
            print(f"[t={gs.t}] reward={reward.value:.2f} replan={should_replan}")

        done, _ = block12_termination_check(gs, cfg)
        if done:
            break

    return gs


if __name__ == "__main__":
    cfg = Config(num_robots=12, max_steps_per_episode=30, enable_llm=True, enable_human_intervention=True)
    final_state = run_episode(cfg)
    print("\nDONE. Final t =", final_state.t)
    print("Actions logged:", len(final_state.history["actions"]))
    print("Rewards logged:", len(final_state.history["rewards"]))



