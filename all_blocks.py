from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import random
import math
import heapq

# -----------------------------
# Types / data structures
# -----------------------------

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
class HumanPlan:
    priorities: List[str]
    safety_constraints: List[str]
    mission_goals: List[str]
    approved_strategies: List[str]
    strategy_index: Optional[int] = None  # 1, 2, or 3 when user picks a candidate; None for custom text

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
    max_steps_per_episode: int = 50
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
# Set to agent_radius (2) so agents can get as close to boundaries as the env allows.
AGENT_RADIUS_CELLS = 2

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
    """Map (drow, dcol) from one path cell to next to direction string."""
    return _OFFSET_TO_DIRECTION.get((dr, dc), "HOLD")


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


# ==========================================================
# BLOCK 2: STATE COLLECTION (RAW MULTI-ROBOT OBSERVATIONS)
# ==========================================================
def block2_state_collection(gs: GlobalState) -> List[LocalObservation]:
    """
    Each robot Ri collects local observation oi: nearby map/objects/enemies, teammate positions,
    local hazards. No full observability; all observations gathered centrally.
    """
    ids = list(gs.robots.keys())

    local_obs: List[LocalObservation] = []
    for rid in ids:
        pos = gs.robots[rid]["position"]

        teammates = [(tid, gs.robots[tid]["position"]) for tid in ids if tid != rid][:3]
        nearby = [x for x in ["wall", "crate", "enemy?"] if random.random() < 0.4]
        hazards = [x for x in ["red_zone", "minefield"] if random.random() < 0.2]

        local_obs.append(LocalObservation(
            robot_id=rid,
            position=pos,
            nearby=nearby,
            teammates=teammates,
            hazards=hazards,
            messages_in=[]
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
    enemy_signals = sum(1 for o in local_obs if "enemy?" in o.nearby)
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
# BLOCK 4: LLM STATE SUMMARIZATION (framework §4)
# ==========================================================
def block4_llm_state_summarization(
    gs: GlobalState, cfg: Config,
    ctf_state: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[StrategyCandidate]]:
    """
    Convert global encoded state S_t into natural-language summary; produce candidate
    strategies with reasoning; generate uncertainties, warnings, risk predictions (framework §4).
    """
    if not cfg.enable_llm:
        return "", []

    # Assume blue side = high x (x >= 60), red side = low x (x < 60); 120x60 world
    SCRIMMAGE_X = 60.0
    ids = list(gs.robots.keys())
    n_blue = len(ids) // 2
    blue_ids = ids[:n_blue]
    # Positions and roles from gs.robots (blue agents only for "our" summary)
    on_our_side = []
    on_their_side = []
    carriers = []
    for rid in blue_ids:
        data = gs.robots.get(rid, {})
        pos = data.get("position", (0, 0))
        x = pos[0] if isinstance(pos, (tuple, list)) else pos[0]
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

    # Natural-language summary: "Frontline weak, enemies in sector B, robots 1-5 congested…"
    summary_parts = []
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
    # Congestion: many in similar area (simplified: count on each side)
    if n_blue >= 4 and len(on_our_side) >= 3:
        summary_parts.append("Robots on our side somewhat congested; could spread or send more to attack.")
    summary_parts.append(gs.map_summary)

    llm_summary = " ".join(summary_parts)
    gs.history["llm_summaries"].append(llm_summary)

    # Candidate strategies with reasoning (framework style)
    strategies = [
        StrategyCandidate(
            text="Team A defend sector D (our flag); Team B flank right; Team C push northern side toward red flag.",
            risks=["Defenders may be outnumbered if Red commits", "Flank could be cut off"],
            uncertainties=["Enemy strength in sector B unknown", "Red flag guard count unknown"],
        ),
        StrategyCandidate(
            text="Team A defend our flag; Team B and C all attack red zone—overwhelm their defense.",
            risks=["Our flag undefended", "Red may score if they have our flag"],
            uncertainties=["How many Red are attacking vs defending"],
        ),
        StrategyCandidate(
            text="Team A hold and defend; Team B scout northern corridor; Team C scout southern corridor, then regroup.",
            risks=["Slower to capture", "Red may score first if they have our flag"],
            uncertainties=["When to switch from scout to full attack"],
        ),
    ]

    return llm_summary, strategies


# ==========================================================
# BLOCK 5: HUMAN STRATEGY INTERVENTION
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
        return HumanPlan([], [], [], [], None)

    if not strategies:
        return HumanPlan(
            priorities=[], safety_constraints=[], mission_goals=[],
            approved_strategies=["No strategy available"],
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
    print("\nChoose a strategy (1–3), or type your own (e.g. 'Two defend, rest attack'):")
    choice = input("Your choice [1]: ").strip() or "1"

    strategy_index = None
    if choice.isdigit() and 1 <= int(choice) <= len(strategies):
        strategy_index = int(choice)
        approved = strategies[strategy_index - 1].text
    else:
        approved = choice if choice else strategies[0].text

    constraints = input("Extra safety constraints (comma-separated, or Enter for none): ").strip()
    safety_constraints = ["Avoid red_zone"]
    if constraints:
        safety_constraints.extend(c.strip() for c in constraints.split(",") if c.strip())

    return HumanPlan(
        priorities=["Defense > Capture"],
        safety_constraints=safety_constraints,
        mission_goals=["Secure perimeter", "Capture opponent flag"],
        approved_strategies=[approved],
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

    # When user picked strategy 1/2/3, use that to set defender count (so teams differ by choice)
    idx = getattr(human_plan, "strategy_index", None)
    if idx is not None and 1 <= idx <= 3:
        # 1 = balanced (2 defend), 2 = aggressive (1 defend), 3 = cautious (3 defend)
        num_defend = {1: min(2, n), 2: min(1, n), 3: min(3, max(1, n - 1))}[idx]
    else:
        # Parse custom text into roles: how many defenders (rest attackers)
        num_defend = 0
        if "split" in approved or "1–2 defenders" in approved or "1-2 defenders" in approved:
            num_defend = min(2, max(1, n // 3))
        elif "team a defend" in approved or ("defend" in approved and ("flank" in approved or "push" in approved or "attack" in approved or "scout" in approved)):
            num_defend = min(2, max(1, n // 3))
        elif "defend first" in approved or "majority defend" in approved:
            num_defend = max(1, n - 2)
        if "defend" in approved and num_defend == 0 and n >= 2:
            num_defend = min(2, max(1, n // 3))

    defend_ids = ids[:num_defend]
    attack_ids = ids[num_defend:]

    teams: Dict[TeamId, List[RobotId]] = {}
    subgoals: Dict[TeamId, Subgoal] = {}

    if defend_ids:
        teams["Defend"] = defend_ids
        subgoals["Defend"] = Subgoal(
            "defend_flag", "Defend our flag", encode_subgoal("Defend our flag"), role="defend"
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


# ==========================================================
# BLOCK 8: LOW-LEVEL MULTI-AGENT EXECUTION 
# ==========================================================
def block8_low_level_execution(
    gs: GlobalState, local_obs: List[LocalObservation], hrl: HRLAction,
    grid_rows: int = GRID_ROWS, grid_cols: int = GRID_COLS,
    opponent_flag_world: Optional[Tuple[float, float]] = None,
    own_flag_world: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Each robot Ri uses policy (here A* placeholder for MAPPO/MASAC) to select primitive action
    (movement). Inputs: local obs oi, encoded subgoal g_k. Outputs: action ai.
    CTF: carrier → own zone; role defend → own zone; role attack → opponent zone. Boundary buffer
    keeps paths inside play area.
    """
    use_flag_goal = opponent_flag_world is not None
    # boundary_buffer = agent radius in cells so agents can get as close to walls as allowed
    boundary_buffer = AGENT_RADIUS_CELLS if use_flag_goal else 0
    grid = _build_grid(grid_rows, grid_cols, boundary_buffer=boundary_buffer)
    robot_actions: List[Dict[str, Any]] = []
    goal_offset_cells = 15

    for o in local_obs:
        rid = o.robot_id
        team_id = gs.robots.get(rid, {}).get("team")
        if team_id is None or team_id not in hrl.subgoals:
            robot_actions.append({"robot_id": rid, "action": "HOLD"})
            continue

        subgoal = hrl.subgoals[team_id]
        fallback_direction = pick_direction_from_vector(subgoal.vector)
        pos = gs.robots.get(rid, {}).get("position", (0, 0))
        x, y = pos
        start = _world_to_grid(x, y)

        if use_flag_goal:
            # Carrier always goes home; Defend team → own zone, Attack team → opponent zone (use team_id explicitly)
            if own_flag_world is not None and gs.robots.get(rid, {}).get("has_flag", False):
                goal_world = own_flag_world
            elif team_id == "Defend" and own_flag_world is not None:
                goal_world = own_flag_world
            else:
                goal_world = opponent_flag_world
            goal = _world_to_grid(float(goal_world[0]), float(goal_world[1]))
        else:
            drow, dcol = _DIRECTION_TO_OFFSET.get(fallback_direction, (0, 0))
            goal_row = clamp(start[0] + goal_offset_cells * drow, 0, grid_rows - 1)
            goal_col = clamp(start[1] + goal_offset_cells * dcol, 0, grid_cols - 1)
            goal = (goal_row, goal_col)

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

        # Stronger boundary avoidance: if we're already near a wall, do not
        # choose a direction that drives us closer to that wall. Nudge the
        # direction back toward the interior instead.
        row, col = start
        margin = (boundary_buffer + 1) if use_flag_goal else 1

        if row < margin and direction in ("N", "NW", "NE"):
            direction = "S"
        elif row > grid_rows - margin - 1 and direction in ("S", "SW", "SE"):
            direction = "N"

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


# Replan every this many env steps (set 0 to disable periodic replan)
REPLAN_INTERVAL_STEPS = 60

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
    1) Periodic: (gs.t - last_replan_step) >= REPLAN_INTERVAL_STEPS (if REPLAN_INTERVAL_STEPS > 0).
    2) Plan failing: score_red increased since last replan (they just scored).
    3) New threat: red_has_blue_flag just became True since last replan.

    ctf_state: score_blue, score_red, blue_has_red_flag, red_has_blue_flag.
    Caller must pass last_replan_step and last_replan_ctf_state and update them when this returns True.
    """
    if ctf_state is None:
        ctf_state = {}
    step = gs.t
    sr = ctf_state.get("score_red", 0)
    red_has_blue = ctf_state.get("red_has_blue_flag", False)

    # 1) Periodic replan
    if REPLAN_INTERVAL_STEPS > 0 and step > 0 and last_replan_step >= 0 and (step - last_replan_step) >= REPLAN_INTERVAL_STEPS:
        return True

    # 2) Plan failing: they just scored since last replan
    if last_replan_ctf_state is not None:
        prev_sr = last_replan_ctf_state.get("score_red", -1)
        if sr > prev_sr:
            return True
        prev_red_had = last_replan_ctf_state.get("red_has_blue_flag", False)
        if red_has_blue and not prev_red_had:
            return True  # they just took our flag (new threat; see also check 3)

    # 3) New threat: red just took our flag since last replan
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
def block12_termination_check(gs: GlobalState, cfg: Config) -> bool:
    """
    Win/loss achieved? Task finished? Human manually ends episode?.
    """
    return gs.t >= cfg.max_steps_per_episode


# -----------------------------
# Example runner (optional)
# -----------------------------
def run_episode(cfg: Config) -> GlobalState:
    gs = block1_environment_initialization(cfg)

    while True:
        local_obs = block2_state_collection(gs)
        gs = block3_global_state_encoding(gs, local_obs)
        llm_summary, strategies = block4_llm_state_summarization(gs, cfg)
        human_plan = block5_human_intervention(strategies, cfg)
        gs.history["human_constraints"].extend(human_plan.safety_constraints)

        hrl = block6_high_level_rl_manager(gs, human_plan)
        block7_subgoal_dispatching(gs, hrl)
        robot_actions = block8_low_level_execution(gs, local_obs, hrl)
        reward = block9_environment_transition(gs, robot_actions)

        should_replan = block10_check_strategic_changes(gs) or hrl.request_replan

        # periodic learning
        if gs.t > 0 and gs.t % 20 == 0:
            block11_learning_and_updates(gs)

        if cfg.log_every and gs.t % cfg.log_every == 0:
            print(f"[t={gs.t}] reward={reward.value:.2f} replan={should_replan}")

        if block12_termination_check(gs, cfg):
            break

    return gs


if __name__ == "__main__":
    cfg = Config(num_robots=12, max_steps_per_episode=30, enable_llm=True, enable_human_intervention=True)
    final_state = run_episode(cfg)
    print("\nDONE. Final t =", final_state.t)
    print("Actions logged:", len(final_state.history["actions"]))
    print("Rewards logged:", len(final_state.history["rewards"]))



