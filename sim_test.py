from __future__ import annotations
 
from dataclasses import dataclass, field

from typing import Dict, List, Tuple, Optional, Any

from collections import defaultdict

import heapq

import math

import random

import re
 
 
# =========================================================

# TYPES

# =========================================================
 
Cell = Tuple[int, int]

RobotId = str
 
 
@dataclass

class Robot:

    robot_id: RobotId

    team: str                  # "BLUE" or "RED"

    position: Cell

    hp: int = 100

    alive: bool = True

    kills: int = 0

    deaths: int = 0

    assists: int = 0

    distance_covered: float = 0.0

    last_position: Optional[Cell] = None
 
 
@dataclass

class ParsedCommand:

    raw_text: str

    action: str                # move, hold, defend, attack, regroup, spread

    scope: str                 # all or robot

    target: str                # all or robot id

    params: Dict[str, Any] = field(default_factory=dict)
 
 
@dataclass

class LocalObservation:

    robot_id: RobotId

    team: str

    position: Cell

    nearby_teammates: List[Tuple[RobotId, Cell]]

    nearby_enemies: List[Tuple[RobotId, Cell]]

    nearby_obstacles: List[Cell]

    visible_regions: List[str]

    hp: int
 
 
@dataclass

class GlobalState:

    timestep: int

    blue_positions: Dict[RobotId, Cell]

    red_positions: Dict[RobotId, Cell]

    alive_blue: int

    alive_red: int

    blue_base: Cell

    red_base: Cell

    contested_regions: List[str]

    threat_map: Dict[RobotId, float]
 
 
@dataclass

class HumanPlan:

    parsed_commands: List[ParsedCommand] = field(default_factory=list)
 
 
@dataclass

class Metrics:

    kills: Dict[RobotId, int] = field(default_factory=lambda: defaultdict(int))

    deaths: Dict[RobotId, int] = field(default_factory=lambda: defaultdict(int))

    damage_dealt: Dict[RobotId, float] = field(default_factory=lambda: defaultdict(float))

    damage_taken: Dict[RobotId, float] = field(default_factory=lambda: defaultdict(float))

    objective_captures: Dict[RobotId, int] = field(default_factory=lambda: defaultdict(int))

    command_compliance: Dict[RobotId, int] = field(default_factory=lambda: defaultdict(int))

    command_violations: Dict[RobotId, int] = field(default_factory=lambda: defaultdict(int))

    # Progress-based command metrics: prev distance to target and last target cell (grid key).
    command_prev_dist: Dict[RobotId, Optional[float]] = field(default_factory=dict)

    command_last_tgt: Dict[RobotId, Optional[Tuple[int, int]]] = field(default_factory=dict)


@dataclass

class PositioningConfig:

    min_separation: int = 1
 
 
# =========================================================

# ENVIRONMENT

# =========================================================
 
GRID_W = 10

GRID_H = 10
 
# 0 = free, 1 = obstacle

GRID = [

    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],

    [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],

    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],

    [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],

    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],

    [0, 0, 1, 0, 1, 1, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],

    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],

    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

]
 
BLUE_BASE = (0, 0)

RED_BASE = (9, 9)
 
REGIONS = {

    "R1": [(1, 1), (1, 2), (2, 1), (2, 2)],

    "R2": [(7, 7), (7, 8), (8, 7), (8, 8)],

    "CENTER": [(4, 4), (4, 5), (5, 4), (5, 5)],

}
 
 
@dataclass

class EnvironmentState:

    blue_robots: Dict[RobotId, Robot]

    red_robots: Dict[RobotId, Robot]

    timestep: int = 0

    max_steps: int = 20

    done: bool = False
 
 
# =========================================================

# HELPERS

# =========================================================
 
def in_bounds(cell: Cell) -> bool:

    x, y = cell

    return 0 <= x < GRID_H and 0 <= y < GRID_W
 
 
def is_free(cell: Cell) -> bool:

    x, y = cell

    return in_bounds(cell) and GRID[x][y] == 0
 
 
def manhattan(a: Cell, b: Cell) -> int:

    return abs(a[0] - b[0]) + abs(a[1] - b[1])
 
 
def euclidean(a: Cell, b: Cell) -> float:

    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
 
 
def get_neighbors(cell: Cell) -> List[Cell]:

    x, y = cell

    candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    return [c for c in candidates if is_free(c)]
 
 
def nearest_cell(start: Cell, cells: List[Cell]) -> Cell:

    return min(cells, key=lambda c: manhattan(start, c))
 
 
def visible_obstacles(position: Cell, radius: int = 2) -> List[Cell]:

    px, py = position

    out = []

    for x in range(max(0, px - radius), min(GRID_H, px + radius + 1)):

        for y in range(max(0, py - radius), min(GRID_W, py + radius + 1)):

            if GRID[x][y] == 1:

                out.append((x, y))

    return out
 
 
def visible_regions(position: Cell, radius: int = 2) -> List[str]:

    out = []

    for region_name, cells in REGIONS.items():

        for c in cells:

            if manhattan(position, c) <= radius:

                out.append(region_name)

                break

    return out
 
 
# =========================================================

# BLOCK 1 — ENVIRONMENT INITIALIZATION

# =========================================================
 
def block1_environment_initialization() -> EnvironmentState:

    blue_robots = {

        "B1": Robot("B1", "BLUE", (0, 0)),

        "B2": Robot("B2", "BLUE", (0, 1)),

        "B3": Robot("B3", "BLUE", (1, 0)),

    }
 
    red_robots = {

        "R1": Robot("R1", "RED", (9, 9)),

        "R2": Robot("R2", "RED", (9, 8)),

        "R3": Robot("R3", "RED", (8, 9)),

    }
 
    return EnvironmentState(

        blue_robots=blue_robots,

        red_robots=red_robots,

        timestep=0,

        max_steps=20,

        done=False,

    )
 
 
# =========================================================

# BLOCK 2 — LOCAL OBSERVATION (FROM ENVIRONMENT)

# =========================================================
 
def block2_local_observation(env: EnvironmentState) -> Dict[RobotId, LocalObservation]:

    obs: Dict[RobotId, LocalObservation] = {}
 
    all_robots = {**env.blue_robots, **env.red_robots}

    for rid, robot in all_robots.items():

        if not robot.alive:

            continue
 
        teammates = []

        enemies = []
 
        for oid, other in all_robots.items():

            if oid == rid or not other.alive:

                continue

            if manhattan(robot.position, other.position) <= 3:

                if other.team == robot.team:

                    teammates.append((oid, other.position))

                else:

                    enemies.append((oid, other.position))
 
        obs[rid] = LocalObservation(

            robot_id=rid,

            team=robot.team,

            position=robot.position,

            nearby_teammates=teammates,

            nearby_enemies=enemies,

            nearby_obstacles=visible_obstacles(robot.position),

            visible_regions=visible_regions(robot.position),

            hp=robot.hp,

        )
 
    return obs
 
 
# =========================================================

# BLOCK 3 — GLOBAL STATE ENCODING (CTDE)

# =========================================================
 
def block3_global_state_encoding(env: EnvironmentState, local_obs: Dict[RobotId, LocalObservation]) -> GlobalState:

    blue_positions = {rid: r.position for rid, r in env.blue_robots.items() if r.alive}

    red_positions = {rid: r.position for rid, r in env.red_robots.items() if r.alive}
 
    contested = []

    for region_name, cells in REGIONS.items():

        blue_in = any(pos in cells for pos in blue_positions.values())

        red_in = any(pos in cells for pos in red_positions.values())

        if blue_in and red_in:

            contested.append(region_name)
 
    threat_map = {}

    for rid, robot in env.red_robots.items():

        if robot.alive:

            threat_map[rid] = float(robot.kills + 1) / float(robot.deaths + 1)
 
    return GlobalState(

        timestep=env.timestep,

        blue_positions=blue_positions,

        red_positions=red_positions,

        alive_blue=sum(1 for r in env.blue_robots.values() if r.alive),

        alive_red=sum(1 for r in env.red_robots.values() if r.alive),

        blue_base=BLUE_BASE,

        red_base=RED_BASE,

        contested_regions=contested,

        threat_map=threat_map,

    )
 
 
# =========================================================

# BLOCK 4 — SYSTEM / LLM STRATEGIC SUMMARY

# =========================================================
 
def block4_system_summary(global_state: GlobalState) -> str:

    summary = []

    summary.append(f"Timestep {global_state.timestep}")

    summary.append(f"Blue alive: {global_state.alive_blue}")

    summary.append(f"Red alive: {global_state.alive_red}")
 
    if global_state.contested_regions:

        summary.append(f"Contested regions: {', '.join(global_state.contested_regions)}")

    else:

        summary.append("No contested regions")
 
    if global_state.threat_map:

        highest_threat = max(global_state.threat_map, key=global_state.threat_map.get)

        summary.append(f"Highest red threat: {highest_threat}")

    else:

        summary.append("No red threats visible")
 
    return " | ".join(summary)


def _pyquaticus_global_state_for_block4(
    env: Any,
    blue_agents: List[Any],
    red_agents: List[Any],
    timestep: int,
) -> GlobalState:
    """
    Build sim_test `GlobalState` from PyQuaticus so `block4_system_summary` can run unchanged.
    Positions are world (x,y) rounded to integers; alive counts exclude tagged agents.
    """
    blue_positions: Dict[RobotId, Cell] = {}
    for aid in blue_agents:
        p = env.players[aid]
        blue_positions[f"R{aid}"] = (int(round(p.pos[0])), int(round(p.pos[1])))
    red_positions: Dict[RobotId, Cell] = {}
    for aid in red_agents:
        p = env.players[aid]
        red_positions[f"R{aid}"] = (int(round(p.pos[0])), int(round(p.pos[1])))
    alive_blue = sum(1 for a in blue_agents if not env.players[a].is_tagged)
    alive_red = sum(1 for a in red_agents if not env.players[a].is_tagged)
    # PyQuaticus: flags[0] = blue, flags[1] = red (see pyquaticus.structs.Team)
    bh = env.flags[0].home
    rh = env.flags[1].home
    # Optional: coarse "contested" when both teams occupy center band (world x in [45,75], y in [20,40])
    contested: List[str] = []
    for bp in blue_positions.values():
        for rp in red_positions.values():
            if (
                45 <= bp[0] <= 75
                and 45 <= rp[0] <= 75
                and 20 <= bp[1] <= 40
                and 20 <= rp[1] <= 40
                and manhattan(bp, rp) <= 25
            ):
                contested.append("CENTER_BAND")
                break
        if contested:
            break
    threat_map: Dict[RobotId, float] = {}
    for rid, pos in red_positions.items():
        threat_map[rid] = 1.0 + 0.01 * float(pos[0])
    return GlobalState(
        timestep=timestep,
        blue_positions=blue_positions,
        red_positions=red_positions,
        alive_blue=alive_blue,
        alive_red=alive_red,
        blue_base=(int(round(bh[0])), int(round(bh[1]))),
        red_base=(int(round(rh[0])), int(round(rh[1]))),
        contested_regions=contested,
        threat_map=threat_map,
    )


# =========================================================

# BLOCK 5 — HUMAN COMMAND INTERFACE (BLUE ONLY)

# =========================================================
 
def parse_blue_command(text: str) -> Optional[ParsedCommand]:

    """

    Supported:

      all move to 4 4

      robot B1 move to 3 5

      all hold region R1

      robot B2 defend 4 4

      all attack red_base

      all regroup at 2 2

      all spread

    """

    s = text.strip().lower()
 
    m = re.match(r"^all\s+(.*)$", s)

    if m:

        scope = "all"

        target = "all"

        rest = m.group(1)

    else:

        m = re.match(r"^robot\s+([a-zA-Z0-9_]+)\s+(.*)$", s)

        if not m:

            return None

        scope = "robot"

        target = m.group(1).upper()
        # Allow "B1" to refer to pyquaticus robot id "R1" (screen agent id 1).
        # Also allow "R1" directly.
        m_id = re.match(r"^[Bb](\d+)$", target)
        if m_id:
            target = f"R{m_id.group(1)}"

        rest = m.group(2)
 
    m = re.match(r"^move\s+to\s+(\d+)\s+(\d+)$", rest)

    if m:

        return ParsedCommand(text, "move", scope, target, {"cell": (int(m.group(1)), int(m.group(2)))})
 
    m = re.match(r"^hold\s+region\s+([a-zA-Z0-9_]+)$", rest)

    if m:

        return ParsedCommand(text, "hold", scope, target, {"region": m.group(1).upper()})
 
    m = re.match(r"^defend\s+(\d+)\s+(\d+)$", rest)

    if m:

        return ParsedCommand(text, "defend", scope, target, {"cell": (int(m.group(1)), int(m.group(2)))})
 
    m = re.match(r"^attack\s+red_base$", rest)

    if m:

        return ParsedCommand(text, "attack", scope, target, {"cell": RED_BASE})
 
    m = re.match(r"^regroup\s+at\s+(\d+)\s+(\d+)$", rest)

    if m:

        return ParsedCommand(text, "regroup", scope, target, {"cell": (int(m.group(1)), int(m.group(2)))})
 
    m = re.match(r"^spread$", rest)

    if m:

        return ParsedCommand(text, "spread", scope, target, {})
 
    return None
 
 
def block5_human_command_interface(first_turn: bool = False) -> HumanPlan:

    """

    On the first turn, allow human to type commands.

    On later turns, user can just type 'done' if they want to keep strategy unchanged.

    """

    if first_turn:

        print("\nEnter BLUE team commands.")

    else:

        print("\nEnter updated BLUE team commands, or type 'done' to keep current strategy.")
 
    print("Examples:")

    print("  all move to 4 4")

    print("  robot B1 move to 3 5")

    print("  all hold region R1")

    print("  robot B2 defend 4 4")

    print("  all attack red_base")

    print("  all regroup at 2 2")

    print("  all spread")
 
    print("Type 'done' OR press Enter on an empty line to start/resume simulation.\n")

    commands: List[ParsedCommand] = []
 
    while True:

        # Some terminals may send hidden unicode whitespace; normalize before checking.
        text = input("BLUE CMD > ")
        text_norm = text.replace("\u200b", "").replace("\ufeff", "").strip()

        if text_norm == "" or text_norm.lower() == "done":

            break
 
        parsed = parse_blue_command(text_norm)

        if parsed is None:

            print("Invalid command. Try again.")

            continue
 
        commands.append(parsed)

        print("Accepted:", parsed)
 
    return HumanPlan(parsed_commands=commands)
 
 
# =========================================================

# BLOCK 6 — STRATEGY ASSIGNMENT (BLUE TEAM ONLY)

# =========================================================
 
def command_applies_to_blue_robot(cmd: ParsedCommand, robot_id: str) -> bool:

    return cmd.scope == "all" or cmd.target == robot_id
 
 
def block6_blue_strategy_assignment(env: EnvironmentState, human_plan: HumanPlan) -> Dict[RobotId, Cell]:

    blue_targets: Dict[RobotId, Cell] = {}
 
    for rid, robot in env.blue_robots.items():

        if robot.alive:

            blue_targets[rid] = robot.position
 
    for cmd in human_plan.parsed_commands:

        for rid, robot in env.blue_robots.items():

            if not robot.alive or not command_applies_to_blue_robot(cmd, rid):

                continue
 
            if cmd.action in ["move", "defend", "regroup", "attack"]:

                blue_targets[rid] = cmd.params["cell"]
 
            elif cmd.action == "hold":

                region_name = cmd.params["region"]

                region_cells = REGIONS.get(region_name, [robot.position])

                blue_targets[rid] = nearest_cell(robot.position, region_cells)
 
            elif cmd.action == "spread":

                spread_cells = [(1, 8), (8, 1), (1, 5), (5, 1), (8, 5), (5, 8)]

                idx = list(env.blue_robots.keys()).index(rid) % len(spread_cells)

                blue_targets[rid] = spread_cells[idx]
 
    return blue_targets
 
 
# =========================================================

# BLOCK 7 — AUTONOMOUS STRATEGY (RED TEAM)

# =========================================================
 
def block7_red_autonomous_strategy(env: EnvironmentState) -> Dict[RobotId, Cell]:

    red_targets: Dict[RobotId, Cell] = {}
 
    blue_alive = [r for r in env.blue_robots.values() if r.alive]

    center_cells = REGIONS["CENTER"]
 
    for rid, red_robot in env.red_robots.items():

        if not red_robot.alive:

            continue
 
        if not blue_alive:

            red_targets[rid] = BLUE_BASE

            continue
 
        nearest_blue = min(blue_alive, key=lambda b: manhattan(red_robot.position, b.position))

        d = manhattan(red_robot.position, nearest_blue.position)
 
        if d <= 4:

            red_targets[rid] = nearest_blue.position

        else:

            red_targets[rid] = nearest_cell(red_robot.position, center_cells)
 
    return red_targets
 
 
# =========================================================

# BLOCK 8 — A* PATH PLANNING

# =========================================================
 
def astar(start: Cell, goal: Cell) -> List[Cell]:

    if not is_free(start) or not is_free(goal):

        return []
 
    open_heap: List[Tuple[int, Cell]] = []

    heapq.heappush(open_heap, (0, start))
 
    came_from: Dict[Cell, Cell] = {}

    g_score: Dict[Cell, int] = {start: 0}
 
    while open_heap:

        _, current = heapq.heappop(open_heap)
 
        if current == goal:

            path = []

            while current in came_from:

                path.append(current)

                current = came_from[current]

            path.append(start)

            path.reverse()

            return path
 
        for nb in get_neighbors(current):

            tentative = g_score[current] + 1

            if nb not in g_score or tentative < g_score[nb]:

                came_from[nb] = current

                g_score[nb] = tentative

                f = tentative + manhattan(nb, goal)

                heapq.heappush(open_heap, (f, nb))
 
    return []
 
 
def block8_path_planning(env: EnvironmentState, blue_targets: Dict[RobotId, Cell], red_targets: Dict[RobotId, Cell]) -> Dict[RobotId, List[Cell]]:

    all_paths: Dict[RobotId, List[Cell]] = {}
 
    for rid, robot in env.blue_robots.items():

        if robot.alive:

            goal = blue_targets.get(rid, robot.position)

            all_paths[rid] = astar(robot.position, goal)
 
    for rid, robot in env.red_robots.items():

        if robot.alive:

            goal = red_targets.get(rid, robot.position)

            all_paths[rid] = astar(robot.position, goal)
 
    return all_paths
 
 
# =========================================================

# BLOCK 9 — POSITIONING & EXECUTION LAYER

# =========================================================
 
def system_positioning_adjust(

    current_positions: Dict[RobotId, Cell],

    proposed_next_positions: Dict[RobotId, Cell],

    cfg: PositioningConfig

) -> Dict[RobotId, Cell]:

    final_positions = dict(proposed_next_positions)

    ids = list(final_positions.keys())
 
    for i in range(len(ids)):

        for j in range(i + 1, len(ids)):

            a = ids[i]

            b = ids[j]

            if final_positions[a] == final_positions[b]:

                # collision: keep a, b stays where it is

                final_positions[b] = current_positions[b]

            elif manhattan(final_positions[a], final_positions[b]) < cfg.min_separation:

                final_positions[b] = current_positions[b]
 
    return final_positions
 
 
def next_step_from_path(path: List[Cell], current: Cell) -> Cell:

    if len(path) >= 2:

        return path[1]

    return current
 
 
def resolve_combat(env: EnvironmentState, metrics: Metrics):

    blue_alive = [r for r in env.blue_robots.values() if r.alive]

    red_alive = [r for r in env.red_robots.values() if r.alive]
 
    for b in blue_alive:

        for r in red_alive:

            if not b.alive or not r.alive:

                continue
 
            if manhattan(b.position, r.position) <= 1:

                # simple engagement

                if random.random() < 0.55:

                    r.alive = False

                    r.deaths += 1

                    b.kills += 1

                    metrics.kills[b.robot_id] += 1

                    metrics.deaths[r.robot_id] += 1

                    print(f"{b.robot_id} (BLUE) eliminated {r.robot_id} (RED)")

                else:

                    b.alive = False

                    b.deaths += 1

                    r.kills += 1

                    metrics.kills[r.robot_id] += 1

                    metrics.deaths[b.robot_id] += 1

                    print(f"{r.robot_id} (RED) eliminated {b.robot_id} (BLUE)")
 
 
def block9_positioning_and_execution(

    env: EnvironmentState,

    all_paths: Dict[RobotId, List[Cell]],

    blue_targets: Dict[RobotId, Cell],

    metrics: Metrics,

    human_plan: HumanPlan,

) -> EnvironmentState:

    current_positions = {}

    proposed_positions = {}
 
    for rid, robot in {**env.blue_robots, **env.red_robots}.items():

        if robot.alive:

            current_positions[rid] = robot.position

            proposed_positions[rid] = next_step_from_path(all_paths.get(rid, []), robot.position)
 
    adjusted = system_positioning_adjust(current_positions, proposed_positions, PositioningConfig())
 
    for rid, robot in env.blue_robots.items():

        if robot.alive:

            old = robot.position

            robot.position = adjusted.get(rid, robot.position)

            robot.last_position = old

            robot.distance_covered += euclidean(old, robot.position)
 
    for rid, robot in env.red_robots.items():

        if robot.alive:

            old = robot.position

            robot.position = adjusted.get(rid, robot.position)

            robot.last_position = old

            robot.distance_covered += euclidean(old, robot.position)
 
    # Progress-based command metrics (grid): compliance at goal or moving closer to blue_targets;
    # violation only if distance to target increases. Dead agents reset state.
    _grid_cmd_eps = 1e-6
    if human_plan.parsed_commands:
        for rid, robot in env.blue_robots.items():
            if not robot.alive:
                metrics.command_prev_dist.pop(rid, None)
                metrics.command_last_tgt.pop(rid, None)
                continue
            if rid not in blue_targets:
                metrics.command_prev_dist.pop(rid, None)
                metrics.command_last_tgt.pop(rid, None)
                continue
            tgt_cell = blue_targets[rid]
            tgt_key = (int(tgt_cell[0]), int(tgt_cell[1]))
            d = float(euclidean(robot.position, tgt_cell))
            if metrics.command_last_tgt.get(rid) != tgt_key:
                metrics.command_last_tgt[rid] = tgt_key
                metrics.command_prev_dist[rid] = None
            prev = metrics.command_prev_dist.get(rid)
            if prev is None:
                metrics.command_prev_dist[rid] = d
                continue
            if robot.position == tgt_cell or d <= _grid_cmd_eps:
                metrics.command_compliance[rid] += 1
            elif d > prev + _grid_cmd_eps:
                metrics.command_violations[rid] += 1
            elif d < prev - _grid_cmd_eps:
                metrics.command_compliance[rid] += 1
            metrics.command_prev_dist[rid] = d

    resolve_combat(env, metrics)
 
    # objective capture example: center control

    for rid, robot in env.blue_robots.items():

        if robot.alive and robot.position in REGIONS["CENTER"]:

            metrics.objective_captures[rid] += 1
 
    for rid, robot in env.red_robots.items():

        if robot.alive and robot.position in REGIONS["CENTER"]:

            metrics.objective_captures[rid] += 1
 
    return env
 
 
# =========================================================

# BLOCK 10 — STRATEGIC CHANGE DETECTION

# =========================================================
 
def block10_strategic_change_detection(env: EnvironmentState, global_state: GlobalState) -> bool:

    if global_state.alive_blue <= 1:

        return True

    if global_state.alive_red <= 1:

        return True

    if global_state.contested_regions:

        return True

    return False
 
 
# =========================================================

# BLOCK 11 — LEARNING / METRICS UPDATE

# =========================================================
 
def block11_metrics_update(env: EnvironmentState, metrics: Metrics) -> Dict[str, Any]:

    player_rows = []
 
    for robot in list(env.blue_robots.values()) + list(env.red_robots.values()):

        kills = metrics.kills[robot.robot_id]

        deaths = max(1, metrics.deaths[robot.robot_id])

        misses = 0  # placeholder; add real shot tracking later
 
        precision = kills / max(1, kills + misses)

        recall = kills / max(1, kills + deaths)

        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
 
        player_rows.append({

            "robot_id": robot.robot_id,

            "team": robot.team,

            "alive": robot.alive,

            "kills": kills,

            "deaths": metrics.deaths[robot.robot_id],

            "kd": kills / deaths,

            "distance_covered": round(robot.distance_covered, 2),

            "objective_captures": metrics.objective_captures[robot.robot_id],

            "command_compliance": metrics.command_compliance[robot.robot_id],

            "command_violations": metrics.command_violations[robot.robot_id],

            "precision": round(precision, 3),

            "recall": round(recall, 3),

            "f1": round(f1, 3),

        })
 
    # team synergy placeholder: average pairwise closeness among alive teammates

    def team_synergy(team_robots: Dict[RobotId, Robot]) -> float:

        alive = [r for r in team_robots.values() if r.alive]

        if len(alive) < 2:

            return 0.0

        dists = []

        for i in range(len(alive)):

            for j in range(i + 1, len(alive)):

                dists.append(manhattan(alive[i].position, alive[j].position))

        avg_dist = sum(dists) / len(dists)

        return round(1.0 / (1.0 + avg_dist), 3)
 
    out = {

        "player_metrics": player_rows,

        "blue_synergy": team_synergy(env.blue_robots),

        "red_synergy": team_synergy(env.red_robots),

    }

    return out
 
 
# =========================================================

# BLOCK 12 — TERMINATION CHECK

# =========================================================
 
def block12_termination_check(env: EnvironmentState) -> bool:

    blue_alive = any(r.alive for r in env.blue_robots.values())

    red_alive = any(r.alive for r in env.red_robots.values())
 
    if not blue_alive or not red_alive:

        env.done = True

    elif env.timestep >= env.max_steps:

        env.done = True

    else:

        env.done = False
 
    return env.done
 
 
# =========================================================

# DISPLAY HELPERS

# =========================================================
 
def print_grid(env: EnvironmentState):

    board = [["." if GRID[x][y] == 0 else "#" for y in range(GRID_W)] for x in range(GRID_H)]
 
    for rid, r in env.blue_robots.items():

        if r.alive:

            x, y = r.position

            board[x][y] = rid[-1]
 
    for rid, r in env.red_robots.items():

        if r.alive:

            x, y = r.position

            board[x][y] = rid[-1].lower()
 
    print("\nGrid:")

    for row in board:

        print(" ".join(row))
 
 
def print_status(env: EnvironmentState, summary: str, metrics_out: Dict[str, Any]):

    print(f"\n===== STEP {env.timestep} =====")

    print(summary)
 
    print("\nBLUE TEAM:")

    for r in env.blue_robots.values():

        print(f"  {r.robot_id}: pos={r.position}, alive={r.alive}, kills={r.kills}, deaths={r.deaths}")
 
    print("\nRED TEAM:")

    for r in env.red_robots.values():

        print(f"  {r.robot_id}: pos={r.position}, alive={r.alive}, kills={r.kills}, deaths={r.deaths}")
 
    print("\nTeam Synergy:")

    print("  BLUE:", metrics_out["blue_synergy"])

    print("  RED :", metrics_out["red_synergy"])
 
    print_grid(env)
 
 
# =========================================================

# MAIN LOOP

# =========================================================
 
def main():

    random.seed(42)

    # Block 1

    env = block1_environment_initialization()

    metrics = Metrics()
 
    # Block 5 initial commands

    human_plan = block5_human_command_interface(first_turn=True)
 
    while True:

        env.timestep += 1
 
        # Block 2

        local_obs = block2_local_observation(env)
 
        # Block 3

        global_state = block3_global_state_encoding(env, local_obs)
 
        # Block 4

        summary = block4_system_summary(global_state)
 
        # Optional command refresh when Block 10 detects a strategic change.

        if env.timestep > 1:

            replan_block10 = block10_strategic_change_detection(env, global_state)

            if replan_block10:

                print("\nReplan triggered.")

                new_plan = block5_human_command_interface(first_turn=False)

                if new_plan.parsed_commands:

                    human_plan = new_plan
 
        # Block 6

        blue_targets = block6_blue_strategy_assignment(env, human_plan)
 
        # Block 7

        red_targets = block7_red_autonomous_strategy(env)
 
        # Block 8

        all_paths = block8_path_planning(env, blue_targets, red_targets)
 
        # Block 9

        env = block9_positioning_and_execution(env, all_paths, blue_targets, metrics, human_plan)
 
        # Block 11

        metrics_out = block11_metrics_update(env, metrics)
 
        # print step

        print_status(env, summary, metrics_out)
 
        # Block 12

        done = block12_termination_check(env)

        if done:

            break
 
    print("\n===== FINAL METRICS =====")

    for row in metrics_out["player_metrics"]:

        print(row)
 
    blue_alive = any(r.alive for r in env.blue_robots.values())

    red_alive = any(r.alive for r in env.red_robots.values())
 
    if blue_alive and not red_alive:

        print("\nBLUE team wins.")

    elif red_alive and not blue_alive:

        print("\nRED team wins.")

    else:

        print("\nMatch ended by time limit or draw.")
 
def main_pyquaticus() -> None:
    """
    pyquaticus runner that reuses your Block 5/6/7/8 parsing + assignment pipeline.

    Command syntax matches this file's Block 5 (`parse_blue_command`):
      - all move to 40 30
      - robot B1 move to 35 20   (B# is accepted as an alias for R#)
      - all hold region R1
      - robot R2 defend 60 30
      - all attack red_base
      - all regroup at 60 30
      - all spread

    Press R anytime to clear the current plan and prompt for new commands (same frame).
    """
    import pygame
    import threading
    import queue
    import json
    import os
    from datetime import datetime
    from pathlib import Path
    import sys as _sys

    from pyquaticus.envs.pyquaticus import PyQuaticusEnv
    from pyquaticus.config import ACTION_MAP
    from pyquaticus.structs import Team
    from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
    from dashboard_command_tracker import GameAnalyticsTracker

    from all_blocks import (
        Config,
        HRLAction,
        block1_environment_initialization,
        block2_state_collection,
        block3_global_state_encoding,
        block7_dispatch_assignments,
        block8_low_level_execution,
        block12_termination_check,
    )

    WALL_SAFETY_DIST = 5.0
    DEFENDER_HOLD_INSET = 15.0

    def _angle_diff(a_deg: float, b_deg: float) -> float:
        return ((a_deg - b_deg) + 180) % 360 - 180

    def _wall_avoid_direction(obs, current_direction: str, agent_heading: float) -> str:
        best_override = None
        min_dist = 1e9
        for i in range(4):
            d = obs.get(f"wall_{i}_distance", 1e9)
            b = obs.get(f"wall_{i}_bearing", 0)

            # Trigger when close; if very close (< 6m) also trigger for walls to the side.
            in_front = -90 <= b <= 90
            very_close = d < 6.0
            if d < WALL_SAFETY_DIST and d < min_dist and (in_front or very_close):
                min_dist = d
                away_rel = (b + 180) % 360
                abs_desired = (agent_heading + away_rel) % 360
                if abs_desired < 0:
                    abs_desired += 360
                if abs_desired >= 337.5 or abs_desired < 22.5:
                    best_override = "N"
                elif 22.5 <= abs_desired < 67.5:
                    best_override = "NE"
                elif 67.5 <= abs_desired < 112.5:
                    best_override = "E"
                elif 112.5 <= abs_desired < 157.5:
                    best_override = "SE"
                elif 157.5 <= abs_desired < 202.5:
                    best_override = "S"
                elif 202.5 <= abs_desired < 247.5:
                    best_override = "SW"
                elif 247.5 <= abs_desired < 292.5:
                    best_override = "W"
                else:
                    best_override = "NW"
        return best_override if best_override is not None else current_direction

    def block_direction_to_env_action(player_heading: float, direction_str: str) -> int:
        """Convert block direction (W/N/E/etc) to pyquaticus discrete action index."""
        if direction_str == "HOLD":
            return 8
        abs_heading = {
            "W": 270,
            "NW": 315,
            "N": 0,
            "NE": 45,
            "E": 90,
            "SE": 135,
            "S": 180,
            "SW": 225,
        }
        desired = abs_heading.get(direction_str, 0)
        rel = _angle_diff(desired, player_heading)
        best_i = 0
        best_err = abs(_angle_diff(ACTION_MAP[0][1], rel))
        for i in range(1, 8):
            err = abs(_angle_diff(ACTION_MAP[i][1], rel))
            if err < best_err:
                best_err = err
                best_i = i
        return best_i

    def _region_to_world(region: str, blue_zone: Tuple[float, float], red_zone: Tuple[float, float]) -> Tuple[float, float]:
        r = region.upper()
        if r in ("R1", "BLUE", "BLUE_BASE", "HOME"):
            return blue_zone
        if r in ("R2", "RED", "RED_BASE"):
            return red_zone
        if r in ("CENTER", "MID", "MIDDLE"):
            return (60.0, 30.0)
        # Fallback: center field
        return (60.0, 30.0)

    def _build_assignments_from_block5(
        blue_rids: list[str],
        plan: HumanPlan,
        blue_zone: Tuple[float, float],
        red_zone: Tuple[float, float],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Translate sim_test Block 5 commands into all_blocks-style per-robot assignments
        that Block 8 understands via `hl_assignment`.
        """
        out: Dict[str, Dict[str, Any]] = {rid: {"subgoal": "IDLE", "target_cell": None} for rid in blue_rids}

        # Some spread waypoints (world coords) that keep agents inside the play area.
        spread_waypoints = [
            (25.0, 20.0),
            (25.0, 40.0),
            (45.0, 15.0),
            (45.0, 45.0),
            (70.0, 20.0),
            (70.0, 40.0),
        ]

        for cmd in plan.parsed_commands:
            for rid in blue_rids:
                if not (cmd.scope == "all" or (cmd.scope == "robot" and cmd.target == rid)):
                    continue

                if cmd.action in ("move", "regroup", "defend"):
                    x, y = cmd.params["cell"]
                    out[rid] = {"subgoal": "REGROUP", "target_cell": (float(x), float(y))}
                elif cmd.action == "attack":
                    out[rid] = {"subgoal": "ATTACK_RED_BASE", "target_cell": red_zone}
                elif cmd.action == "hold":
                    region = str(cmd.params["region"])
                    goal = _region_to_world(region, blue_zone, red_zone)
                    out[rid] = {"subgoal": f"HOLD_{region}", "target_cell": goal}
                elif cmd.action == "spread":
                    i = blue_rids.index(rid) % len(spread_waypoints)
                    out[rid] = {"subgoal": "SPREAD", "target_cell": spread_waypoints[i]}

        return out

    def _prompt_commands_nonblocking(
        env_obj: Any,
        clock_obj: Any,
        first_turn: bool,
    ) -> HumanPlan:
        """
        Collect terminal commands without freezing pygame rendering.
        Runs blocking input in a background thread while main thread keeps rendering.
        """
        result_q: "queue.Queue[Any]" = queue.Queue(maxsize=1)

        def _worker() -> None:
            try:
                result_q.put(block5_human_command_interface(first_turn=first_turn))
            except Exception as exc:  # pragma: no cover
                result_q.put(exc)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        while t.is_alive():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env_obj.close()
                    pygame.quit()
                    _sys.exit(0)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env_obj.close()
                    pygame.quit()
                    _sys.exit(0)
            try:
                env_obj.render()
            except Exception:
                pass
            clock_obj.tick(30)

        out = result_q.get()
        if isinstance(out, Exception):
            raise out
        return out

    pygame.init()

    team_size = int(input("Enter team size (1–6): \n"))
    pos_choice = input(
        "\nSelect starting position:\n"
        "(1) Random\n"
        "(Any other key) Default - Straight line\n"
    )
    start_pos = "random" if pos_choice == "1" else "default"

    mode_choice = input(
        "\nSelect mode (red team base_combined):\n"
        "(1) hard\n"
        "(2) medium\n"
        "(any other key) easy\n"
    )
    mode = "hard" if mode_choice == "1" else "medium" if mode_choice == "2" else "easy"

    env = PyQuaticusEnv(
        team_size=team_size,
        render_mode="human",
        render_agent_ids=True,
        start_pos=start_pos,
    )

    env.reset()
    # Make sure the window is created/shown before blocking on terminal input().
    try:
        env.render()
        pygame.event.pump()
    except Exception:
        # Rendering can be backend-dependent; ignore and continue.
        pass
    blue_agents = env.agents[:team_size]
    red_agents = env.agents[team_size:]

    cfg = Config(num_robots=2 * team_size)
    env.global_state = block1_environment_initialization(cfg)
    gs = env.global_state
    gs.history.setdefault("actions", [])
    gs.history.setdefault("rewards", [])
    gs.history.setdefault("pyquaticus_steps", [])
    gs.history.setdefault("block4_summaries", [])
    gs.history.setdefault("block11_metrics", [])
    _block4_print_every = int(os.environ.get("BLOCK4_PRINT_EVERY", "30"))
    _metrics_every = int(os.environ.get("METRICS_EVERY", "10"))

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = logs_dir / f"pyquaticus_run_{run_stamp}.jsonl"
    print(f"[logging] Writing JSONL step log to: {jsonl_path}")

    # Dashboard tracker (command compliance/violation + event ingest)
    tracker = GameAnalyticsTracker()
    match_id = f"pyq_{run_stamp}"
    tracker.start_match(match_id, timestamp=0.0)

    # Block 11 integration state (pyquaticus -> sim_test Metrics adapter).
    metrics = Metrics()
    metric_robots: Dict[str, Robot] = {}
    prev_pos: Dict[str, Tuple[float, float]] = {}
    prev_tagged: Dict[str, bool] = {}
    prev_scores = {
        "blue": int(env.game_score.get("blue_captures", 0)),
        "red": int(env.game_score.get("red_captures", 0)),
    }
    for aid in env.agents:
        rid = f"R{aid}"
        team_name = "BLUE" if aid in blue_agents else "RED"
        p = env.players[aid]
        pos_i = (int(round(p.pos[0])), int(round(p.pos[1])))
        metric_robots[rid] = Robot(robot_id=rid, team=team_name, position=pos_i, alive=not p.is_tagged)
        prev_pos[rid] = (float(p.pos[0]), float(p.pos[1]))
        prev_tagged[rid] = bool(p.is_tagged)

    blue_policies = [
        Heuristic_CTF_Agent(agent_id=aid, team=Team.BLUE_TEAM, mode=mode) for aid in blue_agents
    ]
    red_policies = [
        Heuristic_CTF_Agent(agent_id=aid, team=Team.RED_TEAM, mode=mode) for aid in red_agents
    ]

    paused = False
    clock = pygame.time.Clock()
    user_replan_requested = False
    print("[PyQuaticus] Controls: SPACE pause | ESC quit | R strategic replan")

    # Prompt user per-plan; Block 10 decides when to clear for a new prompt.
    human_plan = None
    robot_assignments = None
    # When replan clears `human_plan`, Block 5 "done" / empty line should reuse this snapshot.
    last_committed_human_plan: Optional[HumanPlan] = None
    step_count = 0
    # Snapshot CTF state at plan time for score/flag-driven replans.
    last_replan_ctf_state: Optional[Dict[str, Any]] = None
    last_user_commands: List[Dict[str, Any]] = []
    last_block10_state: bool = False  # rising-edge detect to avoid replan spam
    # Previous world position per agent for dashboard attack compliance (moving toward goal).
    dash_prev_pos: Dict[str, Tuple[float, float]] = {}
    # Previous min pairwise blue distance for dashboard spread_eval progress.
    dash_spread_prev_min: Optional[float] = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                _sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    env.close()
                    pygame.quit()
                    _sys.exit(0)
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    user_replan_requested = True

        if paused:
            env.render()
            clock.tick(30)
            continue

        actions = {}
        gs = env.global_state
        gs.t = step_count

        blue_rids = [f"R{aid}" for aid in blue_agents]
        gs.robots = {
            rid: {
                "position": (env.players[blue_agents[i]].pos[0], env.players[blue_agents[i]].pos[1]),
                "has_flag": env.players[blue_agents[i]].has_flag,
                "hp": gs.robots.get(rid, {}).get("hp", 100),
            }
            for i, rid in enumerate(blue_rids)
        }

        # Because gs.robots is rebuilt each frame from env state, we must re-attach hl assignments.
        if robot_assignments is not None:
            block7_dispatch_assignments(gs, robot_assignments)

        blue_env_obs = {aid: env.state_to_obs(aid, normalize=False) for aid in blue_agents}
        local_obs = block2_state_collection(gs, env_obs=blue_env_obs)
        gs = block3_global_state_encoding(gs, local_obs)

        ctf_state = {
            "score_blue": env.game_score["blue_captures"],
            "score_red": env.game_score["red_captures"],
            "blue_has_red_flag": any(env.players[aid].has_flag for aid in blue_agents),
            "red_has_blue_flag": any(env.players[aid].has_flag for aid in red_agents),
        }

        if user_replan_requested:
            print("[Replan] User requested (R key) — enter new BLUE commands or 'done' to repeat last plan.")
            human_plan = None
            robot_assignments = None
            last_replan_ctf_state = dict(ctf_state)
            user_replan_requested = False

        # Block 4 — same `block4_system_summary` as grid-world, fed by PyQuaticus state.
        gs_block4 = _pyquaticus_global_state_for_block4(env, blue_agents, red_agents, step_count)
        block4_summary = block4_system_summary(gs_block4)
        gs.history["block4_summaries"].append({"t": step_count, "summary": block4_summary})
        if _block4_print_every > 0 and step_count % _block4_print_every == 0:
            print(f"[Block4] {block4_summary}")

        # World goals for Block 8 mapping (also used to translate hold/attack targets).
        _red_home = env.flags[int(Team.RED_TEAM)].home
        _blue_home = env.flags[int(Team.BLUE_TEAM)].home
        red_zone_center = (float(_red_home[0]), float(_red_home[1]))
        blue_zone_center = (float(_blue_home[0]), float(_blue_home[1]))

        if human_plan is None:
            # Use sim_test.py Block 5 interface (move/hold/defend/attack/regroup/spread).
            new_plan = _prompt_commands_nonblocking(
                env_obj=env,
                clock_obj=clock,
                first_turn=(step_count == 0),
            )
            if new_plan.parsed_commands:
                human_plan = new_plan
            elif last_committed_human_plan is not None and last_committed_human_plan.parsed_commands:
                human_plan = HumanPlan(
                    parsed_commands=list(last_committed_human_plan.parsed_commands)
                )
            else:
                # No commands entered and nothing to repeat (robots will idle/hold).
                human_plan = HumanPlan(parsed_commands=[])

            if human_plan.parsed_commands:
                last_committed_human_plan = HumanPlan(
                    parsed_commands=list(human_plan.parsed_commands)
                )

            robot_assignments = _build_assignments_from_block5(
                blue_rids, human_plan, blue_zone_center, red_zone_center
            )
            block7_dispatch_assignments(gs, robot_assignments)
            last_replan_ctf_state = dict(ctf_state)
            last_user_commands = [
                {
                    "raw_text": c.raw_text,
                    "action": c.action,
                    "scope": c.scope,
                    "target": c.target,
                    "params": c.params,
                }
                for c in (human_plan.parsed_commands or [])
            ]

            # Register/refresh dashboard commands (simple move-to compliance).
            # We use current step_count as a logical timestamp (also written to JSONL).
            # Expire after 120 "seconds" (steps) by default.
            issued_at = float(step_count)
            expires_at = float(step_count + 120)
            for idx, c in enumerate(human_plan.parsed_commands or []):
                if c.action in ("move", "regroup", "defend") and "cell" in c.params:
                    tx, ty = c.params["cell"]
                    # scope mapping: sim_test cmd scope is all/robot; translate to dashboard scopes.
                    if c.scope == "robot":
                        scope = "player"
                        target = str(c.target)
                    else:
                        scope = "all"
                        target = "all"
                    cmd = tracker.make_move_command(
                        command_id=f"cmd_{step_count}_{idx}",
                        issued_at=issued_at,
                        expires_at=expires_at,
                        scope=scope,
                        target=target,
                        text=c.raw_text,
                        target_x=float(tx),
                        target_y=float(ty),
                    )
                    tracker.add_command(match_id, cmd)

            # Hold / attack / spread: map assignments to dashboard command semantics.
            _dc = 0
            for _rid in blue_rids:
                _a = robot_assignments.get(_rid) or {}
                _sg = str(_a.get("subgoal", "IDLE"))
                _tc = _a.get("target_cell")
                if _tc is None or not hasattr(_tc, "__len__") or len(_tc) < 2:
                    continue
                _tx, _ty = float(_tc[0]), float(_tc[1])
                if _sg.startswith("HOLD_"):
                    tracker.add_command(
                        match_id,
                        tracker.make_hold_command(
                            command_id=f"cmd_{step_count}_hold_{_dc}",
                            issued_at=issued_at,
                            expires_at=expires_at,
                            text="hold",
                            anchor_x=_tx,
                            anchor_y=_ty,
                            player_id=_rid,
                        ),
                    )
                    _dc += 1
                elif _sg == "ATTACK_RED_BASE":
                    tracker.add_command(
                        match_id,
                        tracker.make_attack_command(
                            command_id=f"cmd_{step_count}_attack_{_dc}",
                            issued_at=issued_at,
                            expires_at=expires_at,
                            text="attack",
                            goal_x=float(red_zone_center[0]),
                            goal_y=float(red_zone_center[1]),
                            player_id=_rid,
                        ),
                    )
                    _dc += 1
            if any(
                str((robot_assignments.get(br) or {}).get("subgoal", "")) == "SPREAD"
                for br in blue_rids
            ):
                tracker.add_command(
                    match_id,
                    tracker.make_spread_command(
                        command_id=f"cmd_{step_count}_spread",
                        issued_at=issued_at,
                        expires_at=expires_at,
                        text="spread",
                    ),
                )

            print("Applied per-robot assignments:")
            for rid in blue_rids:
                print(f"  {rid}: {robot_assignments.get(rid)}")

        hrl = HRLAction(teams={}, subgoals={}, request_replan=False)

        wx, wy = 120.0, 60.0
        blue_hold_x = max(DEFENDER_HOLD_INSET, min(wx - DEFENDER_HOLD_INSET, blue_zone_center[0]))
        blue_hold_y = max(22.0, min(38.0, blue_zone_center[1]))
        defender_hold_world = (blue_hold_x, blue_hold_y)
        flank_hold_world = (70.0, max(22.0, min(wy - DEFENDER_HOLD_INSET, 45.0)))

        robot_actions = block8_low_level_execution(
            gs,
            local_obs,
            hrl,
            opponent_flag_world=red_zone_center,
            own_flag_world=blue_zone_center,
            defender_hold_world=defender_hold_world,
            flank_hold_world=flank_hold_world,
            robot_assignments=robot_assignments,
        )

        # Compute base_combined actions for all agents (blue+red), then override blue where human assigns.
        obs_dict = {aid: env.state_to_obs(aid, normalize=False) for aid in env.agents}
        base_actions_blue: Dict[int, int] = {}
        for j, agent_id in enumerate(blue_agents):
            act_17 = blue_policies[j].compute_action(obs_dict)
            base_actions_blue[agent_id] = 8 if act_17 == 16 else act_17 % 8

        # Blue actions.
        for i, agent_id in enumerate(blue_agents):
            rid = f"R{agent_id}"
            # Default: base_combined policy action.
            final_act = base_actions_blue.get(agent_id, 8)

            # If the human provided a non-IDLE assignment for this robot, override base action.
            if robot_assignments is not None:
                a = robot_assignments.get(rid) or {}
                sg = str(a.get("subgoal", "IDLE"))
                if sg != "IDLE":
                    block_action = "HOLD"
                    for ra in robot_actions:
                        if ra.get("robot_id") == rid:
                            block_action = ra.get("action", "HOLD")
                            break

                    player = env.players[agent_id]
                    hl_a = gs.robots.get(rid, {}).get("hl_assignment") or {}
                    sub = str(hl_a.get("subgoal", ""))
                    is_defender = gs.robots.get(rid, {}).get("team") == "Defend" or (
                        sub.startswith("HOLD") or sub.startswith("PROTECT") or sub == "IDLE"
                    )
                    if not is_defender:
                        block_action = _wall_avoid_direction(
                            blue_env_obs[agent_id], block_action, player.heading
                        )
                    final_act = block_direction_to_env_action(player.heading, block_action)

            actions[agent_id] = int(final_act)

        # Red actions.
        for j, agent_id in enumerate(red_agents):
            act_17 = red_policies[j].compute_action(obs_dict)
            actions[agent_id] = 8 if act_17 == 16 else act_17 % 8

        # Pre-step carrier snapshot for capture attribution.
        pre_step_has_flag = {aid: bool(env.players[aid].has_flag) for aid in env.agents}

        # Step env and keep raw return for reward logging across API variants.
        step_out = env.step(actions)
        t_now = step_count

        # Parse reward from env.step return in a version-tolerant way.
        reward_payload: Any = None
        if isinstance(step_out, tuple) and len(step_out) >= 2:
            reward_payload = step_out[1]

        # In-memory history logging (all_blocks-style).
        for aid, act in actions.items():
            gs.history["actions"].append({"t": t_now, "robot_id": f"R{aid}", "action": int(act)})
        gs.history["rewards"].append({"t": t_now, "value": reward_payload, "reason": "pyquaticus_env_step"})

        # Build a JSON-safe reward summary for file logging.
        reward_summary: Dict[str, Any]
        if isinstance(reward_payload, dict):
            reward_summary = {}
            for k, v in reward_payload.items():
                try:
                    reward_summary[str(k)] = float(v)
                except Exception:
                    reward_summary[str(k)] = str(v)
        elif reward_payload is None:
            reward_summary = {"raw": None}
        else:
            try:
                reward_summary = {"raw": float(reward_payload)}
            except Exception:
                reward_summary = {"raw": str(reward_payload)}

        # ---------- Block 11 adapter updates ----------
        # Refresh per-robot adapter state from live pyquaticus players.
        for aid in env.agents:
            rid = f"R{aid}"
            p = env.players[aid]
            now_pos_f = (float(p.pos[0]), float(p.pos[1]))
            now_pos_i = (int(round(p.pos[0])), int(round(p.pos[1])))
            old_pos_f = prev_pos.get(rid, now_pos_f)

            mr = metric_robots[rid]
            mr.last_position = mr.position
            mr.position = now_pos_i
            mr.distance_covered += euclidean(
                (int(round(old_pos_f[0])), int(round(old_pos_f[1]))), now_pos_i
            )
            mr.alive = not bool(p.is_tagged)

            was_tagged = prev_tagged.get(rid, bool(p.is_tagged))
            if (not was_tagged) and bool(p.is_tagged):
                metrics.deaths[rid] += 1
                mr.deaths += 1

            prev_pos[rid] = now_pos_f
            prev_tagged[rid] = bool(p.is_tagged)

        # Attribute captures using pre-step carriers when score increases.
        new_blue_score = int(env.game_score.get("blue_captures", 0))
        new_red_score = int(env.game_score.get("red_captures", 0))
        if new_blue_score > prev_scores["blue"]:
            carriers = [f"R{aid}" for aid in blue_agents if pre_step_has_flag.get(aid, False)]
            for rid in (carriers or []):
                metrics.objective_captures[rid] += 1
        if new_red_score > prev_scores["red"]:
            carriers = [f"R{aid}" for aid in red_agents if pre_step_has_flag.get(aid, False)]
            for rid in (carriers or []):
                metrics.objective_captures[rid] += 1
        prev_scores["blue"] = new_blue_score
        prev_scores["red"] = new_red_score

        # Dashboard event ingest: emit a movement event for each player.
        # Uses world (x,y) and team id ("BLUE"/"RED") so command evaluation can run.
        # prev_x/prev_y support attack-command compliance (moving toward goal vs drifting).
        for aid in env.agents:
            pid = f"R{aid}"
            p = env.players[aid]
            cx, cy = float(p.pos[0]), float(p.pos[1])
            prev = dash_prev_pos.get(pid, (cx, cy))
            tracker.ingest_event(
                {
                    "match_id": match_id,
                    "timestamp": float(t_now),
                    "event": "move",
                    "player_id": pid,
                    "team_id": "BLUE" if aid in blue_agents else "RED",
                    "x": cx,
                    "y": cy,
                    "prev_x": float(prev[0]),
                    "prev_y": float(prev[1]),
                    "is_tagged": bool(p.is_tagged),
                    "has_flag": bool(p.has_flag),
                }
            )
            dash_prev_pos[pid] = (cx, cy)

        # Spread command: one team-level snapshot per step (min pairwise distance among blue).
        if len(blue_agents) < 2:
            min_pair = 1e9
        else:
            bp = [env.players[aid].pos for aid in blue_agents]
            min_pair = float("inf")
            for i in range(len(bp)):
                for j in range(i + 1, len(bp)):
                    d = math.hypot(float(bp[i][0]) - float(bp[j][0]), float(bp[i][1]) - float(bp[j][1]))
                    if d < min_pair:
                        min_pair = d
        tracker.ingest_event(
            {
                "match_id": match_id,
                "timestamp": float(t_now),
                "event": "spread_eval",
                "player_id": "__spread__",
                "team_id": "BLUE",
                "min_pairwise_blue": min_pair,
                "prev_min_pairwise_blue": dash_spread_prev_min,
            }
        )
        dash_spread_prev_min = min_pair

        # Progress-based command metrics (PyQuaticus): compliance at goal or moving closer;
        # violation only if distance to target increases. Skip while tagged or carrying flag
        # (return-home / carrier behavior is not scored against the human waypoint).
        _pq_cmd_eps = 0.35
        _pq_goal_ok = 4.0
        if robot_assignments:
            for rid in blue_rids:
                try:
                    aid = int(rid[1:]) if rid.startswith("R") else int(rid)
                except ValueError:
                    continue
                if bool(env.players[aid].is_tagged):
                    metrics.command_prev_dist.pop(rid, None)
                    metrics.command_last_tgt.pop(rid, None)
                    continue
                if bool(env.players[aid].has_flag):
                    metrics.command_prev_dist.pop(rid, None)
                    metrics.command_last_tgt.pop(rid, None)
                    continue
                a = robot_assignments.get(rid) or {}
                sg = str(a.get("subgoal", "IDLE"))
                if sg == "IDLE":
                    metrics.command_prev_dist.pop(rid, None)
                    metrics.command_last_tgt.pop(rid, None)
                    continue
                tc = a.get("target_cell")
                if not (tc and hasattr(tc, "__len__") and len(tc) >= 2):
                    continue
                tgt_key = (int(round(float(tc[0]))), int(round(float(tc[1]))))
                cur = metric_robots[rid].position
                d = float(euclidean(tgt_key, cur))
                if metrics.command_last_tgt.get(rid) != tgt_key:
                    metrics.command_last_tgt[rid] = tgt_key
                    metrics.command_prev_dist[rid] = None
                prev = metrics.command_prev_dist.get(rid)
                if prev is None:
                    metrics.command_prev_dist[rid] = d
                    continue
                if d <= _pq_goal_ok:
                    metrics.command_compliance[rid] += 1
                elif d > prev + _pq_cmd_eps:
                    metrics.command_violations[rid] += 1
                elif d < prev - _pq_cmd_eps:
                    metrics.command_compliance[rid] += 1
                metrics.command_prev_dist[rid] = d

        blue_metric_robots = {rid: r for rid, r in metric_robots.items() if rid in blue_rids}
        red_metric_robots = {rid: r for rid, r in metric_robots.items() if rid not in blue_rids}
        metrics_env = EnvironmentState(
            blue_robots=blue_metric_robots,
            red_robots=red_metric_robots,
            timestep=t_now,
            max_steps=cfg.max_steps_per_episode,
            done=bool(getattr(env, "dones", {}).get("__all__", False)),
        )
        metrics_out: Optional[Dict[str, Any]] = None
        if (_metrics_every > 0 and t_now % _metrics_every == 0) or bool(getattr(env, "dones", {}).get("__all__", False)):
            metrics_out = block11_metrics_update(metrics_env, metrics)
            gs.history["block11_metrics"].append({"t": t_now, "metrics": metrics_out})
            if _metrics_every > 0 and t_now % _metrics_every == 0:
                print(
                    f"[Block11] t={t_now} blue_synergy={metrics_out['blue_synergy']} "
                    f"red_synergy={metrics_out['red_synergy']}"
                )

        # File logging: one JSON record per env step (full detail).
        dash_status_raw = tracker.command_status.get(match_id, {})
        dash_status_json = {
            str(cmd_id): {
                str(entity): {
                    "compliant_events": int(st.compliant_events),
                    "violation_events": int(st.violation_events),
                    "first_compliance_time": (
                        None if st.first_compliance_time is None else float(st.first_compliance_time)
                    ),
                    "active_time": float(getattr(st, "active_time", 0.0)),
                }
                for entity, st in (entity_map or {}).items()
            }
            for cmd_id, entity_map in (dash_status_raw or {}).items()
        }
        step_record = {
            "t": t_now,
            "block4_summary": block4_summary,
            "user_commands": last_user_commands,
            "score": {
                "blue": int(env.game_score.get("blue_captures", 0)),
                "red": int(env.game_score.get("red_captures", 0)),
            },
            "actions": {str(aid): int(act) for aid, act in actions.items()},
            "reward": reward_summary,
            "ctf_state": ctf_state,
            "robot_assignments": robot_assignments or {},
            "block11_metrics": metrics_out,
            "dashboard_command_status": {
                "active_commands": len(tracker.active_commands.get(match_id, [])),
                "status": dash_status_json,
            },
            "done": bool(getattr(env, "dones", {}).get("__all__", False)),
            "message": str(getattr(env, "message", "")),
        }
        gs.history["pyquaticus_steps"].append(step_record)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(step_record, ensure_ascii=True) + "\n")

        step_count += 1
        gs.t = step_count

        done, reason = block12_termination_check(gs, cfg, env)
        if done:
            print(f"\n--- {reason} ---")
            break

        # Block 10 (your grid-world definition): alive thresholds + contested regions.
        # For PyQuaticus, we reuse the env-fed GlobalState built for Block 4.
        # Note: `_pyquaticus_global_state_for_block4` only needs env + robot ids.
        gs_block4_after = _pyquaticus_global_state_for_block4(
            env, blue_agents, red_agents, gs.t
        )
        should_replan_block10 = block10_strategic_change_detection(env, gs_block4_after)
        # Only replan on the rising edge of Block 10 condition (False -> True).
        block10_rising = should_replan_block10 and (not last_block10_state)
        last_block10_state = should_replan_block10
        red_scored_since_plan = False
        red_took_flag_since_plan = False
        if last_replan_ctf_state is not None:
            red_scored_since_plan = (
                ctf_state.get("score_red", 0) > last_replan_ctf_state.get("score_red", 0)
            )
            red_took_flag_since_plan = (
                bool(ctf_state.get("red_has_blue_flag", False))
                and not bool(last_replan_ctf_state.get("red_has_blue_flag", False))
            )
        should_replan = (
            block10_rising
            or red_scored_since_plan
            or red_took_flag_since_plan
        )
        if should_replan:
            if red_scored_since_plan:
                print("[Replan] Trigger: red score increased.")
            elif red_took_flag_since_plan:
                print("[Replan] Trigger: red has blue flag.")
            elif block10_rising:
                print("[Replan] Trigger: Block 10 strategic change.")
            human_plan = None
            robot_assignments = None
            last_replan_ctf_state = dict(ctf_state)

        env.render()
        clock.tick(30)

 
if __name__ == "__main__":
    import sys
    if "--grid" in sys.argv:
        main()
    else:
        main_pyquaticus()
 