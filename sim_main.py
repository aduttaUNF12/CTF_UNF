"""
Capture the Flag simulation (HRL + human-in-the-loop).
- Blue team: all_blocks pipeline (blocks 1–8) with A* low-level execution; Block 10 triggers
  replan (return to Human+LLM) when strategic change detected.
- Red team: base_combined heuristic policy (attack/defend).
- Action space: 9 discrete actions (8 directions + hold).
"""
import sys
import pygame
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from pyquaticus.config import ACTION_MAP, config_dict_std
from pyquaticus.structs import Team
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from all_blocks import (
    Config,
    block1_environment_initialization,
    block2_state_collection,
    block3_global_state_encoding,
    block4_llm_state_summarization,
    block5_human_intervention,
    block6_high_level_rl_manager,
    block7_subgoal_dispatching,
    block8_low_level_execution,
    block10_check_strategic_changes,
    block12_termination_check,
)

# Wall avoidance: steer away when distance to any wall is below this (meters)
WALL_SAFETY_DIST = 18.0

# Defenders hold near home base but inset from boundaries so they don't run into walls (meters).
DEFENDER_HOLD_INSET = 15.0

def _angle_diff(a_deg: float, b_deg: float) -> float:
    """Angular difference in [-180, 180] degrees."""
    return ((a_deg - b_deg) + 180) % 360 - 180


def _wall_avoid_direction(obs, current_direction: str, agent_heading: float) -> str:
    """
    If any wall is close and in front half (bearing in [-90, 90]), override to
    steer away. Wall bearing is relative to agent heading; we convert to absolute
    direction. Uses unnormalized obs: wall_i_distance, wall_i_bearing.
    """
    best_override = None
    min_dist = 1e9
    for i in range(4):
        d = obs.get(f"wall_{i}_distance", 1e9)
        b = obs.get(f"wall_{i}_bearing", 0)
        # Trigger when close; if very close (< 6m) also trigger for walls to the side
        in_front = -90 <= b <= 90
        very_close = d < 6.0
        if d < WALL_SAFETY_DIST and d < min_dist and (in_front or very_close):
            min_dist = d
            # b is relative to agent; away = turn 180 in agent frame
            away_rel = (b + 180) % 360
            # Convert to absolute heading (maritime: N=0, E=90)
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
    """
    Convert a block direction (W, NW, N, ...) to pyquaticus 9-action index.
    Env uses relative heading; maritime convention N=0, E=90, S=180, W=270.
    """
    if direction_str == "HOLD":
        return 8
    abs_heading = {
        "W": 270, "NW": 315, "N": 0, "NE": 45,
        "E": 90, "SE": 135, "S": 180, "SW": 225,
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


def main():
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

    obs = env.reset()
    blue_agents = env.agents[:team_size]
    red_agents = env.agents[team_size:]

    cfg = Config(num_robots=2 * team_size)
    env.global_state = block1_environment_initialization(cfg)

    red_policies = [
        Heuristic_CTF_Agent(agent_id=aid, team=Team.RED_TEAM, mode=mode)
        for aid in red_agents
    ]

    paused = False
    clock = pygame.time.Clock()
    # Prompt human once per episode (or when Block 10 requests replan); reuse plan otherwise
    human_plan = None
    step_count = 0
    last_replan_step = -1
    last_replan_ctf_state = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    env.close()
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            actions = {}

            gs = env.global_state
            gs.t = step_count  # for Block 10 (check strategic changes / replan)
            # Restrict gs.robots to blue agents only; use R0-R5 to match screen agent IDs (0-5)
            blue_rids = [f"R{aid}" for aid in blue_agents]
            gs.robots = {
                rid: {
                    "position": (env.players[blue_agents[i]].pos[0], env.players[blue_agents[i]].pos[1]),
                    "has_flag": env.players[blue_agents[i]].has_flag,
                    "hp": gs.robots.get(rid, {}).get("hp", 100),
                }
                for i, rid in enumerate(blue_rids)
            }

            blue_env_obs = {
                aid: env.state_to_obs(aid, normalize=False) for aid in blue_agents
            }
            local_obs = block2_state_collection(gs, env_obs=blue_env_obs)
            gs = block3_global_state_encoding(gs, local_obs)
            # CTF state for Block 10 (replan checks) and for Block 4 when we prompt
            ctf_state = {
                "score_blue": env.game_score["blue_captures"],
                "score_red": env.game_score["red_captures"],
                "blue_has_red_flag": any(
                    env.players[aid].has_flag for aid in blue_agents
                ),
                "red_has_blue_flag": any(
                    env.players[aid].has_flag for aid in red_agents
                ),
            }
            # Only call Block 4 (and LLM) when we need to show strategies to the human.
            # This avoids running LLM every step (slow/choppy) and ensures fresh strategies when reprompting.
            if human_plan is None:
                llm_summary, strategies = block4_llm_state_summarization(
                    gs, Config(), ctf_state=ctf_state
                )
                human_plan = block5_human_intervention(
                    strategies, Config(), llm_summary=llm_summary
                )
                last_replan_step = step_count
                last_replan_ctf_state = dict(ctf_state)
            hrl = block6_high_level_rl_manager(gs, human_plan)
            block7_subgoal_dispatching(gs, hrl)
            # Show strategy assignment when we just got a new plan (R0-R5 = blue screen IDs 0-5)
            if human_plan is not None and last_replan_step == step_count:
                def_rids = [rid for rid in blue_rids if gs.robots.get(rid, {}).get("team") == "Defend"]
                flank_rids = [rid for rid in blue_rids if gs.robots.get(rid, {}).get("team") == "Flank"]
                atk_rids = [rid for rid in blue_rids if gs.robots.get(rid, {}).get("team") == "Attack"]
                parts = [f"Defend (→blue zone): {def_rids}"]
                if flank_rids:
                    parts.append(f"Flank (→flank right): {flank_rids}")
                parts.append(f"Attack (→red zone): {atk_rids}")
                print("Strategy applied — " + "; ".join(parts))
            # If replan needed, clear plan so next step will re-prompt
            should_replan = block10_check_strategic_changes(
                gs,
                ctf_state=ctf_state,
                last_replan_step=last_replan_step,
                last_replan_ctf_state=last_replan_ctf_state,
            ) or hrl.request_replan
            if should_replan:
                human_plan = None
                last_replan_step = step_count
                last_replan_ctf_state = dict(ctf_state)
            # A* goals: red zone (attack), blue zone (defend / return with flag). Ensure (x,y) tuples.
            _red_home = env.flags[int(Team.RED_TEAM)].home
            _blue_home = env.flags[int(Team.BLUE_TEAM)].home
            red_zone_center = (float(_red_home[0]), float(_red_home[1]))
            blue_zone_center = (float(_blue_home[0]), float(_blue_home[1]))
            # Defenders hold near home base, firmly in the middle band so they never touch bottom/top walls.
            wx, wy = 120.0, 60.0  # match pyquaticus world_size
            blue_hold_x = max(DEFENDER_HOLD_INSET, min(wx - DEFENDER_HOLD_INSET, blue_zone_center[0]))
            blue_hold_y = max(22.0, min(38.0, blue_zone_center[1]))
            defender_hold_world = (blue_hold_x, blue_hold_y)
            # Flank right: waypoint near scrimmage (x=70) on the top flank (y=45), so they advance along the right side
            flank_hold_world = (70.0, max(22.0, min(wy - DEFENDER_HOLD_INSET, 45.0)))
            robot_actions = block8_low_level_execution(
                gs, local_obs, hrl,
                opponent_flag_world=red_zone_center,
                own_flag_world=blue_zone_center,
                defender_hold_world=defender_hold_world,
                flank_hold_world=flank_hold_world,
            )

            for i, agent_id in enumerate(blue_agents):
                rid = f"R{agent_id}"
                block_action = "HOLD"
                for ra in robot_actions:
                    if ra.get("robot_id") == rid:
                        block_action = ra.get("action", "HOLD")
                        break
                player = env.players[agent_id]
                # Defenders should hold near blue zone; wall avoidance was pushing them west (away from wall = toward red). Skip it for Defend team.
                is_defender = gs.robots.get(rid, {}).get("team") == "Defend"
                if not is_defender:
                    block_action = _wall_avoid_direction(
                        blue_env_obs[agent_id], block_action, player.heading
                    )
                actions[agent_id] = block_direction_to_env_action(
                    player.heading, block_action
                )

            obs_dict = {
                aid: env.state_to_obs(aid, normalize=False) for aid in env.agents
            }
            for j, agent_id in enumerate(red_agents):
                act_17 = red_policies[j].compute_action(obs_dict)
                actions[agent_id] = 8 if act_17 == 16 else act_17 % 8

            obs, _, _, _, _ = env.step(actions)
            step_count += 1
            gs.t = step_count
            done, reason = block12_termination_check(gs, cfg, env)
            if done:
                print(f"\n--- {reason} ---")
                break

        env.render()
        clock.tick(30)


if __name__ == "__main__":
    main()
