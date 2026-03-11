"""
Capture the Flag simulation using original pyquaticus rules.
- Blue team: all_blocks pipeline (blocks 1–8) with A* low-level execution.
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
)

# Wall avoidance: steer away when distance to any wall is below this (meters)
WALL_SAFETY_DIST = 18.0

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
            for i, agent_id in enumerate(blue_agents):
                rid = f"R{i + 1}"
                if rid in gs.robots:
                    player = env.players[agent_id]
                    gs.robots[rid]["position"] = (player.pos[0], player.pos[1])
                    gs.robots[rid]["has_flag"] = player.has_flag

            local_obs = block2_state_collection(gs)
            gs = block3_global_state_encoding(gs, local_obs)
            _, strategies = block4_llm_state_summarization(gs, Config())
            human_plan = block5_human_intervention(strategies, Config())
            hrl = block6_high_level_rl_manager(gs, human_plan)
            block7_subgoal_dispatching(gs, hrl)
            # A* goals: red zone center (attack) and blue zone center (return with flag)
            red_zone_center = env.flags[int(Team.RED_TEAM)].home
            blue_zone_center = env.flags[int(Team.BLUE_TEAM)].home
            robot_actions = block8_low_level_execution(
                gs, local_obs, hrl,
                opponent_flag_world=red_zone_center,
                own_flag_world=blue_zone_center,
            )

            # Unnormalized obs for blue (needed for wall distances)
            blue_obs = {
                aid: env.state_to_obs(aid, normalize=False) for aid in blue_agents
            }
            for i, agent_id in enumerate(blue_agents):
                rid = f"R{i + 1}"
                block_action = "HOLD"
                for ra in robot_actions:
                    if ra.get("robot_id") == rid:
                        block_action = ra.get("action", "HOLD")
                        break
                player = env.players[agent_id]
                # Physics-aware boundary avoidance: steer away from close walls
                block_action = _wall_avoid_direction(
                    blue_obs[agent_id], block_action, player.heading
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

        env.render()
        clock.tick(30)


if __name__ == "__main__":
    main()
