import sys
import pygame
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from all_blocks import *
import random

def main():
    pygame.init()

    # Prompt user for team size and position
    team_size = int(input("Enter team size (1–6): \n"))
    pos_choice = input(f"\nSelect starting position:\n" + 
                       "(1) Random\n" +
                       "(Any other key) Default - Straight line\n")
    start_pos = "random" if pos_choice == "1" else "default"
    
    mode_choice = input(
        "\nSelect mode:\n"
        "(1) hard\n"
        "(2) medium\n"
        "(any other key) easy\n"
    )
    mode = "hard" if mode_choice == "1" else "medium" if mode_choice == "2" else "easy"


    env = PyQuaticusEnv(
        team_size=team_size,
        render_mode="human",
        render_agent_ids=True,
        start_pos=start_pos
    )

    env.reset()

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

            # Split teams
            blue_agents = env.agents[:team_size]
            red_agents = env.agents[team_size:]

            # BLUE TEAM movement logic
            if not hasattr(env, "global_state"):
                # if not created yet (safety)
                cfg = Config(num_robots=len(blue_agents))
                env.global_state = block1_environment_initialization(cfg)

            gs = env.global_state

            # Run Blocks 2–8
            local_obs = block2_state_collection(gs)
            gs = block3_global_state_encoding(gs, local_obs)
            _, strategies = block4_llm_state_summarization(gs, Config())
            human_plan = block5_human_intervention(strategies, Config())
            hrl = block6_high_level_rl_manager(gs, human_plan)
            block7_subgoal_dispatching(gs, hrl)
            
            # robot_actions = block8_low_level_execution(gs, local_obs, hrl) -- removed for now
            robot_actions = [{"action": random.choice(list(DISCRETE_ACTIONS.keys()))} for _ in blue_agents]
            # Convert block actions → PyQuaticus discrete actions
            for i, agent in enumerate(blue_agents):
                block_action = robot_actions[i]["action"]
                actions[agent] = DISCRETE_ACTIONS.get(block_action, 8) 

            # Update block environment transition
            # block9_environment_transition(gs, robot_actions) -- removed for now

            # RED TEAM random movement logic
            for agent in red_agents:
                actions[agent] = env.action_spaces[agent].sample()

            env.step(actions)

        env.render()   # custom render handles text + flip
        clock.tick(30)

if __name__ == "__main__":
    main()
