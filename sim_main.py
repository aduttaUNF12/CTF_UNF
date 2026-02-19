import sys
import pygame
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from all_blocks import *

def main():
    pygame.init()

    # Prompt user for team size and position
    team_size = int(input("Enter team size (1–6): \n"))
    pos_choice = input(f"Select starting position:\n" + 
                       "(1) Random\n" +
                       "(Any other key) Default - Straight line\n")

    if pos_choice == '1':
        start_pos = "random"
    else:
        start_pos = "default"

    # Red team is enemy - keep random for now and wait for strategy
    # Blue team uses block1 moves

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
            actions = {
                agent: env.action_spaces[agent].sample()
                for agent in env.agents
            }
            env.step(actions)

        env.render()   # custom render handles text + flip
        clock.tick(30)

if __name__ == "__main__":
    main()
