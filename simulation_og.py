import sys
import pygame
from pyquaticus.envs.pyquaticus import PyQuaticusEnv

def main():
    pygame.init()

    team_size = int(input("Enter team size (1–5): "))

    env = PyQuaticusEnv(
        team_size=team_size,
        render_mode="human",
        render_agent_ids=True
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
