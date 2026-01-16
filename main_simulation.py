import sys
import pygame
from pyquaticus.envs.pyquaticus import PyQuaticusEnv

# Draw score overlay
def draw_overlay(env):
    font = pygame.font.SysFont(None, 28)
    score = env.game_score

    text = (
        f"Blue Captures: {score['blue_captures']} | "
        f"Red Captures: {score['red_captures']}"
    )

    label = font.render(text, True, (0, 0, 0))
    env.screen.blit(label, (10, 10))

# Main simulation loop
def main():
    # Get user input
    try:
        team_size = int(input("Enter team size (1–5): "))
        num_steps = int(input("Enter number of steps: "))
    except ValueError:
        print("Invalid input.")
        return

    # Create environment
    env = PyQuaticusEnv(
        team_size=team_size,
        render_mode="human",
        render_agent_ids=True
    )

    obs = env.reset()

    paused = False
    clock = pygame.time.Clock()

    print("Controls:")
    print("SPACE = Pause/Resume")
    print("ESC   = Quit")

    # Simulation loop
    for step in range(num_steps):

        # Event handling
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

        # Pause mode
        if paused:
            env.render()
            draw_overlay(env)
            pygame.display.flip()
            clock.tick(30)
            continue

        # Build actions
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}

        # Step environment
        obs, rewards, terms, truncs, infos = env.step(actions)

        # Render
        env.render()
        draw_overlay(env)
        pygame.display.flip()

        if all(terms.values()):
            print("Episode finished.")
            break

        clock.tick(30)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
