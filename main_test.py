from pyquaticus.envs.pyquaticus import PyQuaticusEnv
import pygame

teamSize = int(input("Enter team size: "))
numSteps = int(input("Enter number of steps to run: "))

# Create environment
env = PyQuaticusEnv(team_size=teamSize, render_mode="human")

# Reset environment
obs = env.reset()

for step in range(numSteps):
    # Use the correct action space dictionary
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}

    obs, rewards, terms, truncs, _ = env.step(actions)
    env.render()

    # Episode ends when all agents are done
    if all(terms.values()):
        print("Episode finished.")
        break

# keep window open
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

env.close()

