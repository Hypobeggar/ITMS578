from qlearning_agent import QLearningAgent
from environment import NetworkEnvironment

env = NetworkEnvironment("../data/network_data.csv")
agent = QLearningAgent(n_states=env.n_states, n_actions=2)

rounds = 5
total_rewards = []

for episode in range(rounds):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        episode_reward += reward

    total_rewards.append(episode_reward)
    print(f"Round {episode+1}: Positive Score = {episode_reward}")

print("Training complete.")
