import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3", render_mode="ansi")
num_of_states = env.observation_space.n # 500
num_of_actions = env.action_space.n # 6

# Q table for each state-action pair, initialized with zeros
Q = np.zeros((num_of_states, num_of_actions)) # 500x6

# Q-learning parameters
episodes = 20000 # number of episodes in the training phase
alpha = 0.9 # learning rate
gamma = 0.99 # reward decay rate / discount factor
epsilon = 0.99 # exploration rate (% of time exploring in the beginning)
min_epsilon = 0.1 # minimum exploration rate 
epsilon_decay = 0.99 # epsilon decay rate


# Main loop for Q learning (Training Phase)
for episode in range(episodes):
    state, info = env.reset() # Random state 1-500 to begin with (300 viable options within the 1-500 range)
    # if episode < 1:
    #     print(f"state: {state}, action mask: {info['action_mask']}")
    done = False

    while not done:
        # Choose action using epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploration: choose random action
        else:
            action = np.argmax(Q[state]) # Exploitation: choose best action from Q-table

        # Perform action and observe next state and reward
        next_state, reward, done, truncated, info = env.step(action)
        if truncated: # If step limit (200) is reached, break the loop
            break

        # Update Q-value for state-action pair
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

         # Move to next state
        state = next_state

   # Decay epsilon to reduce exploration over time
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Training Phase Completed.")

# Evaluate the trained policy over 10 evaluation runs with 100 episodes in each (WAS THIS THE INTENDED WAY??)
num_of_eval_runs = 10
episodes_per_eval_run = 100
total_rewards = 0
total_steps = 0

for eval_run in range(num_of_eval_runs):
    print(f"Eval run {eval_run+1} in progress...")
    for episode in range(episodes_per_eval_run): # Each evaluation run has same amount of episodes as the learning phase 
        state, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action = np.argmax(Q[state])
            next_state, reward, done, truncated, info = env.step(action)
            total_rewards += reward
            total_steps += 1
            state = next_state

print("--------EVALUATION RESULTS--------")
print(f"Average reward: {total_rewards/(episodes_per_eval_run * num_of_eval_runs)}")
print(f"Average steps taken: {total_steps/(episodes_per_eval_run * num_of_eval_runs)}")
print("----------------------------------")

