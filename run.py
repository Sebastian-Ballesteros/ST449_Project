

# ---
# Code to replicate a simplified RL approach for a Central Bank environment
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

# Stable-Baselines3 for RL algorithms
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ------------------------------------
# 1) LOAD YOUR DATA
# ------------------------------------
# Done before
# ------------------------------------
# 2) DEFINE THE CENTRAL BANK ENVIRONMENT
# ------------------------------------
class CentralBankEnv(gym.Env):
    """
    A simplified environment where:
      - State: (output_gap, inflation_gap, interest_rate)
      - Action: a change in interest rate (continuous)
      - Reward: negative of (sum of squared gaps for inflation and output)
      
    We'll 'walk' through historical data in order. 
    The environment does not truly simulate the effect of 
    the chosen interest rate on the next state but rather 
    replays the real-world data. 
    A more advanced approach would incorporate a small macro model 
    to determine next state from the chosen action.
    """
    def __init__(self, df, inflation_target=2.0):
        super(CentralBankEnv, self).__init__()
        
        # Store historical data
        self.df = df.reset_index(drop=True)
        
        # We'll define the inflation by quarter-over-quarter changes 
        # in GDP Deflator (approx) or yoy. For simplicity, let's approximate 
        # quarterly inflation as the percent change from previous quarter's deflator:
        self.df['Inflation'] = self.df['GDP Deflator'].pct_change() * 100
        
        # Fill the first inflation with 0 (or drop the first row)
        self.df['Inflation'].fillna(0, inplace=True)
        
        # We'll define 'inflation_gap' = inflation - inflation_target
        self.df['Inflation_Gap'] = self.df['Inflation'] - inflation_target
        
        # Episode tracking
        self.current_step = 0
        self.max_step = len(self.df) - 1
        
        # Store the target
        self.inflation_target = inflation_target
        
        # Define the action space: let's assume we can change interest rate by +/- 1 percentage point
        # from the previous quarter. If you want to allow bigger moves, change the range accordingly.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Define the observation space: 
        # We'll feed the agent (Output_gap, Inflation_Gap, Current_Interest_Rate)
        # Let's set a broad range for each dimension.
        obs_high = np.array([ 10.0,  20.0,  30.0], dtype=np.float32)  # Arbitrary
        obs_low  = np.array([-10.0, -20.0,   0.0], dtype=np.float32)  # Arbitrary
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        """
        Reset environment at the beginning of an episode.
        """
        super().reset(seed=seed)
        
        # Start from a random quarter or from the beginning:
        # Let's just start from the beginning for now.
        self.current_step = 0
        
        # Return the initial observation
        return self._get_observation(), {}
    
    def step(self, action):
        """
        Take one step in the environment:
          1. Interpret action as a change in interest rate from the current interest rate.
          2. Move current_step to next time index.
          3. Compute reward based on how far inflation and output gap are from the desired levels.
        """
        # Current interest rate from data:
        current_interest_rate = self.df.loc[self.current_step, 'Avg_Interest_Rate']
        
        # Apply the action (change)
        delta_interest_rate = float(action[0])
        chosen_interest_rate = current_interest_rate + delta_interest_rate
        
        # Move to next step
        self.current_step += 1
        
        # If we are at or beyond the end of the data, consider the episode done.
        if self.current_step >= self.max_step:
            done = True
            next_obs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            reward = 0.0
        else:
            done = False
            
            # The real next interest rate in the dataset is historically recorded as:
            real_next_interest_rate = self.df.loc[self.current_step, 'Avg_Interest_Rate']
            
            # Next output gap
            next_output_gap = self.df.loc[self.current_step, 'Output_gap (%)']
            
            # Next inflation gap
            next_inflation_gap = self.df.loc[self.current_step, 'Inflation_Gap']
            
            # If you wanted to simulate the effect of chosen_interest_rate on the next state,
            # you would have a small macro model. For this example, we just use the historical data directly.
            # next_output_gap, next_inflation_gap = macro_model(chosen_interest_rate, ...)
            
            # Construct the next observation
            next_obs = np.array([next_output_gap, next_inflation_gap, real_next_interest_rate], 
                                dtype=np.float32)
            
            # Reward: we want to minimize the absolute gap in output and inflation
            # For example, we can use negative sum of squares of the two gaps
            # i.e. Reward = -[(output_gap)^2 + (inflation_gap)^2]
            reward = - (next_output_gap**2 + next_inflation_gap**2)
        
        return next_obs, reward, done, False, {}
    
    def _get_observation(self):
        """
        Constructs the initial observation for the current_step.
        """
        output_gap = self.df.loc[self.current_step, 'Output_gap (%)']
        inflation_gap = self.df.loc[self.current_step, 'Inflation_Gap']
        interest_rate = self.df.loc[self.current_step, 'Avg_Interest_Rate']
        
        return np.array([output_gap, inflation_gap, interest_rate], dtype=np.float32)
    
    def render(self):
        """
        Optionally implement any visualization here.
        """
        pass

# ------------------------------------
# 3) CREATE AN INSTANCE OF THE ENV AND WRAP IT
# ------------------------------------
env = CentralBankEnv(merged_df, inflation_target=2.0)

# Use a DummyVecEnv to handle vectorized environments for Stable-Baselines
# (If you want a single environment, you can just do DummyVecEnv([lambda: env]))
vec_env = DummyVecEnv([lambda: env])

# ------------------------------------
# 4) TRAIN AN AGENT (PPO EXAMPLE)
# ------------------------------------
model = PPO("MlpPolicy", vec_env, verbose=1)

# For demonstration, we will do just a few timesteps. Increase for a real run:
model.learn(total_timesteps=1000)

# ------------------------------------
# 5) TEST / EVALUATION
# ------------------------------------
# We can test the learned policy by running a few episodes and collecting data.

num_test_episodes = 1  # you can do more
for ep in range(num_test_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
    
    print(f"Episode {ep+1} finished after {step_count} timesteps, total reward = {total_reward:.2f}")

# ------------------------------------
# 6) (OPTIONAL) VISUALIZE RESULTS
# ------------------------------------
# Can record interest rate choices, output gap, inflation gap, etc. 
# inside the environment for plotting after training. 
# This snippet simply stores data:

# Re-run an episode to store data
obs, _ = env.reset()
done = False
history = []
while not done:
    action, _ = model.predict(obs, deterministic=True)
    next_obs, reward, done, truncated, info = env.step(action)
    
    # Store
    history.append({
        "step": env.current_step,
        "output_gap": obs[0],
        "inflation_gap": obs[1],
        "interest_rate": obs[2],
        "action": action[0],
        "reward": reward
    })
    
    obs = next_obs

# Convert to DataFrame for analysis
history_df = pd.DataFrame(history)
print(history_df)

# Plot reward over time
plt.figure(figsize=(10, 5))
plt.plot(history_df['step'], history_df['reward'], label='Reward')
plt.title("Reward Over Time for One Episode")
plt.xlabel("Time Step")
plt.ylabel("Reward")
plt.legend()
plt.show()

# Plot interest rate and actions
plt.figure(figsize=(10, 5))
plt.plot(history_df['step'], history_df['interest_rate'], label='Interest Rate')
plt.plot(history_df['step'], history_df['action'], label='Action (ΔInterest)')
plt.title("Interest Rate and Action Over Time")
plt.xlabel("Time Step")
plt.ylabel("Rate (%) / Action")
plt.legend()
plt.show()