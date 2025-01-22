import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader



data_folder = 'Data'
filename = 'Clean_data.csv'
file_path = os.path.join(data_folder, filename)

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file '{file_path}' does not exist.")

# Read the CSV without parsing dates
merged_df = pd.read_csv(file_path)

Env_df = merged_df[['Inflation_Rate (%)','Avg_Interest_Rate','Output_gap (%)']]

from ANN import ANN_Model

ann_model_y = ANN_Model.load('ANN_Weights/y.pth',input_dim= 4, hidden_dim=3)
ann_model_pi = ANN_Model.load('ANN_Weights/pi.pth',input_dim= 7, hidden_dim=4)


# Define Variables
merged_df['Lag_y1'] = merged_df['Output_gap (%)'].shift(1)
merged_df['Lag_y2'] = merged_df['Output_gap (%)'].shift(2)
merged_df['Lag_pi1'] = merged_df['GDP Deflator'].shift(1)
merged_df['Lag_pi2'] = merged_df['GDP Deflator'].shift(2)
merged_df['Lag_i1'] = merged_df['Avg_Interest_Rate'].shift(1)
merged_df['Lag_i2'] = merged_df['Avg_Interest_Rate'].shift(2)

# Drop rows with NaN values created due to lagging
merged_df = merged_df.dropna()

print(merged_df)

from sklearn.linear_model import LinearRegression

# Linear Model (SVAR) Implementation

X_y = merged_df[['Lag_y1', 'Lag_pi1', 'Lag_i1', 'Lag_i2']]
y_y = merged_df['Output_gap (%)']

X_pi = merged_df[['Output_gap (%)', 'Lag_y1', 'Lag_y2', 'Lag_pi1', 'Lag_pi2', 'Lag_i1', 'Lag_i2']]
y_pi = merged_df['Inflation_Rate (%)']

ols_model_y = LinearRegression()
ols_model_pi = LinearRegression()

    # Fit Models
ols_model_y.fit(X_y, y_y)
ols_model_pi.fit(X_pi, y_pi)


from PaperEnv import PaperBasedEconomyEnv

ols_env = PaperBasedEconomyEnv(
    historical_df=Env_df,
    model_y= ols_model_y,
    model_pi= ols_model_pi,
    lookback_periods=2,
    inflation_target=2,
    output_gap_target=0.0,
    max_steps=50  # Force each episode to be 50 steps
    
)


from stable_baselines3.common.monitor import Monitor
log_dir = "./logs/"
ols_env = Monitor(ols_env, filename=log_dir+'ols')


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ols_venv = DummyVecEnv([lambda: ols_env])  # wrap your environment

ols_model = PPO(
    policy="MlpPolicy",
    env=ols_venv,
    verbose=1,
    device = "cpu"
    # You can tune many hyperparameters here
)
ols_model.learn(total_timesteps=500_000)


ols_model.save("PPO_weights/ols_ppo")