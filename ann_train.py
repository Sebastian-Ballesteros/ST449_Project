import pandas as pd
import os



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

from PaperEnv import PaperBasedEconomyEnv

ann_env = PaperBasedEconomyEnv(
    historical_df=Env_df,
    model_y= ann_model_y,
    model_pi= ann_model_pi,
    lookback_periods=2,
    inflation_target=2,
    output_gap_target=0.0,
    max_steps=50  # limit episode to n steps
)



from stable_baselines3.common.monitor import Monitor
log_dir = "./logs/"
ann_env = Monitor(ann_env, filename=log_dir+'ann')



from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ann_venv = DummyVecEnv([lambda: ann_env])  # wrap your environment

ann_model = PPO(
    policy="MlpPolicy",
    env=ann_venv,
    verbose=1,
    device = "cpu"
    # You can tune many hyperparameters here
)
ann_model.learn(total_timesteps=500_000)


ann_model.save("PPO_weights/ann_ppo")