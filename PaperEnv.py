import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box

class PaperBasedEconomyEnv(gym.Env):
    def __init__(
        self,
        model_y,            # e.g. a LinearRegression() for Output Gap
        model_pi,           # e.g. a LinearRegression() for Inflation
        historical_df,      # historical data for reset()
        lookback_periods=2,
        inflation_target=2.0,
        output_gap_target=0.0,
        max_steps=50,
        

    ):
        super(PaperBasedEconomyEnv, self).__init__()

        # Store initial DataFrame and reset index
        self.model_y = model_y       # The trained model for predicting Output Gap
        self.model_pi = model_pi     # The trained model for predicting Inflation
        self.lookback_periods = lookback_periods
        self.inflation_target = inflation_target
        self.output_gap_target = output_gap_target
        self.historical_df = historical_df
        self.df = historical_df.iloc[0:3].copy()
        self.max_steps = max_steps  # number of timesteps per episode

        # Column references for readability
        self.cols = {
            'inflation': 'Inflation_Rate (%)',
            'output_gap': 'Output_gap (%)',
            'interest_rate': 'Avg_Interest_Rate'
        }

        # Action space: choose the next interest rate
        self.action_space = Box(low=-20.0, high= 20.0, shape=(1,), dtype=np.float32)

        # Observation space: for each of the last N periods, we store
        # [inflation, output_gap, interest_rate], plus 1 extra for interest_rate(t-1)
        obs_space_size = 3 * self.lookback_periods# + 1
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_space_size,), dtype=np.float32
        )

        self.current_idx = self.lookback_periods
        self.done = False

    def reset(self, seed=None, options=None):
        """
        Reset the environment:
          1. Choose a random start index that allows for the lookback period.
          2. Initialize self.df with 3 consecutive rows starting from the random index.
          3. Reset step counters and done flag.
          4. Return the initial observation.
        """
        super().reset(seed=seed)

        # Calculate the maximum possible starting index
        required_rows = self.lookback_periods + 1  # 2 lookbacks + 1 current step = 3 rows
        max_start = len(self.historical_df) - required_rows

        if max_start <= 0:
            raise ValueError(
                "DataFrame is too small for the given lookback_periods. "
                "Reduce lookback_periods or provide more data."
            )

        # Randomly select a starting index
        random_start_index = self.np_random.integers(low=0, high=max_start + 1)

        # Initialize self.df with 3 consecutive rows from the original DataFrame
        self.df = self.historical_df.iloc[random_start_index:random_start_index + required_rows].reset_index(drop=True).copy()

        # Set current_idx to point to the last row in the initial df
        self.current_idx = self.lookback_periods  # Zero-based indexing

        # Reset step counters and done flag
        self.episode_step = 0
        self.done = False

        # Get the initial observation
        obs = self._get_state()
        info = {}
        return obs, info


    def step(self, action):
        """
        1. Parse the current state
        2. Construct the feature vectors for each model
        3. Predict next_output_gap and next_inflation
        4. Compute reward
        5. Append new row to DataFrame
        6. Advance time index
        7. Return (state, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset().")
        
        self.episode_step += 1

        # Action is the chosen interest rate for this step
        interest_rate = float(action[0])
        # Current state (shape: (3*lookback_periods + 1,))
        state = self._get_state()

        # ---------------------------------------------------------
        # 1) Parse the state for clarity
        #    Example for lookback=2:
        #    state = [
        #        inflation(t-2), interest_rate(t-2), output_gap(t-2)
        #        inflation(t-1), interest_rate(t-1), output_gap(t-1)
        #    ]
        # ---------------------------------------------------------
        # Let's label them (assuming lookback_periods=2)
        inflation_t2        = state[0]      # inflation(t-2)
        interest_rate_t2    = state[1]  # interest_rate(t-2)
        output_gap_t2       = state[2]     # output_gap(t-2)
        inflation_t1        = state[3]      # inflation(t-1)
        interest_rate_t1    = state[4]  # interest_rate(t-1)
        output_gap_t1       = state[5]     # output_gap(t-1)
        # state[6] is interest_rate(t-1) repeated in your original code
        # (You might want to adjust that logic—see explanation below.)

        # ---------------------------------------------------------
        # 2) Construct model_y features
        #    Based on how model_y was trained:
        #        X_y columns = ['Lag_y1', 'Lag_pi1', 'Lag_i1', 'Lag_i2']
        #
        #    This implies:
        #       Lag_y1 = output_gap(t-1)
        #       Lag_pi1 = inflation(t-1)
        #       Lag_i1 = interest_rate(t-1)
        #       Lag_i2 = interest_rate(t-2)
        # ---------------------------------------------------------
        features_y = pd.DataFrame({
            'Lag_y1': [output_gap_t1],
            'Lag_pi1': [inflation_t1],
            'Lag_i1': [interest_rate_t1],
            'Lag_i2': [interest_rate_t2]
        })

        # Predict the next output gap
        next_output_gap = self.model_y.predict(features_y)[0]

        # ---------------------------------------------------------
        # 3) Construct model_pi features
        #    Based on how model_pi was trained:
        #        X_pi columns = [
        #           'Output_gap (%)',
        #           'Lag_y1', 'Lag_y2',
        #           'Lag_pi1', 'Lag_pi2',
        #           'Lag_i1'
        #        ]
        #
        #    This implies:
        #       'Output_gap (%)' = *current* output gap used for next inflation
        #       Lag_y1 = output_gap(t-1)
        #       Lag_y2 = output_gap(t-2)
        #       Lag_pi1 = inflation(t-1)
        #       Lag_pi2 = inflation(t-2)
        #       Lag_i1 = interest_rate(t-1)
        #
        #    Here, we have a choice to use the brand-new next_output_gap or
        #    the last known output_gap(t-1). In some setups, we feed the
        #    newly predicted output_gap back in. That is an economic modeling
        #    choice—just be consistent with your original training approach.
        # ---------------------------------------------------------
        features_pi = pd.DataFrame({
            'Output_gap (%)': [next_output_gap],  # or output_gap_t1 if you prefer
            'Lag_y1': [output_gap_t1],
            'Lag_y2': [output_gap_t2],
            'Lag_pi1': [inflation_t1],
            'Lag_pi2': [inflation_t2],
            'Lag_i1': [interest_rate_t1],
            'Lag_i2': [interest_rate_t2]
        })

        # Predict the next inflation
        next_inflation = self.model_pi.predict(features_pi)[0]

        # ---------------------------------------------------------
        # 4) Compute reward
        #    Negative MSE-like penalty from inflation & output gap deviation
        # ---------------------------------------------------------
        reward = -(
            0.5 * (next_inflation - self.inflation_target) ** 2
            + 0.5 * (next_output_gap - self.output_gap_target) ** 2
        )

        # ---------------------------------------------------------
        # 5) Append new row to the DataFrame
        # ---------------------------------------------------------
        new_row = {
            self.cols['inflation']: next_inflation,
            self.cols['output_gap']: next_output_gap,
            self.cols['interest_rate']: interest_rate
        }
        # NOTE: .append() is deprecated in recent Pandas; here’s one alternative:

        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

        # ---------------------------------------------------------
        # 6) Advance time index and check termination
        # ---------------------------------------------------------
        self.current_idx += 1

        if self.episode_step >= self.max_steps:
            self.done = True

        # 7) Return the new state, reward, done, truncated, info
        
        truncated = False

        return self._get_state(), reward, self.done, truncated, {}

    def _get_state(self):
        """
        Builds an array of shape (3*lookback_periods + 1,).
        For lookback=2, we collect:
          [inflation(t-2), rate(t-2), gap(t-2), inflation(t-1), rate(t-1), gap(t-1)]
        """
        # Retrieve the current row and the previous row
        row_t_minus_1 = self.df.iloc[self.current_idx]
        row_t_minus_2 = self.df.iloc[self.current_idx - 1]

        # Convert both rows to lists
        current_row_list = row_t_minus_1.tolist()
        previous_row_list = row_t_minus_2.tolist()

        # Combine the two lists
        state = previous_row_list + current_row_list

        return np.array(state, dtype=np.float32)

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":

    Mock_df =pd.DataFrame({'Inflation_Rate (%)':[1,2,3,4],
             'Avg_Interest_Rate' : [5,6,7,8],
             'Output_gap (%)' : [9,10,11,12]
    })

    # Suppose model_y and model_pi are scikit-learn models, e.g.:
    # model_y = LinearRegression().fit(X_y, y_y)
    # model_pi = LinearRegression().fit(X_pi, y_pi)
    #
    # Or if you're returning functions (predict_y, predict_pi) from linear_model(),
    # you can wrap them in small classes that define a .predict() method.
    # For demonstration, let's assume they are already fitted regressions.
    class MockModel:
        def predict(self, X):
            # Some dummy logic; in reality, this would be your real model
            return np.ones(shape=(X.shape[0],)) * 0.3


    mock_model_y = MockModel()
    mock_model_pi = MockModel()

    env = PaperBasedEconomyEnv( historical_df = Mock_df,
                                model_y = mock_model_y,
                                model_pi = mock_model_pi, 
                                lookback_periods = 2)
    state = env.reset()

    for step_i in range(10):
        action = env.action_space.sample()  # Random interest rate
        state, reward, done, truncated, info = env.step(action)
        print(f"Step={step_i}, State={state}, Reward={reward}, Truncated={truncated}, Done={done}")
        if done:
            break
