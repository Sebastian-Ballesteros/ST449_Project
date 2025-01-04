import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class MacroDataGenerator:
    def __init__(self, start_date='2020-01-01', periods=60):
        """
        Initialize the generator with parameters
        
        Args:
            start_date (str): Start date for the time series
            periods (int): Number of monthly periods to generate
        """
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.periods = periods
        
        # Initial state ranges based on historical data
        self.inflation_range = (1.0, 4.0)
        self.output_gap_range = (-4.0, 2.0)
        
        # Parameters for the data generation process
        self.inflation_persistence = 0.7  # AR(1) coefficient for inflation
        self.output_gap_persistence = 0.8  # AR(1) coefficient for output gap
        self.inflation_volatility = 0.3
        self.output_gap_volatility = 0.5
        
        # Taylor rule parameters
        self.neutral_rate = 2.5
        self.inflation_response = 1.5
        self.output_gap_response = 0.5

    def generate_data(self):
        """
        Generate a dataset of macroeconomic variables
        
        Returns:
            pandas.DataFrame: DataFrame containing date, inflation, output gap, and interest rate
        """
        # Initialize arrays
        dates = [self.start_date + timedelta(days=30*i) for i in range(self.periods)]
        inflation = np.zeros(self.periods)
        output_gap = np.zeros(self.periods)
        interest_rate = np.zeros(self.periods)
        
        # Set initial values
        inflation[0] = np.random.uniform(*self.inflation_range)
        output_gap[0] = np.random.uniform(*self.output_gap_range)
        
        # Generate time series
        for t in range(1, self.periods):
            # Generate inflation with persistence and random shock
            inflation_shock = np.random.normal(0, self.inflation_volatility)
            inflation[t] = (self.inflation_persistence * inflation[t-1] + 
                          (1 - self.inflation_persistence) * np.mean(self.inflation_range) + 
                          inflation_shock)
            
            # Generate output gap with persistence and random shock
            output_shock = np.random.normal(0, self.output_gap_volatility)
            output_gap[t] = (self.output_gap_persistence * output_gap[t-1] + 
                           (1 - self.output_gap_persistence) * np.mean(self.output_gap_range) + 
                           output_shock)
            
            # Ensure values stay within realistic ranges
            inflation[t] = np.clip(inflation[t], *self.inflation_range)
            output_gap[t] = np.clip(output_gap[t], *self.output_gap_range)
        
        # Generate interest rates using a Taylor-type rule
        for t in range(self.periods):
            interest_rate[t] = (self.neutral_rate + 
                              self.inflation_response * (inflation[t] - 2.0) + 
                              self.output_gap_response * output_gap[t])
            interest_rate[t] = max(0.0, interest_rate[t])  # Zero lower bound
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'inflation': np.round(inflation, 2),
            'output_gap': np.round(output_gap, 2),
            'interest_rate': np.round(interest_rate, 2)
        })
        
        return df

# Example usage
if __name__ == "__main__":
    generator = MacroDataGenerator(start_date='1987-01-01', periods=80)
    macro_data = generator.generate_data()
    print(macro_data.head())
    
    # Optional: Save to CSV
    macro_data.to_csv('macro_data.csv', index=False)