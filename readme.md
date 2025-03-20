# ST449 - Optimal Monetary Policy Using Reinforcement Learning


This repository contains the implementation of models and simulations used to optimize monetary policy using reinforcement learning and traditional approaches. Below is the directory layout and basic usage instructions.

As a rule of thumb .ipynb files contain the same code as the .py files of the same name. The difference is that .ipynb files contain a step by step rundown of how the code works, aswell as some additional performance metrics while .py  were included for importing aswell as performance improvements when training.

---

## Directory Layout

### Main Files
- **`main.ipynb`**: The primary notebook that integrates all components, including data processing, model training, and evaluation.
- **`PaperEnv.ipynb`**: Contains the implementation of the custom Gymnasium environment (`PaperBasedEconomyEnv`) used for training and testing models.
- **`Linear_Model.ipynb`**: Implementation of the Structural Vector Autoregression (SVAR) linear model for predicting output gap and inflation.
- **`ANN.ipynb`**: Artificial Neural Network (ANN) implementation for approximating the relationships between macroeconomic variables.

### Python Scripts
- **`ann_train.py`**: Script to train the agent on a Artificial Neural Network environment quickly.
- **`ols_train.py`**: Script to train the agent on a OLS environment quickly.
- **`PaperEnv.py`**: The custom Gymnasium environment (`PaperBasedEconomyEnv`) implemented in a standalone script.
- **`ANN.py`**: Contains helper functions and utilities for ANN training and evaluation.

### Supporting Directories
- **`Data`**: Folder containing raw and preprocessed datasets used in the analysis.
- **`ANN_Weights`**: Directory storing trained ANN model weights for reuse in simulations.
- **`PPO_weights`**: Directory storing weights for models trained using Proximal Policy Optimization (PPO).
- **`logs`**: Directory containing logs for training and debugging.

### Miscellaneous
- **`Notes`**: Any project-related notes or documentation.

---


