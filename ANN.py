import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

class ANN(nn.Module):
    """A simple feedforward ANN with 1 hidden layer."""
    def __init__(self, input_dim, hidden_dim):
        super(ANN, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

class ANN_Model:
    """
    Model for predicting the output gap, using an ANN internally.
    It includes StandardScalers for X (features) and y (target),
    so that predict() can handle unscaled data automatically.
    """
    def __init__(self, input_dim, hidden_dim=3, lr=0.01):
        self.model = ANN(input_dim, hidden_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Scalers for features and target
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.train_losses = []
        self.val_losses = []
        self.fitted = False

    def fit(self, X_train, y_train, X_val, y_val, 
            max_epochs=1000, patience=10, batch_size=32):
        """
        Train the model using early stopping and mini-batch processing.
        This method fits scalers internally, so input (X_train, y_train) 
        should be unscaled data.
        """
        # Fit the scalers on the training data
        y_train = y_train.reshape(-1, 1)  # ensure 2D for scaler
        y_val = y_val.reshape(-1, 1)

        self.scaler_X.fit(X_train)
        self.scaler_y.fit(y_train)

        # Transform both training and validation data
        X_train_scaled = self.scaler_X.transform(X_train)
        y_train_scaled = self.scaler_y.transform(y_train)
        X_val_scaled   = self.scaler_X.transform(X_val)
        y_val_scaled   = self.scaler_y.transform(y_val)

        # Convert to PyTorch tensors
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1)
        X_val_t   = torch.tensor(X_val_scaled,   dtype=torch.float32)
        y_val_t   = torch.tensor(y_val_scaled,   dtype=torch.float32).view(-1)

        # Create DataLoader for mini-batch training
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = np.inf
        epochs_no_improve = 0
        best_state_dict = None

        for epoch in range(max_epochs):
            # Training (Mini-Batch)
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                predictions = self.model(X_batch).squeeze()
                loss = self.criterion(predictions, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)

            # Average training loss
            train_loss = epoch_loss / len(train_dataset)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t).squeeze()
                val_loss = self.criterion(val_preds, y_val_t).item()

            # Store losses for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = self.model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load the best model weights
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        self.fitted = True

    def predict(self, input_data):
        """
        Predict method that accepts unscaled NumPy data,
        scales it internally, and returns predictions in the original scale.
        
        :param input_data: A NumPy array of features (unscaled).
        :return: A NumPy array of predictions in original scale (unscaled).
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.values  # or input_data.to_numpy()

        # Scale incoming features
        input_scaled = self.scaler_X.transform(input_data)

        self.model.eval()
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        with torch.no_grad():
            preds_scaled = self.model(input_tensor).squeeze().numpy()

        # Inverse-transform the predictions back to original scale
        preds_scaled_2D = preds_scaled.reshape(-1, 1)
        preds_unscaled = self.scaler_y.inverse_transform(preds_scaled_2D).ravel()
        return preds_unscaled

    def save(self, filepath):
        """
        Save the trained model and scalers to a file.

        :param filepath: Path to the file where the model will be saved.
        """
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        # Create a dictionary to save all necessary components
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        # Save the checkpoint using torch.save
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath, input_dim, hidden_dim=3, lr=0.01):
        """
        Load the model and scalers from a saved file.

        :param filepath: Path to the file from which to load the model.
        :param input_dim: Number of input features (must match the saved model).
        :param hidden_dim: Number of hidden units (must match the saved model).
        :param lr: Learning rate for the optimizer (optional, not used in prediction).
        :return: An instance of ANN_Model with loaded weights and scalers.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at {filepath}")

        # Initialize a new instance
        instance = cls(input_dim, hidden_dim, lr)

        # Load the checkpoint
        checkpoint = torch.load(filepath)

        # Load the model state
        instance.model.load_state_dict(checkpoint['model_state_dict'])

        # Load the scalers
        instance.scaler_X = checkpoint['scaler_X']
        instance.scaler_y = checkpoint['scaler_y']

        # Load training history (optional)
        instance.train_losses = checkpoint.get('train_losses', [])
        instance.val_losses = checkpoint.get('val_losses', [])

        instance.fitted = True
        print(f"Model loaded from {filepath}")
        return instance
