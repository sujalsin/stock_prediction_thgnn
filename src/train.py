from typing import Dict, Any, List, Tuple
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from thgnn_model import THGNNPredictor
from data_preprocessing import StockDataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from yaml file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

class TemporalBatch:
    def __init__(self, graphs: List[List[Data]], targets: torch.Tensor):
        self.graphs = graphs  # Keep as list of lists
        self.targets = targets
    
    def to(self, device):
        """Move batch to device"""
        processed_graphs = []
        for seq in self.graphs:
            processed_seq = []
            for g in seq:
                if isinstance(g, Data):
                    g.x = g.x.to(device)
                    g.edge_index = g.edge_index.to(device)
                    processed_seq.append(g)
                else:
                    # If g is a tuple (x, edge_index)
                    x, edge_index = g
                    x = x.to(device)
                    edge_index = edge_index.to(device)
                    processed_seq.append(Data(x=x, edge_index=edge_index))
            processed_graphs.append(processed_seq)
        
        self.graphs = processed_graphs
        self.targets = self.targets.to(device)
        return self

def collate_temporal(batch: List[Tuple[List[Data], torch.Tensor]]) -> TemporalBatch:
    """
    Collate function for temporal graph data
    Args:
        batch: List of tuples (graph_sequence, target)
    Returns:
        TemporalBatch object
    """
    # Unzip the batch into sequences and targets
    sequences, targets = zip(*batch)
    
    # Stack targets into a tensor
    targets = torch.stack(list(targets)).squeeze()  # [batch_size]
    
    # Create TemporalBatch
    return TemporalBatch(graphs=list(sequences), targets=targets)

def collate_temporal(batch):
    """Custom collate function for temporal graph data."""
    sequences = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return sequences, targets

class Trainer:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = THGNNPredictor(
            input_dim=self.config['model']['input_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model'].get('dropout', 0.2)
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['training']['patience'],
            min_delta=1e-5
        )
        
        # Create directories
        os.makedirs(self.config['paths']['model_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['log_dir'], exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        # Use tqdm for progress tracking
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # Move batch to device
            batch = batch.to(self.device)
            
            # Forward pass
            out = self.model(batch)
            loss = F.mse_loss(out, batch.targets.unsqueeze(-1))
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['clip_grad_norm'])
            self.optimizer.step()
            
            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                logging.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            # Update total loss
            total_loss += loss.item()
            
            # Clear memory
            del batch, out, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return total_loss / len(train_loader)

    def validate_epoch(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = F.mse_loss(out, batch.targets.unsqueeze(-1))
                total_loss += loss.item()
                
                # Clear memory
                del batch, out, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return total_loss / len(val_loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        logger.info("Starting training...")
        best_val_loss = float('inf')
        
        for epoch in range(self.config['training']['num_epochs']):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            logging.info(f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]}')
            logging.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Model checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.config['paths']['model_dir'], 'best_model.pth')
                )
                logger.info(f'Model saved with validation loss: {val_loss:.4f}')
            
            # Early stopping
            if self.early_stopping.step(val_loss):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Clear memory after each epoch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def test(self, test_loader: DataLoader):
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.model(batch)
                loss = F.mse_loss(predictions.squeeze(), batch.targets)
                
                # Store predictions and targets
                all_predictions.append(predictions.squeeze())
                all_targets.append(batch.targets)
                
                total_loss += loss.item()
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        # Calculate metrics
        metrics = calculate_metrics(all_predictions, all_targets)
        
        # Log results
        logger.info("Test Results:")
        logger.info(f"Loss: {total_loss / len(test_loader):.6f}")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.6f}")
        
        # Plot predictions
        plot_predictions(all_predictions.cpu().numpy(), all_targets.cpu().numpy(), save_path=os.path.join(self.config['paths']['plot_dir'], 'predictions.png'))
        
        return all_predictions, all_targets, metrics

def calculate_metrics(predictions, targets):
    """Calculate various evaluation metrics."""
    predictions = predictions.detach().numpy()
    targets = targets.detach().numpy()
    
    # MSE and RMSE
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    
    # MAE
    mae = mean_absolute_error(targets, predictions)
    
    # Directional Accuracy
    directional_acc = np.mean((predictions[1:] >= predictions[:-1]) == (targets[1:] >= targets[:-1]))
    
    # R-squared
    r2 = r2_score(targets, predictions)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'directional_accuracy': directional_acc,
        'r2': r2
    }

def plot_predictions(predictions, targets, save_path='plots/prediction_vs_actual.png'):
    """Plot predictions vs actual values."""
    plt.figure(figsize=(12, 6))
    plt.plot(targets, label='Actual', color='blue', alpha=0.7)
    plt.plot(predictions, label='Predicted', color='red', alpha=0.7)
    plt.title('Stock Price Predictions vs Actual Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Price Change')
    plt.legend()
    plt.grid(True)
    
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Model parameters
    input_dim = 13  # Updated for new features
    hidden_dim = 128
    num_layers = 2
    dropout = 0.1
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 32
    window_size = 10
    
    # Initialize data processor
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']  # Example symbols
    start_date = '2018-01-01'
    end_date = '2023-12-31'
    
    logger.info("Initializing data processor...")
    data_processor = StockDataProcessor(symbols=symbols, start_date=start_date, end_date=end_date)
    
    # Download and process data
    logger.info("Processing data...")
    data_processor.download_data()
    train_data, val_data, test_data = data_processor.prepare_dataset(window_size=window_size)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_temporal)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_temporal)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_temporal)
    
    # Initialize model
    logger.info("Initializing model...")
    model = THGNNPredictor(input_dim=input_dim, 
                          hidden_dim=hidden_dim,
                          num_layers=num_layers,
                          dropout=dropout)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            
            # Move batch to device
            sequences = [[g.to(device) for g in seq] for seq in batch[0]]
            targets = batch[1].to(device)
            
            # Forward pass
            predictions = model(sequences)
            loss = criterion(predictions.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_predictions.extend(predictions.squeeze().detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Calculate training metrics
        train_metrics = calculate_metrics(train_targets, train_predictions)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = [[g.to(device) for g in seq] for seq in batch[0]]
                targets = batch[1].to(device)
                
                predictions = model(sequences)
                loss = criterion(predictions.squeeze(), targets)
                
                val_loss += loss.item()
                val_predictions.extend(predictions.squeeze().cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate validation metrics
        val_metrics = calculate_metrics(val_targets, val_predictions)
        
        # Log metrics
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f}')
        logger.info(f'Train Metrics: {train_metrics}')
        logger.info(f'Val Loss: {val_loss:.4f}')
        logger.info(f'Val Metrics: {val_metrics}')
        
        # Early stopping check
        if early_stopping(val_loss):
            logger.info("Early stopping triggered")
            break
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Load best model for testing
    logger.info("Loading best model for testing...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Testing phase
    test_loss = 0.0
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences = [[g.to(device) for g in seq] for seq in batch[0]]
            targets = batch[1].to(device)
            
            predictions = model(sequences)
            loss = criterion(predictions.squeeze(), targets)
            
            test_loss += loss.item()
            test_predictions.extend(predictions.squeeze().cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    
    test_loss /= len(test_loader)
    
    # Calculate test metrics
    test_metrics = calculate_metrics(test_targets, test_predictions)
    
    logger.info("\nTest Results:")
    logger.info(f'Test Loss: {test_loss:.4f}')
    logger.info(f'Test Metrics: {test_metrics}')
    
    # Plot results
    plot_predictions(test_targets, test_predictions, 'test_predictions.png')
    plot_loss_curves(train_losses, val_losses, 'loss_curves.png')
    
    return model, test_metrics

def calculate_metrics(y_true, y_pred):
    """Calculate various performance metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate directional accuracy
    direction_true = np.sign(np.array(y_true))
    direction_pred = np.sign(np.array(y_pred))
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'DirectionalAccuracy': directional_accuracy
    }

def plot_predictions(y_true, y_pred, filename):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Actual vs Predicted Stock Returns')
    plt.savefig(filename)
    plt.close()

def plot_loss_curves(train_losses, val_losses, filename):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig(filename)
    plt.close()

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

if __name__ == "__main__":
    main()
