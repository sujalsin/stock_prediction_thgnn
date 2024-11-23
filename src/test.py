import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch
import numpy as np
from data_preprocessing import StockDataset
from thgnn_model import THGNNPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collate_fn(batch):
    """Custom collate function for temporal PyTorch Geometric data objects."""
    # Each item in batch is (sequence_of_graphs, target)
    sequences = [item[0] for item in batch]  # List of graph sequences
    targets = torch.stack([item[1] for item in batch])
    
    # Process each time step separately
    batch_size = len(sequences)
    seq_length = len(sequences[0])  # Assuming all sequences have same length
    
    # Create a list of batched graphs for each time step
    batched_sequence = []
    for t in range(seq_length):
        # Collect graphs from the same time step across all sequences
        graphs_t = [seq[t] for seq in sequences]
        # Batch the graphs from this time step
        batched_graphs_t = Batch.from_data_list(graphs_t)
        batched_sequence.append(batched_graphs_t)
    
    return batched_sequence, targets

class Trainer:
    def __init__(self, model, learning_rate, device, early_stopping_patience):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs('checkpoints', exist_ok=True)
        
    def train_epoch(self, train_loader):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (batched_sequence, targets) in enumerate(train_loader):
            # Move data to device
            for i in range(len(batched_sequence)):
                batched_sequence[i] = batched_sequence[i].to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batched_sequence)
            loss = self.criterion(predictions.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    def evaluate(self, data_loader):
        """Evaluate the model on validation or test set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batched_sequence, targets in data_loader:
                # Move data to device
                for i in range(len(batched_sequence)):
                    batched_sequence[i] = batched_sequence[i].to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(batched_sequence)
                loss = self.criterion(predictions.squeeze(), targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, num_epochs):
        """Train the model for specified number of epochs."""
        logger.info("Starting training...")
        
        for epoch in range(num_epochs):
            logger.info(f'Epoch {epoch+1}/{num_epochs}')
            
            # Train one epoch
            train_loss = self.train_epoch(train_loader)
            logger.info(f'Training Loss: {train_loss:.4f}')
            
            # Evaluate on validation set
            val_loss = self.evaluate(val_loader)
            logger.info(f'Validation Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save the best model
                self.save_checkpoint(epoch, val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f'Early stopping triggered after {epoch+1} epochs')
                    break
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        checkpoint_path = os.path.join('checkpoints', 'best_model.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f'Saved checkpoint to {checkpoint_path}')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load configuration
    config = {
        'start_date': '2020-01-01',
        'end_date': '2023-12-31',
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
        'window_size': 20,
        'model': {
            'input_dim': 13,  # Updated to match actual number of features
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.3
        },
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 50,
        'early_stopping_patience': 10
    }

    # Create dataset
    logger.info("Downloading and processing data...")
    dataset = StockDataset(
        symbols=config['symbols'],
        window_size=config['window_size'],
        start_date=config['start_date'],
        end_date=config['end_date']
    )

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    logger.info(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize model
    model = THGNNPredictor(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model'].get('dropout', 0.2)
    ).to(device)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        learning_rate=config['learning_rate'],
        device=device,
        early_stopping_patience=config['early_stopping_patience']
    )

    # Train model
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader, config['num_epochs'])

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss = trainer.evaluate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
