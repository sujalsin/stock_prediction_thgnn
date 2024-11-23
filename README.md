# Temporal and Heterogeneous Graph Neural Network for Stock Price Prediction

This project implements a Temporal and Heterogeneous Graph Neural Network (THGNN) for stock price prediction, based on the paper "Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction" by Sheng Xiang et al.

## Project Structure

```
stock_prediction_thgnn/
├── checkpoints/            # Saved model checkpoints
├── configs/               # Configuration parameters
├── data/                 # Stock market data
├── src/                  # Source code
│   ├── data_preprocessing.py  # Data processing and graph construction
│   ├── thgnn_model.py        # THGNN model architecture
│   └── test.py               # Training and evaluation scripts
└── README.md
```

## Features

- Temporal graph neural network for capturing dynamic stock relations
- Multi-head attention mechanism for temporal and graph-level attention
- Comprehensive technical indicators:
  1. Returns
  2. Log Returns
  3. MA5 (5-day Moving Average)
  4. MA20 (20-day Moving Average)
  5. MA50 (50-day Moving Average)
  6. Volatility
  7. Upper Bollinger Band
  8. Lower Bollinger Band
  9. RSI (Relative Strength Index)
  10. MACD
  11. ROC (Rate of Change)
  12. OBV (On-Balance Volume)
  13. Volume Moving Average
- Dynamic graph construction for each trading day
- Early stopping and model checkpointing
- Configurable hyperparameters via YAML config

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Configure the model and training parameters in `configs/config.yaml`:
   - Adjust stock symbols (currently: AAPL, GOOGL, MSFT, AMZN, META)
   - Set date range (currently: 2020-01-01 to 2023-12-31)
   - Modify model architecture
   - Tune training parameters

2. Run the training script:
```bash
python src/test.py
```

## Model Architecture

The THGNN model consists of several key components:

1. **Graph Construction**:
   - Creates dynamic graphs for each trading day
   - Nodes represent stocks with 13 technical features
   - Edges represent stock relationships
   - Window size of 20 trading days

2. **Temporal Attention**:
   - Captures temporal dependencies
   - Uses multi-head attention mechanism
   - Processes sequence of daily graphs
   - Handles variable-length sequences

3. **Graph Neural Network**:
   - Models stock relationships
   - Processes node features through graph convolutions
   - Learns hierarchical representations
   - Incorporates skip connections

4. **Prediction Layer**:
   - Forecasts next-day returns
   - Combines temporal and graph features
   - Uses MSE loss for return prediction

## Performance Metrics

Current model performance:
- Training Loss: ~1.0879
- Validation Loss: 0.7198 (best)
- Test Loss: 1.0368
- Early stopping after 13 epochs

Dataset Statistics:
- Training: 690 samples
- Validation: 147 samples
- Testing: 149 samples

## Data Processing

- Window Size: 20 trading days
- 5 Major Tech Stocks: AAPL, GOOGL, MSFT, AMZN, META
- Date Range: 2020-01-01 to 2023-12-31
- Feature Normalization: Z-score standardization
- Dynamic graph construction based on stock correlations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{xiang2021temporal,
  title={Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction},
  author={Xiang, Sheng and others},
  year={2021}
}
