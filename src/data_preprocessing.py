import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple, Dict
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataProcessor:
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        """
        Initialize the stock data processor
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.scaler = StandardScaler()
        self.target_scaler = None
        
    def download_data(self):
        """Download stock data using yfinance"""
        logger.info("Starting data download for %d symbols...", len(self.symbols))
        for symbol in tqdm(self.symbols, desc="Downloading stock data"):
            try:
                # Add delay to prevent rate limiting
                time.sleep(1)
                stock = yf.Ticker(symbol)
                df = stock.history(start=self.start_date, end=self.end_date)
                if len(df) > 0:
                    self.data[symbol] = df
                    logger.info("Downloaded %d records for %s", len(df), symbol)
                else:
                    logger.warning("No data available for %s", symbol)
            except Exception as e:
                logger.error("Error downloading %s: %s", symbol, str(e))
                
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for a single stock"""
        try:
            df = df.copy()
            
            # Basic price indicators
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close']/df['Close'].shift(1))
            
            # Moving averages
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
            # Volatility indicators
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Calculate Bollinger Bands components separately
            middle_band = df['Close'].rolling(window=20).mean()
            std_dev = df['Close'].rolling(window=20).std()
            df['upper_band'] = middle_band + (2 * std_dev)
            df['lower_band'] = middle_band - (2 * std_dev)
            
            # Momentum indicators
            df['RSI'] = self.calculate_rsi(df['Close'])
            df['MACD'] = self.calculate_macd(df['Close'])
            df['ROC'] = df['Close'].pct_change(periods=10)
            
            # Volume indicators
            df['OBV'] = self.calculate_obv(df)
            df['volume_ma'] = df['Volume'].rolling(window=20).mean()
            
            # Fill NaN values with 0
            df = df.fillna(0)
            
            # Clip extreme values
            for col in df.columns:
                if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = np.clip(df[col], -10, 10)
            
            return df
        except Exception as e:
            logger.error("Error calculating features: %s", str(e))
            raise
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)."""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def calculate_obv(self, df):
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = float(df['Volume'].iloc[0])
        
        for i in range(1, len(df)):
            if float(df['Close'].iloc[i]) > float(df['Close'].iloc[i-1]):
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif float(df['Close'].iloc[i]) < float(df['Close'].iloc[i-1]):
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def create_graph_data(self, date: str) -> Data:
        """Create graph data for a specific date"""
        try:
            features_list = []
            feature_columns = [
                'returns', 'log_returns', 'MA5', 'MA20', 'MA50',
                'volatility', 'upper_band', 'lower_band', 'RSI', 'MACD', 'ROC', 'OBV', 'volume_ma'
            ]
            
            for symbol in self.symbols:
                if symbol in self.data:
                    df = self.data[symbol]
                    if date in df.index:
                        features = df.loc[date, feature_columns].values
                        features_list.append(features)
                        
            if not features_list:
                return None
                
            # Create node features
            x = torch.FloatTensor(features_list).to(torch.float32)
            
            # Create edges (fully connected graph)
            num_nodes = len(features_list)
            edge_index = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edge_index.append([i, j])
            edge_index = torch.LongTensor(edge_index).t()
            
            return Data(x=x, edge_index=edge_index)
        except Exception as e:
            logger.error("Error creating graph data for date %s: %s", date, str(e))
            return None
    
    def prepare_dataset(self, window_size: int = 20) -> Tuple[List[Tuple[Data, torch.Tensor]], List[Tuple[Data, torch.Tensor]], List[Tuple[Data, torch.Tensor]]]:
        """Prepare dataset with temporal graphs and target values"""
        logger.info("Preparing dataset...")
        try:
            # Process all stocks
            logger.info("Processing technical indicators for each stock...")
            for symbol in tqdm(self.symbols, desc="Processing stocks"):
                if symbol in self.data:
                    self.data[symbol] = self.calculate_technical_indicators(self.data[symbol])
            
            # Get common dates
            logger.info("Finding common trading days...")
            date_sets = [set(self.data[symbol].index) for symbol in self.symbols if symbol in self.data]
            if not date_sets:
                raise ValueError("No data available for any symbols")
            common_dates = sorted(date_sets[0].intersection(*date_sets[1:]))
            
            logger.info(f"Found {len(common_dates)} common trading days")
            
            # Pre-calculate all graph data to avoid repeated calculations
            logger.info("Pre-calculating graph data for all dates...")
            graph_data = {}
            for date in tqdm(common_dates, desc="Creating graph data"):
                graph = self.create_graph_data(date)
                if graph is not None:
                    graph_data[date] = graph
            
            # Create temporal sequences
            logger.info("Creating temporal sequences...")
            sequences = []
            valid_dates = sorted(graph_data.keys())
            
            for i in tqdm(range(len(valid_dates) - window_size), desc="Creating temporal sequences"):
                seq_dates = valid_dates[i:i + window_size]
                target_date = valid_dates[i + window_size]
                
                # Get sequence of graphs
                seq_graphs = [graph_data[date] for date in seq_dates]
                
                # Get target (next day return for the first stock, typically the market index)
                target = torch.tensor(self.data[self.symbols[0]].loc[target_date, 'returns'], dtype=torch.float32)
                
                sequences.append((seq_graphs, target))
            
            # Split into train/val/test
            total_size = len(sequences)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            
            train_data = sequences[:train_size]
            val_data = sequences[train_size:train_size + val_size]
            test_data = sequences[train_size + val_size:]
            
            logger.info(f"Created dataset with {len(sequences)} samples")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error("Error preparing dataset: %s", str(e))
            raise
    
    def normalize_features(self, datasets):
        """Normalize node features across the dataset"""
        logger.info("Normalizing features...")
        try:
            # Collect all features from all datasets
            all_features = []
            for dataset in datasets:  # Iterate through train, val, test
                for sequence, _ in dataset:
                    for graph in sequence:
                        all_features.append(graph.x.numpy())
            
            # Fit scaler
            all_features = np.vstack(all_features)
            self.scaler.fit(all_features)
            
            # Transform features for each dataset
            normalized_datasets = []
            for dataset in datasets:
                normalized_dataset = []
                for sequence, target in dataset:
                    normalized_sequence = []
                    for graph in sequence:
                        x_normalized = torch.FloatTensor(
                            self.scaler.transform(graph.x.numpy())
                        ).to(torch.float32)
                        normalized_sequence.append(
                            Data(x=x_normalized, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
                        )
                    normalized_dataset.append((normalized_sequence, target))
                normalized_datasets.append(normalized_dataset)
            
            logger.info("Feature normalization completed")
            return normalized_datasets
        except Exception as e:
            logger.error("Error normalizing features: %s", str(e))
            raise

class StockDataset:
    def __init__(self, symbols, start_date, end_date, window_size=20):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        
        # Define sector information
        self.sector_info = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'META': 'Technology',
            'AMZN': 'Consumer Cyclical'
        }
        
        # Define sector relationships (higher value = stronger relationship)
        self.sector_relationships = {
            ('Technology', 'Technology'): 1.0,
            ('Technology', 'Consumer Cyclical'): 0.7,
            ('Consumer Cyclical', 'Consumer Cyclical'): 1.0,
            ('Consumer Cyclical', 'Technology'): 0.7
        }
        
        self.logger = logging.getLogger(__name__)
        self.stock_data = self._download_stock_data()
        self.processed_data = self._process_data()
        self.graph_data = self._create_graph_data()
        self.temporal_sequences = self._create_temporal_sequences()
        
    def _download_stock_data(self):
        """Download stock data for all symbols."""
        logger.info("Starting data download for %d symbols...", len(self.symbols))
        stock_data = {}
        
        for symbol in tqdm(self.symbols, desc="Downloading stock data"):
            try:
                stock = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
                if len(stock) > 0:
                    stock_data[symbol] = stock
                    logger.info(f"Downloaded {len(stock)} records for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {str(e)}")
        
        return stock_data
    
    def _calculate_technical_indicators(self, df):
        """Calculate technical indicators for a stock."""
        # Basic price indicators
        df['returns'] = df['Adj Close'].pct_change()
        df['log_returns'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
        
        # Moving averages
        df['MA5'] = df['Adj Close'].rolling(window=5).mean()
        df['MA20'] = df['Adj Close'].rolling(window=20).mean()
        df['MA50'] = df['Adj Close'].rolling(window=50).mean()
        
        # Volatility indicators
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Calculate Bollinger Bands components separately
        middle_band = df['Adj Close'].rolling(window=20).mean()
        std_dev = df['Adj Close'].rolling(window=20).std()
        df['upper_band'] = middle_band + (2 * std_dev)
        df['lower_band'] = middle_band - (2 * std_dev)
        
        # Momentum indicators
        df['RSI'] = self.calculate_rsi(df['Adj Close'])
        df['MACD'] = self.calculate_macd(df['Adj Close'])
        df['ROC'] = df['Adj Close'].pct_change(periods=10)
        
        # Volume indicators
        df['OBV'] = self.calculate_obv(df)
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)."""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def calculate_obv(self, df):
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = float(df['Volume'].iloc[0])
        
        for i in range(1, len(df)):
            if float(df['Adj Close'].iloc[i]) > float(df['Adj Close'].iloc[i-1]):
                obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
            elif float(df['Adj Close'].iloc[i]) < float(df['Adj Close'].iloc[i-1]):
                obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _process_data(self):
        """Process raw stock data into features."""
        logger.info("Preparing dataset...")
        processed_data = {}
        
        # Process each stock
        for symbol, df in self.stock_data.items():
            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)
            
            # Select features
            features = ['returns', 'log_returns', 'MA5', 'MA20', 'MA50',
                       'volatility', 'upper_band', 'lower_band', 'RSI', 'MACD', 'ROC', 'OBV', 'volume_ma']
            
            processed_df = df[features].copy()
            
            # Handle missing values
            processed_df = processed_df.fillna(0)
            
            # Normalize features
            for col in processed_df.columns:
                mean = processed_df[col].mean()
                std = processed_df[col].std()
                if std != 0:
                    processed_df[col] = (processed_df[col] - mean) / std
                else:
                    processed_df[col] = 0
            
            processed_data[symbol] = processed_df
        
        return processed_data
    
    def _create_edge_index_and_weights(self, date):
        """Create edge index and weights based on sector relationships and correlations."""
        num_stocks = len(self.symbols)
        edges = []
        edge_weights = []
        
        # Calculate pairwise correlations using a 30-day window
        returns_matrix = np.array([
            self.processed_data[symbol].loc[:date]['returns'].tail(30).values
            for symbol in self.symbols
        ])
        correlation_matrix = np.corrcoef(returns_matrix)
        
        # Create edges between all pairs of stocks
        for i in range(num_stocks):
            for j in range(num_stocks):
                if i != j:
                    symbol_i = self.symbols[i]
                    symbol_j = self.symbols[j]
                    
                    # Get sector relationship weight
                    sector_i = self.sector_info[symbol_i]
                    sector_j = self.sector_info[symbol_j]
                    sector_weight = self.sector_relationships.get(
                        (sector_i, sector_j),
                        self.sector_relationships.get((sector_j, sector_i), 0.5)
                    )
                    
                    # Get correlation weight
                    corr_weight = abs(correlation_matrix[i, j])
                    if np.isnan(corr_weight):
                        corr_weight = 0.5
                    
                    # Combine sector and correlation weights
                    weight = 0.6 * sector_weight + 0.4 * corr_weight
                    
                    # Add edge if weight is significant
                    if weight > 0.3:
                        edges.append([i, j])
                        edge_weights.append(weight)
        
        if not edges:  # If no significant edges, create fully connected graph
            for i in range(num_stocks):
                for j in range(num_stocks):
                    if i != j:
                        edges.append([i, j])
                        edge_weights.append(0.5)  # Default weight
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        
        return edge_index, edge_weights
        
    def _create_graph(self, date):
        """Create graph with node features, edge index, and edge weights."""
        # Get node features
        node_features = []
        for symbol in self.symbols:
            features = [
                self.processed_data[symbol].loc[date, 'returns'],
                self.processed_data[symbol].loc[date, 'log_returns'],
                self.processed_data[symbol].loc[date, 'MA5'],
                self.processed_data[symbol].loc[date, 'MA20'],
                self.processed_data[symbol].loc[date, 'MA50'],
                self.processed_data[symbol].loc[date, 'volatility'],
                self.processed_data[symbol].loc[date, 'upper_band'],
                self.processed_data[symbol].loc[date, 'lower_band'],
                self.processed_data[symbol].loc[date, 'RSI'],
                self.processed_data[symbol].loc[date, 'MACD'],
                self.processed_data[symbol].loc[date, 'ROC'],
                self.processed_data[symbol].loc[date, 'OBV'],
                self.processed_data[symbol].loc[date, 'volume_ma']
            ]
            node_features.append(features)
        
        x = torch.FloatTensor(node_features).to(torch.float32)
        
        # Create edge index and weights
        edge_index, edge_weights = self._create_edge_index_and_weights(date)
        
        # Create graph
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weights.unsqueeze(-1)
        )
        
        return graph
    
    def _get_common_dates(self):
        """Find common trading days across all stocks."""
        common_dates = None
        for symbol in self.symbols:
            dates = self.processed_data[symbol].index
            if common_dates is None:
                common_dates = set(dates)
            else:
                common_dates = common_dates.intersection(dates)
        
        common_dates = sorted(list(common_dates))
        logger.info(f"Found {len(common_dates)} common trading days")
        return common_dates
    
    def _create_graph_data(self):
        """Create graph data for each time step."""
        logger.info("Pre-calculating graph data for all dates...")
        graph_data = []
        
        # Get common dates across all stocks
        dates = self._get_common_dates()
        
        for date in tqdm(dates, desc="Creating graph data"):
            graph = self._create_graph(date)
            graph_data.append(graph)
        
        return graph_data
    
    def _create_temporal_sequences(self):
        """Create temporal sequences of graphs."""
        logger.info("Creating temporal sequences...")
        sequences = []
        targets = []
        
        dates = self._get_common_dates()
        
        for i in tqdm(range(len(dates) - self.window_size), desc="Creating temporal sequences"):
            # Get sequence of graphs
            sequence = self.graph_data[i:i + self.window_size]
            
            # Get target (next day return for first stock)
            target_date = dates[i + self.window_size]
            target = self.processed_data[self.symbols[0]].loc[target_date, 'returns']
            
            sequences.append(sequence)
            targets.append(target)
        
        return list(zip(sequences, targets))
    
    def __len__(self):
        return len(self.temporal_sequences)
    
    def __getitem__(self, idx):
        sequence, target = self.temporal_sequences[idx]
        return sequence, torch.tensor(target, dtype=torch.float32)
