import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Multi-head attention parameters
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        
        # Query, Key, Value transformations
        self.q_transform = nn.Linear(hidden_dim, hidden_dim)
        self.k_transform = nn.Linear(hidden_dim, hidden_dim)
        self.v_transform = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Scaling factor for dot product attention
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Transform input into query, key, value
        Q = self.q_transform(x)
        K = self.k_transform(x)
        V = self.v_transform(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = torch.softmax(energy, dim=-1)
        
        # Apply attention to values
        x = torch.matmul(attention, V)
        
        # Reshape and project output
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hidden_dim)
        x = self.output_proj(x)
        
        return x

class THGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_layers=2, dropout=0.1):
        super(THGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Feature transformation layer
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Skip connection weights
        self.skip_weights = nn.Parameter(torch.ones(num_layers))
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # Transform input features
        x = self.feature_transform(x)
        
        # Store intermediate representations for skip connections
        layer_outputs = []
        
        # Apply graph convolution layers with skip connections
        current_x = x
        for conv in self.conv_layers:
            # Apply convolution
            conv_out = conv(current_x, edge_index, edge_weight)
            conv_out = F.relu(conv_out)
            
            # Apply layer normalization
            conv_out = self.layer_norm(conv_out)
            
            # Store layer output
            layer_outputs.append(conv_out)
            
            # Update current representation
            current_x = conv_out
        
        # Compute weighted sum of layer outputs using skip connections
        skip_weights = self.softmax(self.skip_weights)
        final_representation = sum(w * out for w, out in zip(skip_weights, layer_outputs))
        
        # Apply temporal attention if in training mode
        if self.training:
            final_representation = self.temporal_attention(final_representation)
        
        # Make prediction
        out = self.predictor(final_representation)
        
        return out

class THGNNPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(THGNNPredictor, self).__init__()
        
        # THGNN for processing individual graphs
        self.thgnn = THGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Temporal attention for sequence processing
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, sequences):
        batch_size = len(sequences)
        seq_len = len(sequences[0])
        
        # Process each graph in each sequence
        sequence_embeddings = []
        for i in range(batch_size):
            seq_embeddings = []
            for j in range(seq_len):
                graph = sequences[i][j]
                # Process individual graph
                # Reshape x to ensure it's 2D [num_nodes, features]
                x = graph.x.view(-1, self.thgnn.input_dim)
                # Process individual graph
                graph_embedding = self.thgnn(x, graph.edge_index)
                # Take mean of node embeddings as graph embedding
                graph_embedding = graph_embedding.mean(dim=0)
                seq_embeddings.append(graph_embedding)
            sequence_embeddings.append(torch.stack(seq_embeddings))
        
        # Stack all sequences
        sequence_embeddings = torch.stack(sequence_embeddings)  # [batch_size, seq_len, hidden_dim]
        
        # Apply temporal attention
        attended = self.temporal_attention(sequence_embeddings)
        
        # Take the last timestep's representation
        final_embedding = attended[:, -1, :]
        
        # Make prediction
        prediction = self.predictor(final_embedding)
        
        return prediction
