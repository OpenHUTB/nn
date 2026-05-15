import torch
import torch.nn as nn

class TemporalAggregator(nn.Module):
    """
    A GRU-based module for aggregating fused sensor features over time.
    """

    def __init__(self, input_dim=256, hidden_dim=256, output_dim=256, num_layers=2):
        """
        input_dim: Dimension of fused features.
        hidden_dim: Hidden state size for GRU.
        output_dim: Final output feature size.
        num_layers: Number of GRU layers.
        """
        super(TemporalAggregator, self).__init__()

        # GRU for temporal aggregation
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Final projection layer to match output size
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state=None):
        """
        Forward pass for temporal feature aggregation.
        :param x: Input tensor (B, T, input_dim) - sequence of fused features
        :param hidden_state: Optional initial hidden state (num_layers, B, hidden_dim)
        :return: Aggregated feature vector (B, output_dim)
        """
        # Process sequence with GRU
        out, hidden_state = self.gru(x, hidden_state)  # (B, T, hidden_dim)

        # Take the last timestep's output
        last_out = out[:, -1, :]  # (B, hidden_dim)

        # Project to output dimension
        final_out = self.projection(last_out)  # (B, output_dim)

        return final_out, hidden_state

