import torch
from torch import nn

from product_transformer.positional_encoding import PositionalEncoding


class ProductTransformer(nn.Module):

    def __init__(self, d_model: int = 768, n_layers: int = 12, n_head: int = 12, d_hidden: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_hidden = d_hidden
        self.dropout = dropout

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, d_hidden, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, sessions tensor ``[B, L, D]``.
            padding_mask: Tensor, shape ``[B, L]``.

        Returns:
            output: Tensor, shape ``[B, L, D]``.
        """
        L = x.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).bool()                      # [L, L]
        x = self.pos_encoder(x)                                                                     # [B, L, D]
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask, mask=causal_mask)        # [B, L, D]
        return x
