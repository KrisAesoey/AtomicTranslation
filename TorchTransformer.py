import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        max_length,
        embed_size=256,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        dropout=0.1,
        pad_idx=0,
        device="cuda",
    ):
        super().__init__()

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.src_position = nn.Embedding(max_length, embed_size)
        self.trg_position = nn.Embedding(max_length, embed_size)
        
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

        self.dropout = nn.Dropout(dropout)
        
        self.pad_idx = pad_idx
        
        self.device = device
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        N_src, src_length = src.shape
        N_trg, trg_length = trg.shape

        trg_mask = self.make_trg_mask(trg)

        src_positions = torch.arange(0, src_length).expand(
            N_src, src_length).to(self.device)
        trg_positions = torch.arange(0, trg_length).expand(
            N_trg, trg_length).to(self.device)

        src = self.dropout(self.src_embedding(
            src) + self.src_position(src_positions))
        trg = self.dropout(self.trg_embedding(
            trg) + self.trg_position(trg_positions))

        out = self.transformer(src, trg, tgt_mask=trg_mask)
        out = self.fc_out(out)
        return out
