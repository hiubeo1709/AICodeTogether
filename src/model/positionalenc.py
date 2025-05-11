import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class CodeCompletionModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, max_seq_length=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)  # padding_idx=0 cho <pad>
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def forward(self, x, tgt_mask=None):
        # x: (batch_size, seq_len)
        device = x.device
        seq_len = x.size(1)

        # Embedding và positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Tạo causal mask
        if tgt_mask is None:
            tgt_mask = self.generate_causal_mask(seq_len, device)

        # Sử dụng TransformerDecoder
        # TransformerDecoder cần memory, nhưng trong task này ta chỉ dùng tgt (autoregressive)
        memory = torch.zeros_like(x)  # Dummy memory (không cần trong trường hợp này)
        output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask)

        # Đầu ra
        logits = self.fc_out(output)  # (batch_size, seq_len, vocab_size)
        return logits