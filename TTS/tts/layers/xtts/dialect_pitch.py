import torch
import torch.nn as nn

class DialectPitchPredictor(nn.Module):
    """
    Given a dialect ID, predict an f0 sequence of arbitrary length.
    """
    def __init__(self, dialect_emb_module: nn.Embedding, hidden_dim: int = 256):
        super().__init__()
        self.dialect_emb = dialect_emb_module
        self.mlp = nn.Sequential(
            nn.Linear(self.dialect_emb.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, dialect_id: torch.LongTensor, target_len: int) -> torch.Tensor:
        emb = self.dialect_emb(dialect_id)     
        h   = self.mlp(emb)                
        f0_1 = self.proj(h)               
        return f0_1.expand(-1, target_len)     

class PitchEmbedding(nn.Module):
    def __init__(self, num_bins=256, embed_dim=512, f0_min=50.0, f0_max=800.0):
        super().__init__()
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.log_min = None
        self.log_range = None
        self.num_bins = num_bins
        self.embedding = nn.Embedding(num_bins, embed_dim)

    def forward(self, f0_seq: torch.Tensor) -> torch.Tensor:
        if self.log_min is None or self.log_min.device != f0_seq.device:
            log_min = torch.log(torch.tensor(self.f0_min, device=f0_seq.device, dtype=f0_seq.dtype))
            log_max = torch.log(torch.tensor(self.f0_max, device=f0_seq.device, dtype=f0_seq.dtype))
            self.log_min = log_min
            self.log_range = log_max - log_min

        f0 = torch.clamp(f0_seq, self.f0_min, self.f0_max)
        f0_norm = (torch.log(f0) - self.log_min) / self.log_range
        idx = (f0_norm * (self.num_bins - 1)).round().clamp(0, self.num_bins - 1).long()
        return self.embedding(idx) 
