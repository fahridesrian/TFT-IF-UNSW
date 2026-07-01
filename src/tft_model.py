"""
tft_model.py
Implementasi Temporal Fusion Transformer (Lim et al., 2021) dari nol,
diadaptasi menjadi classification head (bukan quantile forecasting),
mengikuti komponen pada Gambar 1 proposal:
  - Variable Selection Network (VSN)
  - Gated Residual Network (GRN), memakai GLU (Dauphin et al., 2017),
    ELU (Clevert et al., 2016), dan Layer Normalization (Ba et al., 2016)
  - LSTM encoder-decoder untuk dependensi lokal jangka pendek
  - Interpretable (masked/causal) multi-head attention (Vaswani et al., 2017)
    untuk dependensi jangka panjang
  - Classification head (Linear + softmax) per titik waktu target T,
    menggantikan quantile output layer pada TFT asli, karena tugas di sini
    adalah klasifikasi multikelas (Normal/Generic/Exploits/Fuzzers), bukan
    forecasting kontinu. Fungsi objektif menjadi cross-entropy (bukan
    quantile loss).

Tidak ada fitur statik (static covariates) maupun known-future-inputs yang
dipakai (proposal Bagian 5.4.2), karena keduanya tidak tersedia secara
alami dalam konteks deteksi intrusi ini; seluruh input berupa observasi
historis (past inputs) sepanjang jendela W.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    """Gated Linear Unit (Dauphin et al., 2017)."""
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size * 2)
        self.output_size = output_size

    def forward(self, x):
        x = self.fc(x)
        a, b = x[..., :self.output_size], x[..., self.output_size:]
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    """GRN sesuai Persamaan 5 proposal: GRN(a, c) = LayerNorm(a + GLU(eta1)).
    eta1 adalah transformasi non-linear dari a (dan konteks c bila ada)."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int = None,
                 dropout: float = 0.1, context_size: int = None):
        super().__init__()
        output_size = output_size or input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context_fc = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(hidden_size, output_size)

        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, a, c=None):
        eta2 = self.fc1(a)
        if c is not None and self.context_fc is not None:
            eta2 = eta2 + self.context_fc(c)
        eta1 = self.fc2(self.elu(eta2))
        eta1 = self.dropout(eta1)
        gated = self.glu(eta1)
        residual = self.skip(a)
        return self.layer_norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    """VSN: memilih & membobot fitur secara adaptif per timestep, memakai
    softmax atas GRN gabungan seluruh fitur (Lim et al., 2021)."""
    def __init__(self, num_features: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size

        # Setiap fitur skalar diproyeksikan ke hidden_size lebih dulu.
        self.feature_linears = nn.ModuleList(
            [nn.Linear(1, hidden_size) for _ in range(num_features)]
        )
        self.flattened_grn = GatedResidualNetwork(
            input_size=num_features * hidden_size,
            hidden_size=hidden_size,
            output_size=num_features,
            dropout=dropout,
        )
        self.feature_grns = nn.ModuleList(
            [GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
             for _ in range(num_features)]
        )

    def forward(self, x):
        # x: (batch, time, num_features)
        b, t, f = x.shape
        transformed = [self.feature_linears[i](x[..., i:i + 1]) for i in range(f)]  # list of (b,t,h)
        stacked = torch.stack(transformed, dim=-2)  # (b, t, f, h)
        flattened = stacked.reshape(b, t, f * self.hidden_size)

        weights = self.flattened_grn(flattened)  # (b, t, f)
        weights = F.softmax(weights, dim=-1)

        processed = torch.stack(
            [self.feature_grns[i](transformed[i]) for i in range(f)], dim=-2
        )  # (b, t, f, h)

        combined = (processed * weights.unsqueeze(-1)).sum(dim=-2)  # (b, t, h)
        return combined, weights  # weights dipakai untuk interpretabilitas (Bagian 6.j)


class TemporalFusionTransformer(nn.Module):
    """TFT untuk klasifikasi temporal multikelas horizon-pendek.

    Input : (batch, W, num_features)  -- sekuens historis
    Output: (batch, T, num_classes)   -- probabilitas kelas per titik target
    """
    def __init__(self, num_features: int, hidden_size: int = 64, num_classes: int = 4,
                 T: int = 3, lstm_layers: int = 1, attn_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.T = T
        self.hidden_size = hidden_size

        self.vsn = VariableSelectionNetwork(num_features, hidden_size, dropout)

        self.lstm_encoder = nn.LSTM(
            hidden_size, hidden_size, num_layers=lstm_layers, batch_first=True
        )
        # Decoder queries dibentuk dari T "positional token" yang dipelajari,
        # dikondisikan pada representasi akhir encoder (tidak ada known-future-inputs).
        self.decoder_query_tokens = nn.Parameter(torch.randn(T, hidden_size) * 0.02)
        self.lstm_decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers=lstm_layers, batch_first=True
        )

        self.post_lstm_gate = GLU(hidden_size, hidden_size)
        self.post_lstm_norm = nn.LayerNorm(hidden_size)

        self.static_enrichment_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=attn_heads, dropout=dropout, batch_first=True
        )
        self.post_attn_gate = GLU(hidden_size, hidden_size)
        self.post_attn_norm = nn.LayerNorm(hidden_size)

        self.ff_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.post_ff_gate = GLU(hidden_size, hidden_size)
        self.post_ff_norm = nn.LayerNorm(hidden_size)

        self.output_head = nn.Linear(hidden_size, num_classes)

        self._last_attn_weights = None  # untuk interpret.py

    def forward(self, x):
        # x: (batch, W, num_features)
        b, W, _ = x.shape

        vsn_out, vsn_weights = self.vsn(x)  # (b, W, h), (b, W, f)

        enc_out, (h_n, c_n) = self.lstm_encoder(vsn_out)  # (b, W, h)

        dec_in = self.decoder_query_tokens.unsqueeze(0).expand(b, -1, -1)  # (b, T, h)
        dec_out, _ = self.lstm_decoder(dec_in, (h_n, c_n))  # (b, T, h)

        # Gated skip-connection setelah LSTM (Lim et al., 2021)
        lstm_seq = torch.cat([enc_out, dec_out], dim=1)  # (b, W+T, h)
        vsn_seq = torch.cat(
            [vsn_out, dec_in], dim=1
        )  # padanan input sebelum LSTM untuk residual
        gated_lstm = self.post_lstm_gate(lstm_seq)
        gated_lstm = self.post_lstm_norm(vsn_seq + gated_lstm)

        enriched = self.static_enrichment_grn(gated_lstm)  # (b, W+T, h)

        # Causal mask: posisi target hanya boleh melihat encoder + target sebelumnya,
        # posisi encoder hanya melihat encoder sebelumnya (masked self-attention,
        # sesuai "masked interpretable multi-head attention" pada Gambar 1 proposal).
        seq_len = W + self.T
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )

        attn_out, attn_weights = self.attention(
            enriched, enriched, enriched, attn_mask=causal_mask, need_weights=True,
            average_attn_weights=True,
        )
        self._last_attn_weights = attn_weights.detach()  # (b, seq_len, seq_len)

        gated_attn = self.post_attn_gate(attn_out)
        gated_attn = self.post_attn_norm(enriched + gated_attn)

        ff_out = self.ff_grn(gated_attn)
        gated_ff = self.post_ff_gate(ff_out)
        final = self.post_ff_norm(gated_attn + gated_ff)  # (b, W+T, h)

        target_repr = final[:, W:, :]  # (b, T, h) -- ambil hanya posisi target
        logits = self.output_head(target_repr)  # (b, T, num_classes)

        self._last_vsn_weights = vsn_weights.detach()
        return logits

    def get_last_interpretation(self):
        """Dipanggil setelah forward() dalam mode eval() untuk mengambil
        bobot VSN dan attention terakhir (proposal Bagian 6.j)."""
        return {
            "vsn_weights": self._last_vsn_weights,
            "attn_weights": self._last_attn_weights,
        }
