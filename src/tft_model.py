"""
tft_model.py
Implementasi Temporal Fusion Transformer (Lim et al., 2021) dari nol,
diadaptasi menjadi classification head (bukan quantile forecasting),
mengikuti komponen pada Gambar 1 proposal:
  - Variable Selection Network (VSN) dengan *entity-style embedding* untuk
    fitur kategorikal (Lim et al., 2021, Sec. 4.2)
  - Gated Residual Network (GRN), memakai GLU (Dauphin et al., 2017),
    ELU (Clevert et al., 2016), dan Layer Normalization (Ba et al., 2016)
  - LSTM encoder-decoder untuk dependensi lokal jangka pendek
  - Interpretable masked/causal multi-head attention (Lim et al., 2021,
    Persamaan 13-16; dasar dari Vaswani et al., 2017): W_V di-share antar
    head dan attention di-rata-rata antar head agar bisa dibaca langsung
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
import math

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
    """VSN (Lim et al., 2021, Sec. 4.2 / Persamaan 6-8 proposal): memilih &
    membobot fitur secara adaptif per timestep memakai softmax atas GRN
    gabungan. Setiap fitur ditransformasi ke hidden_size lebih dulu:
      - fitur kategorikal -> entity-style embedding (nn.Embedding per fitur,
        kardinalitas dari preprocessing, termasuk token "unknown")
        -- keputusan terkunci A13/2026-09-27, mengikuti paper asli;
      - fitur kontinu -> transformasi linear(1 -> h)."""
    def __init__(self, num_features: int, hidden_size: int, dropout: float = 0.1,
                 categorical_features: dict = None):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.categorical_features = dict(categorical_features or {})

        self.feature_encoders = nn.ModuleList()
        for i in range(num_features):
            if i in self.categorical_features:
                self.feature_encoders.append(
                    nn.Embedding(self.categorical_features[i], hidden_size)
                )
            else:
                self.feature_encoders.append(nn.Linear(1, hidden_size))

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
        transformed = []
        for i, enc in enumerate(self.feature_encoders):
            if i in self.categorical_features:
                transformed.append(enc(x[..., i].long()))       # (b, t, h)
            else:
                transformed.append(enc(x[..., i:i + 1]))        # (b, t, h)
        stacked = torch.stack(transformed, dim=-2)  # (b, t, f, h)
        flattened = stacked.reshape(b, t, f * self.hidden_size)

        weights = self.flattened_grn(flattened)  # (b, t, f)
        weights = F.softmax(weights, dim=-1)

        processed = torch.stack(
            [self.feature_grns[i](transformed[i]) for i in range(f)], dim=-2
        )  # (b, t, f, h)

        combined = (processed * weights.unsqueeze(-1)).sum(dim=-2)  # (b, t, h)
        return combined, weights  # weights dipakai untuk interpretabilitas (Bagian 6.j)


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head attention (Lim et al., 2021, Persamaan 13-16;
    keputusan terkunci A12/2026-09-27). Berbeda dari multi-head attention
    standar (Vaswani et al., 2017), bobot value W_V DI-SHARE antar head
    (Eq. 14) dan matriks attention di-rata-rata antar head (Eq. 15):

        InterpretableMultiHead(Q, K, V) = H_tilde * W_H
        H_tilde = attention(Q, K, V W_V) = Ã(Q, K) * V W_V
        Ã(Q, K) = (1/H) * sum_h softmax( Q W_q^h (K W_k^h)^T / sqrt(d_attn) )

    Karena rata-rata bersifat linear, mean_h(A_h @ V W_V) = mean_h(A_h) @ V W_V
    persis, sehingga H_tilde di atas sama dengan Eq. 14/16. Hasilnya SATU
    matriks attention (b, L, L) yang dapat dibaca langsung untuk analisis
    temporal (proposal Bagian 6.j.iii). Mask aditif (0 / -inf) untuk causal
    masking diterapkan sebelum softmax."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim harus habis dibagi num_heads")
        self.num_heads = num_heads
        self.d_attn = embed_dim // num_heads
        self.w_q = nn.ModuleList(
            [nn.Linear(embed_dim, self.d_attn, bias=False) for _ in range(num_heads)]
        )
        self.w_k = nn.ModuleList(
            [nn.Linear(embed_dim, self.d_attn, bias=False) for _ in range(num_heads)]
        )
        self.w_v = nn.Linear(embed_dim, self.d_attn, bias=False)  # di-share antar head
        self.w_o = nn.Linear(self.d_attn, embed_dim, bias=False)  # W_H
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        # query/key/value: (batch, L, embed_dim); attn_mask: (L, L) aditif 0/-inf
        v_proj = self.w_v(value)  # (b, L, d_attn) -- sama untuk semua head (Eq. 14)
        weights_sum = None
        for h in range(self.num_heads):
            scores = torch.matmul(self.w_q[h](query), self.w_k[h](key).transpose(-2, -1))
            scores = scores / math.sqrt(self.d_attn)
            if attn_mask is not None:
                scores = scores + attn_mask
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)  # dropout pada prob. attention per-head
            weights_sum = weights if weights_sum is None else weights_sum + weights
        avg_weights = weights_sum / self.num_heads          # Ã(Q, K) -- (b, L, L)
        out = self.w_o(torch.matmul(avg_weights, v_proj))   # H_tilde W_H -- (b, L, embed_dim)
        return out, avg_weights


class TemporalFusionTransformer(nn.Module):
    """TFT untuk klasifikasi temporal multikelas horizon-pendek.

    Input : (batch, W, num_features)  -- sekuens historis
    Output: (batch, T, num_classes)   -- probabilitas kelas per titik target
    """
    def __init__(self, num_features: int, hidden_size: int = 64, num_classes: int = 4,
                 T: int = 3, lstm_layers: int = 1, attn_heads: int = 4, dropout: float = 0.1,
                 categorical_features: dict = None):
        super().__init__()
        self.T = T
        self.hidden_size = hidden_size

        self.vsn = VariableSelectionNetwork(
            num_features, hidden_size, dropout, categorical_features
        )

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

        self.attention = InterpretableMultiHeadAttention(
            embed_dim=hidden_size, num_heads=attn_heads, dropout=dropout
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
            enriched, enriched, enriched, attn_mask=causal_mask
        )
        self._last_attn_weights = attn_weights.detach()  # Ã: (b, seq_len, seq_len)

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
