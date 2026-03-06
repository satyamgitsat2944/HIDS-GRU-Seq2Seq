"""
GRU Encoder-Decoder with Bahdanau Attention
Based on: "Intrusion Prediction with System-call Sequence-to-Sequence Model"

Architecture:
    Encoder: Multi-layer GRU
    Attention: Bahdanau (additive) attention
    Decoder: GRU with attention context at each step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Multi-layer GRU Encoder.
    
    Paper Section 3.2:
    - Reads source syscall sequence
    - Produces hidden states at each timestep
    - Final hidden state becomes context vector c
    
    Paper hyperparameters:
    - hidden_dim = 256
    - num_layers  = 3
    - dropout     = 0.5
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers=3, dropout=0.5):
        super().__init__()

        # Embedding layer: maps syscall token ID -> dense vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multi-layer GRU
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,       # input shape: (batch, seq_len, features)
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, src, src_lens):
        """
        src     : (batch, src_len)  — token IDs
        src_lens: (batch,)          — actual lengths before padding
        
        Returns:
            outputs: (batch, src_len, hidden_dim) — all hidden states
            hidden : (num_layers, batch, hidden_dim) — final hidden state
        """
        # Embed tokens
        embedded = self.dropout(self.embedding(src))
        # (batch, src_len, embed_dim)

        # Pack padded sequences for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Pass through GRU
        packed_outputs, hidden = self.gru(packed)

        # Unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True
        )
        # outputs: (batch, src_len, hidden_dim)
        # hidden : (num_layers, batch, hidden_dim)

        return outputs, hidden


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    
    Paper Section 3.2, Equations 5, 6, 7:
    
    e_ij = v^T * tanh(W_a * h_j + U_a * s_i)
    a_ij = exp(e_ij) / sum_k(exp(e_ik))
    c_i  = sum_j(a_ij * h_j)
    
    Where:
        h_j = encoder hidden state at step j
        s_i = decoder hidden state at step i
        c_i = context vector (weighted sum of encoder states)
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: (batch, src_len, hidden_dim)
        decoder_hidden : (batch, hidden_dim)
        
        Returns:
            context     : (batch, hidden_dim)
            attn_weights: (batch, src_len)
        """
        # decoder_hidden: (batch, hidden_dim) -> (batch, 1, hidden_dim)
        decoder_hidden = decoder_hidden.unsqueeze(1)

        # Energy scores
        # W_a * h_j : (batch, src_len, hidden_dim)
        # U_a * s_i : (batch, 1, hidden_dim) — broadcasts over src_len
        energy = torch.tanh(
            self.W_a(encoder_outputs) + self.U_a(decoder_hidden)
        )
        # energy: (batch, src_len, hidden_dim)

        # Squeeze to scalar score per position
        scores = self.v_a(energy).squeeze(-1)
        # scores: (batch, src_len)

        # Normalize with softmax -> attention weights
        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights: (batch, src_len)

        # Weighted sum of encoder outputs -> context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        # context: (batch, hidden_dim)

        return context, attn_weights


class Decoder(nn.Module):
    """
    GRU Decoder with Bahdanau Attention.
    
    At each step:
    1. Embed previous token
    2. Compute attention context from encoder states
    3. Concatenate embedding + context -> GRU input
    4. Project GRU output -> vocabulary scores
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_layers=3, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim)

        # GRU input = embedding + context vector
        self.gru = nn.GRU(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output projection: hidden -> vocab
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, token, hidden, encoder_outputs):
        """
        One decoding step.
        
        token          : (batch,)                    — previous token
        hidden         : (num_layers, batch, hidden) — decoder hidden state
        encoder_outputs: (batch, src_len, hidden)    — all encoder states
        
        Returns:
            logits      : (batch, vocab_size)
            hidden      : (num_layers, batch, hidden)
            attn_weights: (batch, src_len)
        """
        # Embed token: (batch,) -> (batch, 1, embed_dim)
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))

        # Attention: use top layer of decoder hidden state
        top_hidden = hidden[-1]  # (batch, hidden_dim)
        context, attn_weights = self.attention(encoder_outputs, top_hidden)

        # Concatenate embedding + context
        # embedded: (batch, 1, embed_dim)
        # context : (batch, hidden_dim) -> (batch, 1, hidden_dim)
        gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)

        # GRU step
        output, hidden = self.gru(gru_input, hidden)
        # output: (batch, 1, hidden_dim)

        # Project to vocabulary
        logits = self.fc_out(output.squeeze(1))
        # logits: (batch, vocab_size)

        return logits, hidden, attn_weights


class Seq2SeqHIDS(nn.Module):
    """
    Full Seq2Seq HIDS Model.
    Combines Encoder + Attention Decoder.
    
    Paper: GRU-3 with lr=0.1 achieves best BLEU score of 40.5
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=3, dropout=0.5):
        super().__init__()

        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim,
                               num_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim,
                               num_layers, dropout)

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        """
        Full forward pass with teacher forcing.
        
        Teacher forcing (paper training strategy):
            - 50% of the time: feed REAL previous token to decoder
            - 50% of the time: feed PREDICTED previous token
            This speeds up training and improves stability.
        
        src     : (batch, src_len)
        src_lens: (batch,)
        tgt     : (batch, tgt_len)
        
        Returns:
            outputs: (batch, tgt_len, vocab_size) — predictions at each step
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]

        # Store predictions
        outputs = torch.zeros(
            batch_size, tgt_len, self.vocab_size
        ).to(src.device)

        # Encode source sequence
        encoder_outputs, hidden = self.encoder(src, src_lens)

        # First decoder input is <SOS> token
        dec_input = tgt[:, 0]  # (batch,)

        # Decode step by step
        for t in range(1, tgt_len):
            logits, hidden, _ = self.decoder(
                dec_input, hidden, encoder_outputs
            )
            outputs[:, t, :] = logits

            # Teacher forcing decision
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher:
                dec_input = tgt[:, t]           # use real token
            else:
                dec_input = logits.argmax(-1)   # use predicted token

        return outputs

    def predict(self, src, src_lens, max_len=20, device='cpu'):
        """
        Greedy decoding for inference (no teacher forcing).
        Returns predicted syscall token sequence.
        """
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src, src_lens)

            # Start with SOS token
            dec_input = torch.tensor([1] * src.shape[0]).to(device)  # SOS=1
            predicted = []

            for _ in range(max_len):
                logits, hidden, attn = self.decoder(
                    dec_input, hidden, encoder_outputs
                )
                dec_input = logits.argmax(-1)
                predicted.append(dec_input.cpu().numpy())

                # Stop if all sequences predicted EOS
                if (dec_input == 2).all():  # EOS=2
                    break

        return predicted