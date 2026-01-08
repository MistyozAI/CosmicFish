import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CosmicConfig:
    """Configuration class for CosmicFish."""

    def __init__(self,
                 vocab_size=50257,
                 block_size=2048,  # 512 if 120M or 90M
                 n_layer=24, # 12 for 120M & 10 for 90M
                 n_head=16,
                 n_embd=960, # 704 for 120M & 640 for 90M
                 bias=True,
                 dropout=0.1,
                 n_query_groups=4,  # For Grouped-Query Attention (GQA)
                 eps=1e-6,  # For RMSNorm
                 use_rotary=True,  # For Rotary Positional Embeddings
                 use_swiglu=True,  # For SwiGLU activation
                 use_qk_norm=False,  # Optional query-key normalization
                 use_gqa=True):  # For Grouped-Query Attention
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.bias = bias
        self.dropout = dropout
        self.eps = eps
        self.use_rotary = use_rotary
        self.use_swiglu = use_swiglu
        self.use_qk_norm = use_qk_norm
        self.use_gqa = use_gqa
        self.n_query_groups = n_query_groups if use_gqa else n_head
        # Ensure n_head is divisible by n_query_groups
        assert n_head % self.n_query_groups == 0, "n_head must be divisible by n_query_groups"


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * (x / rms)


def precompute_freqs_cis(dim, end, theta=10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions
    """
    # Ensure end matches the maximum sequence length (block_size)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    # Complex exponentials (cosine and sine components)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply rotary embeddings to input tensors using the given frequency tensor
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # This is the part that needs fixing - ensure we only use the frequencies we need
    seq_len = xq_.size(2)

    # Make sure freqs_cis has enough values for our sequence length
    if freqs_cis.size(0) < seq_len:
        raise ValueError(f"freqs_cis has only {freqs_cis.size(0)} values but sequence length is {seq_len}")

    # Just use the sequence length we need
    freqs_cis_seq = freqs_cis[:seq_len]

    # Apply rotary embeddings
    xq_out = torch.view_as_real(xq_ * freqs_cis_seq.unsqueeze(0)).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_seq.unsqueeze(0)).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLU(nn.Module):
    """
    SwiGLU, maybe change for CosmicFish 1.5??
    """

    def __init__(self, size, bias=True):
        super().__init__()
        self.linear_gate = nn.Linear(size, size, bias=bias)
        self.linear_swish = nn.Linear(size, size, bias=bias)

    def forward(self, x):
        gate = self.linear_gate(x)
        swish = self.linear_swish(x) * torch.sigmoid(self.linear_swish(x))
        return gate * swish


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) implementation
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key part that needs fixing: projections need the right dimensions
        head_dim = config.n_embd // config.n_head
        self.head_dim = head_dim
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_query_groups = config.n_query_groups

        # Calculate the correct dimensions
        self.kv_heads = config.n_head // config.n_query_groups if config.use_gqa else config.n_head

        # This was the issue - calculate the correct output size for the QKV projection
        # For queries: we need n_head * head_dim
        # For keys and values: we need kv_heads * head_dim each
        qkv_proj_size = (config.n_head + 2 * self.kv_heads) * head_dim

        # Fix the projection to have the right output size
        self.c_attn = nn.Linear(config.n_embd, qkv_proj_size, bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.dropout = config.dropout

        # flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # For pre-computation of attention pattern (when not using flash attention)
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

        # query-key normalization
        self.qk_norm = getattr(config, 'use_qk_norm', False)
        if self.qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=getattr(config, 'eps', 1e-6))
            self.k_norm = RMSNorm(head_dim, eps=getattr(config, 'eps', 1e-6))

    def forward(self, x, freqs_cis=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # Calculate qkv for all heads in a batch
        qkv = self.c_attn(x)

        # Get the head dimension
        head_dim = C // self.n_head

        # Split into query, key, value with the correct sizes
        q_size = self.n_head * head_dim
        k_size = self.kv_heads * head_dim
        v_size = self.kv_heads * head_dim

        # Make sure the sizes sum up to what we expect
        assert q_size + k_size + v_size == qkv.size(
            -1), f"QKV sizes {q_size}+{k_size}+{v_size} don't match {qkv.size(-1)}"

        # Split into query, key, value
        q, k, v = qkv.split([q_size, k_size, v_size], dim=2)

        # Reshape query, key, value for attention computation
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.kv_heads, head_dim).transpose(1, 2)  # (B, kv_heads, T, hs)
        v = v.view(B, T, self.kv_heads, head_dim).transpose(1, 2)  # (B, kv_heads, T, hs)

        # If kv_heads < n_heads, we need to repeat k and v
        if self.kv_heads < self.n_head:
            repeats = self.n_head // self.kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # Apply rotary embeddings if provided
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # Apply query-key normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Compute attention scores
        if self.flash:
            # Use Flash Attention if available
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # Compute manually with explicit causal mask
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reshape back to (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection and dropout
        y = self.resid_dropout(self.c_proj(y))

        return y


class Block(nn.Module):
    """
    Transformer block with improved components
    """

    def __init__(self, config):
        super().__init__()
        # Pre-normalization layers (using RMSNorm)
        self.ln_1 = RMSNorm(config.n_embd, eps=config.eps)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.eps)

        # Attention layer - using GQA by default
        self.attn = GroupedQueryAttention(config)

        # Different MLP implementations based on configuration
        if config.use_swiglu:
            # SwiGLU MLP - FIXED: gate projection now matches up projection dimension
            self.mlp = nn.ModuleDict(dict(
                gate=nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),  # Changed from n_embd to 4*n_embd
                up=nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                down=nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
                act=nn.SiLU(),
                dropout=nn.Dropout(config.dropout),
            ))

            m = self.mlp
            self.mlpf = lambda x: m.dropout(m.down(m.act(m.up(x)) * m.gate(x)))
        else:
            # Traditional MLP with GELU
            self.mlp = nn.ModuleDict(dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
                act=nn.GELU(),  # Fallback to GELU if SwiGLU is not used
                dropout=nn.Dropout(config.dropout),
            ))

            m = self.mlp
            self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x, freqs_cis=None):
        # Pre-norm architecture
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlpf(self.ln_2(x))
        return x


class CosmicFish(nn.Module):
    """
    CosmicFish model with:
    - Rotary Positional Embeddings
    - Grouped-Query Attention
    - SwiGLU activation
    - RMSNorm
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize transformer components
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd, eps=config.eps),  # Final layer norm
        ))

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Share weights between embedding and linear head
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute rotary embedding frequencies if using RoPE
        if config.use_rotary:
            head_dim = config.n_embd // config.n_head
            self.freqs_cis = precompute_freqs_cis(head_dim, config.block_size)
        else:
            self.freqs_cis = None
            # Use traditional positional embeddings if not using rotary
            self.transformer.wpe = nn.Embedding(config.block_size, config.n_embd)

        # Initialize weights
        self.apply(self._init_weights)

        # Apply a special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize the weights - sophisticated initialization strategy."""
        if isinstance(module, nn.Linear):
            # Special initialization for SwiGLU components
            if any(x in module._get_name().lower() for x in ['gate', 'up', 'down']):
                std = 1 / math.sqrt(module.in_features)
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This method sets up the optimizer with weight decay management.
        """
        # Separate parameters with and without weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, RMSNorm, torch.nn.Embedding)

        # Categorize parameters based on module type
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn

                # Skip the shared embedding weight from lm_head
                if fpn == 'lm_head.weight':
                    continue

                # Categorize parameters
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Validate all parameters are accounted for
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {k: v for k, v in param_dict.items() if k != 'lm_head.weight'}

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(
            param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"

        # Create optimizer groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        # Initialize optimizer
        betas = tuple(betas)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        return optimizer

    def forward(self, idx, targets=None):
        """Forward pass through the model."""
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Get the token embeddings
        tok_emb = self.transformer.wte(idx)

        # Process based on whether we're using rotary embeddings or not
        if self.config.use_rotary:
            # Direct token embeddings with dropout if using rotary
            x = self.transformer.drop(tok_emb)
            # Ensure freqs_cis is on the right device
            freqs_cis = self.freqs_cis.to(device) if self.freqs_cis is not None else None
        else:
            # Add in the traditional positional embeddings
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
            freqs_cis = None

        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x, freqs_cis)

        # Apply final normalization
        x = self.transformer.ln_f(x)

        # Calculate outputs and loss
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # For inference, only compute logits for the last token
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text by sampling from the model, token by token.
        """
        for _ in range(max_new_tokens):
            # Ensure sequence doesn't exceed maximum length
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
