"""
Transformer Model

A clean, standalone transformer implementation using only PyTorch.
No flash-attn or other custom CUDA kernels required.

Features:
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention (GQA) support
    - Optional QK normalization
    - Optional sliding window attention
    - HuggingFace compatible (PreTrainedModel)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_transformer import TransformerConfig


# =============================================================================
# Rotary Position Embeddings
# =============================================================================


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for cos/sin values
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cos/sin cache if needed."""
        if seq_len > self._cached_seq_len or self._cos_cached is None:
            self._cached_seq_len = max(seq_len, self.max_seq_len)
            t = torch.arange(self._cached_seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.

        Args:
            q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
            k: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
            seq_offset: Position offset for generation

        Returns:
            Rotated (q, k) tensors
        """
        seq_len = q.shape[1]
        self._update_cache(seq_offset + seq_len, q.device, q.dtype)

        cos = self._cos_cached[seq_offset : seq_offset + seq_len]
        sin = self._sin_cached[seq_offset : seq_offset + seq_len]

        # Reshape for broadcasting: (seq_len, head_dim) -> (1, seq_len, 1, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)

        return q_rot, k_rot

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotation using cos and sin."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1) * sin + x * cos


# =============================================================================
# Attention
# =============================================================================


class Attention(nn.Module):
    """
    Multi-Head Attention with support for:
    - Grouped Query Attention (GQA)
    - Rotary Position Embeddings (RoPE)
    - Optional QK normalization
    - Optional sliding window
    """

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Optional QK normalization
        if config.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # Rotary embeddings
        self.rotary = RotaryEmbedding(
            dim=self.head_dim,
            base=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
        )

        self.window_size = config.window_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: (batch, seq, hidden) -> (batch, seq, heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Optional QK normalization
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Compute position offset from cache
        seq_offset = 0
        if past_key_values is not None:
            seq_offset = past_key_values[0].shape[1]

        # Apply rotary embeddings
        q, k = self.rotary(q, k, seq_offset=seq_offset)

        # Update KV cache
        if past_key_values is not None:
            k = torch.cat([past_key_values[0], k], dim=1)
            v = torch.cat([past_key_values[1], v], dim=1)

        new_cache = (k, v) if use_cache else None

        # Expand KV for grouped query attention
        if self.num_kv_groups > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
            k = k.reshape(batch_size, -1, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.num_kv_groups, -1)
            v = v.reshape(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, seq, heads, dim) -> (batch, heads, seq, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=self._make_causal_mask(seq_len, k.shape[2], q.device, q.dtype),
            dropout_p=0.0,
            is_causal=False,  # We provide our own mask
        )

        # Reshape back: (batch, heads, seq, dim) -> (batch, seq, hidden)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(attn_output)

        return output, None, new_cache

    def _make_causal_mask(
        self,
        q_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create causal attention mask with optional sliding window."""
        # Create base causal mask
        mask = torch.full((q_len, kv_len), float("-inf"), device=device, dtype=dtype)
        
        # Fill in the causal part (can attend to current and past positions)
        row_idx = torch.arange(q_len, device=device)
        col_idx = torch.arange(kv_len, device=device)
        
        # Account for offset when q_len != kv_len (generation)
        offset = kv_len - q_len
        causal_mask = col_idx <= (row_idx.unsqueeze(1) + offset)
        
        # Apply sliding window if configured
        if self.window_size is not None:
            window_mask = col_idx >= (row_idx.unsqueeze(1) + offset - self.window_size + 1)
            causal_mask = causal_mask & window_mask

        mask = mask.masked_fill(causal_mask, 0.0)
        return mask


# =============================================================================
# MLP
# =============================================================================


class MLP(nn.Module):
    """SwiGLU-style MLP."""

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.hidden_size = config.hidden_size

        # Compute intermediate size
        if config.intermediate_size is not None:
            intermediate_size = config.intermediate_size
        else:
            intermediate_size = config.hidden_size * config.hidden_ratio

        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, self.hidden_size, bias=False)

        # Activation
        if config.hidden_act == "swish" or config.hidden_act == "silu":
            self.act_fn = F.silu
        elif config.hidden_act == "gelu":
            self.act_fn = F.gelu
        elif config.hidden_act == "relu":
            self.act_fn = F.relu
        else:
            raise ValueError(f"Unknown activation: {config.hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# Transformer Block
# =============================================================================


class TransformerBlock(nn.Module):
    """A single transformer block with pre-norm architecture."""

    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()

        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(config, layer_idx)

        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, new_cache = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attentions,)
        if use_cache:
            outputs += (new_cache,)

        return outputs


# =============================================================================
# Full Model
# =============================================================================


class TransformerPreTrainedModel(PreTrainedModel):
    """Base class for transformer models."""

    config_class = TransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class TransformerModel(TransformerPreTrainedModel):
    """Transformer model outputting raw hidden states."""

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def set_input_embeddings(self, value: nn.Embedding):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds

        # Initialize cache list if needed
        if use_cache and past_key_values is None:
            past_key_values = [None] * len(self.layers)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_cache = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    layer_cache,
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=layer_cache,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache.append(layer_outputs[-1])

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class TransformerForCausalLM(TransformerPreTrainedModel):
    """Transformer model with a language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.model = TransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embeddings

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embeddings = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def get_decoder(self) -> TransformerModel:
        return self.model

    def set_decoder(self, decoder: TransformerModel):
        self.model = decoder

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Prepare inputs for generation (handles KV cache)."""
        if past_key_values is not None:
            # Only use the last token if we have cache
            input_ids = input_ids[:, -1:]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "use_cache": True,
        }
