from transformers.configuration_utils import PretrainedConfig


class TransformerConfig(PretrainedConfig):
    """
    Configuration class for a standard transformer language model.

    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Dimension of the hidden representations
        num_hidden_layers: Number of transformer blocks
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA). None = same as num_heads (MHA)
        qkv_bias: Whether to use bias in QKV projections
        qk_norm: Whether to apply RMSNorm to Q and K
        window_size: Sliding window size for attention. None = full attention
        rope_theta: Base for rotary position embeddings
        max_position_embeddings: Maximum sequence length
        hidden_ratio: MLP hidden dimension ratio (hidden_size * hidden_ratio)
        intermediate_size: Explicit MLP hidden dimension (overrides hidden_ratio)
        hidden_act: Activation function ("swish", "gelu", "relu")
        initializer_range: Std for weight initialization
        norm_eps: Epsilon for layer normalization
        use_cache: Whether to return key/value states for generation
        tie_word_embeddings: Whether to tie input/output embeddings
    """

    model_type = "transformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
