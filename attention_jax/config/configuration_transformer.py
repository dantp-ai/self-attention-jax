import dataclasses


@dataclasses.dataclass
class TransformerConfig:
    vocab_size: int = 27
    n_embed: int = 32
    n_attention_heads: int = 4
    n_transformer_block: int = 1
    pad_token_id: int = 0
    dropout_rate: float = 0.0
    seq_len: int = 15

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
