import math

import chex
import haiku as hk
import jax.numpy as jnp
from jax import nn


def layer_norm(x, name):
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)


class Head(hk.Module):
    def __init__(self, head_size, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.head_size = head_size
        self.key = hk.Linear(head_size, with_bias=False)
        self.query = hk.Linear(head_size, with_bias=False)
        self.value = hk.Linear(head_size, with_bias=False)

    def __call__(self, x, mask, training=False):

        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        chex.assert_rank([k, q], 3)

        assert k.shape == (B, T, self.head_size)

        attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
        attn_logits = attn_logits / math.sqrt(C)

        assert attn_logits.shape == (B, T, T)

        mask = mask[:, None, :]
        causal_mask = jnp.tril(jnp.ones((1, T, T)))
        mask = mask * causal_mask

        attn_logits = jnp.where(mask, attn_logits, -1e30)

        attention = nn.softmax(attn_logits, axis=2)

        v = self.value(x)
        out = jnp.matmul(attention, v)

        if training:
            attention = hk.dropout(hk.next_rng_key(), self.dropout_rate, attention)

        return out


class MultiHeadAttention(hk.Module):
    def __init__(self, num_heads, head_size, n_embed, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.heads = [Head(head_size, dropout_rate) for _ in range(num_heads)]
        self.proj = hk.Linear(n_embed)

    def __call__(self, x, mask, training=False):
        out = jnp.concatenate([h(x, mask, training=training) for h in self.heads], 2)
        out = self.proj(out)

        if training:
            out = hk.dropout(hk.next_rng_key(), self.dropout_rate, out)

        return out


class MLP(hk.Module):
    def __init__(self, n_embed, init_scale, widening_factor, dropout_rate, name=""):
        super().__init__()
        self.name = name
        self.dropout_rate = dropout_rate
        self._init_scale = init_scale
        self._widening_factor = widening_factor
        initializers = hk.initializers.VarianceScaling(self._init_scale)
        self.ln1 = hk.Linear(self._widening_factor * n_embed, w_init=initializers)
        self.ln2 = hk.Linear(n_embed, w_init=initializers)

    def __call__(self, x, training=False):
        x = self.ln1(x)
        x = nn.gelu(x)
        x = self.ln2(x)

        if training:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)

        return x


class Block(hk.Module):
    def __init__(
        self, n_embed, n_head, dropout_rate, init_scale, widening_factor, name=""
    ):
        super().__init__()
        self.name = name
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embed, dropout_rate)
        self.mlp = MLP(
            n_embed, init_scale, widening_factor, dropout_rate, name=f"{self.name}_mlp"
        )

    def __call__(self, x, mask, training=False):
        x_norm = layer_norm(x, name=f"{self.name}_ln_1")
        x_attn = self.sa(x_norm, mask, training)
        x = x + x_attn
        x_norm = layer_norm(x, name=f"{self.name}_ln_2")
        x_mlp = self.mlp(x_norm, training)
        x = x + x_mlp

        return x


class Transformer(hk.Module):
    def __init__(self, cfg):
        super().__init__(name="Transformer")
        self.pad_token_id = cfg.pad_token_id
        self.token_embedding_table = hk.Embed(cfg.vocab_size, cfg.n_embed)
        self.position_embedding_table = hk.Embed(cfg.seq_len, cfg.n_embed)
        init_scale = 2.0 / cfg.n_transformer_block
        widening_factor = 4
        self.blocks = [
            Block(
                cfg.n_embed,
                cfg.n_attention_heads,
                cfg.dropout_rate,
                init_scale,
                widening_factor,
                name=f"block_{i}",
            )
            for i in range(cfg.n_transformer_block)
        ]
        self.head = hk.Linear(cfg.vocab_size)
        self.seq_len = cfg.seq_len
        self.vocab_size = cfg.vocab_size
        self.n_embed = cfg.n_embed

    def __call__(self, inputs, training=False):
        dims = chex.Dimensions(B=inputs.shape[0], T=self.seq_len)
        chex.assert_shape(inputs, dims["BT"])

        input_mask = jnp.greater(inputs, self.pad_token_id)

        chex.assert_equal_shape([input_mask, inputs])

        token_emb = self.token_embedding_table(inputs)
        dims = chex.Dimensions(B=token_emb.shape[0], T=self.seq_len, C=self.n_embed)

        chex.assert_shape(token_emb, dims["BTC"])

        pos_emb = self.position_embedding_table(jnp.arange(dims.T))
        x = token_emb + pos_emb

        chex.assert_shape(x, dims["BTC"])

        for block in self.blocks:
            x = block(x, input_mask, training)
        x = layer_norm(x, "ln_atn")

        chex.assert_shape(x, dims["BTC"])

        logits = self.head(x)

        dims = chex.Dimensions(B=logits.shape[0], T=self.seq_len, V=self.vocab_size)
        chex.assert_shape(logits, dims["BTV"])

        return logits
