import haiku as hk
import jax.numpy as jnp
import pytest
from jax import random

from .. import model_config
from attention_jax.utils import build_forward_fn


@pytest.fixture
def main_rng():
    return random.PRNGKey(42)


def test_forward(main_rng):
    main_rng, x_rng = random.split(main_rng)
    batch_size = 16
    x = random.choice(
        x_rng,
        jnp.arange(model_config.vocab_size),
        shape=(batch_size, model_config.seq_len),
    )
    main_rng, x_rng = random.split(main_rng)

    forward_fn = build_forward_fn(model_config)
    forward_fn = hk.transform(forward_fn)

    params = forward_fn.init(x_rng, inputs=x, training=False)
    main_rng, x_rng = random.split(main_rng)

    output = forward_fn.apply(params=params, inputs=x, rng=main_rng, training=False)

    assert output.shape == (batch_size, model_config.seq_len, model_config.vocab_size)
