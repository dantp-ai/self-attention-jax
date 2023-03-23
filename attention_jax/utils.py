from attention_jax.model import Transformer


def build_forward_fn(model_config):
    """Builds the forward function for the model and returns it as a callable."""

    def forward_fn(inputs, training=False):
        """Forward function for the model."""
        model = Transformer(model_config)
        out = model(inputs, training=training)

        return out

    return forward_fn
