"""Implements pixelCNN as described in the paper:
"Pixel Recurrent Neural Networks https://arxiv.org/abs/1601.06759"
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Sequence


class Conv1x1(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=1,
            strides=1,
            padding="CAUSAL",
            rngs=rngs,
        )

    def __call__(self, x_i_bnh):
        return nnx.relu(jax.vmap(self.conv)(x_i_bnh))


class ResidualConvBlock(nnx.Module):
    """Residual Convolutional Block for PixelCNN."""
    def __init__(self,
                 features,
                 rngs: nnx.Rngs):
        self.in_conv = Conv1x1(features*2, features, rngs)

        center = 1 # as kernel is of shape (3, 3)
        # todo: support convolutional masking for rgb images.
        # kernel cannot look at pixels in future to the center pixel.
        mask_b = jnp.ones((3, 3, features, features))
        # center pixel can attend to itself for mask type B.
        mask_b = mask_b.at[center+1:, :, :, :].set(0)
        mask_b = mask_b.at[center, center+1:, :, :].set(0)
        mask_b = nnx.Variable(mask_b)

        self.conv = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(3, 3),
            padding='SAME',
            mask=mask_b,
            rngs=rngs
        )

        self.out_conv = Conv1x1(features, features * 2, rngs)

    def __call__(self, x_bmn2h):
        x_bmnh = nnx.relu(jax.vmap(self.in_conv)(x_bmn2h))
        x_bmnh = nnx.relu(jax.vmap(self.conv)(x_bmnh))
        x_bmn2h_out = nnx.relu(jax.vmap(self.out_conv)(x_bmnh))
        return x_bmn2h + x_bmn2h_out


class PixelCNN(nnx.Module):
    def __init__(self,
                 features: int,
                 num_layers: int,
                 is_rgb: bool,
                 output_conv_out_channels: Sequence[int],
                 preds_dim: int,
                 rngs: nnx.Rngs):
        self.features = features
        self.num_layers = num_layers
        image_channels = 3 if is_rgb else 1
        self.preds_dim = preds_dim

        mask_a = jnp.ones((7, 7, image_channels, features * 2))
        mask_a = mask_a.at[3:, 4:, :, :].set(0)
        mask_a = mask_a.at[4:].set(0)
        mask_a = mask_a.at[3, 3].set(0)
        mask_a = nnx.Variable(mask_a)
        # todo: support convolutional masking for rgb images.
        self.input_conv = nnx.Conv(
            in_features=image_channels,
            out_features=features * 2,
            kernel_size=(7, 7),
            strides=1,
            padding="SAME",
            mask=mask_a,
            rngs=rngs,
        )

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(ResidualConvBlock(features, rngs))

        self.output_convs = [
            Conv1x1(features * 2, output_conv_out_channels[0], rngs)
        ] + [
            Conv1x1(output_conv_out_channels[i - 1], output_conv_out_channels[i], rngs)
            for i in range(1, len(output_conv_out_channels))
        ]

        if is_rgb:
            # todo: properly support rgb images according to the paper.
            self.head = nnx.Linear(output_conv_out_channels[-1], 256, rngs=rngs)
        else:
            self.head = nnx.Linear(output_conv_out_channels[-1], preds_dim, rngs=rngs)

    def __call__(self, im_bmnc):
        x_bmn2h = jax.vmap(self.input_conv)(im_bmnc)

        for residual_block in self.layers:
            x_bmn2h = residual_block(x_bmn2h)

        x = x_bmn2h

        for output_conv in self.output_convs:
            x = jax.vmap(output_conv)(x)

        image_logits = self.head(x)
        return image_logits

    def generate(self, im_height: int, im_width: int, batch_size: int, key: int):
        b = batch_size
        m = im_height
        n = im_width
        image_batch = jnp.zeros((b, m, n, 1), dtype=jnp.float32)

        @nnx.scan
        def gen_pixels(carry, x):
            image_batch, key = carry
            i, j = x // n, x % n
            image_logits = self(image_batch)
            if self.preds_dim == 1:
                probs = nnx.sigmoid(image_logits)
                sampled_pixels = jax.random.bernoulli(key, probs[:, i, j])
            else:
                sampled_pixels = jax.random.categorical(key, image_logits[:, i, j])
            image_batch = image_batch.at[:, i, j].set(sampled_pixels)
            key, _ = jax.random.split(key)
            return (image_batch, key), None

        carry = (image_batch, key)
        xs = jnp.arange(m * n)
        final_carry, _ = gen_pixels(carry, xs)

        image_batch = final_carry[0]
        return image_batch
