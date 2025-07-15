"""Implements the pixelRNN variant called RowLSTM as described in the paper:
"Pixel Recurrent Neural Networks https://arxiv.org/abs/1601.06759"
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Sequence


class RowLSTMCell(nnx.Module):
    def __init__(
        self,
        is_kernel_length: int,
        ss_kernel_length: int,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
    ):
        center = is_kernel_length // 2
        mask_b = jnp.ones((is_kernel_length, in_features, 4 * out_features))
        # center pixel can attend to itself for mask type B.
        mask_b = nnx.Variable(mask_b.at[center + 1 :, :, :].set(0))
        # todo: support convolutional masking for rgb images.
        self.is_kernel = nnx.Conv(
            in_features=in_features,
            out_features=4 * out_features,
            kernel_size=is_kernel_length,
            strides=1,
            padding="CAUSAL",
            mask=mask_b,
            rngs=rngs,
        )
        self.ss_kernel = nnx.Conv(
            in_features=out_features,
            out_features=4 * out_features,
            kernel_size=ss_kernel_length,
            strides=1,
            padding="CAUSAL",
            rngs=rngs,
        )
        self.output_1x1_conv = nnx.Conv(
            in_features=out_features,
            out_features=in_features,
            kernel_size=1,
            strides=1,
            padding="CAUSAL",
            rngs=rngs,
        )

    def __call__(self, x_i_bnh, h_im1_bnh, c_im1_bnh):
        # computes state for an entire image row at once.
        ss_component = jax.vmap(self.ss_kernel)(h_im1_bnh)
        is_component = jax.vmap(self.is_kernel)(x_i_bnh)
        z_bn4h = ss_component + is_component
        f, i, o, g = jnp.split(z_bn4h, 4, axis=-1)

        forget_gate = nnx.sigmoid(f)
        input_gate = nnx.sigmoid(i)
        output_gate = nnx.sigmoid(o)
        cell_gate = nnx.tanh(g)

        c_i = forget_gate * c_im1_bnh + input_gate * cell_gate
        h_i = output_gate * nnx.tanh(c_i)

        # residual connection
        h_i_out = jax.vmap(self.output_1x1_conv)(h_i) + x_i_bnh

        return h_i, c_i, h_i_out


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


class RowLSTM(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        is_rgb: bool = False,
        hidden_dim: int = 128,
        is_kernel_length: int = 3,
        ss_kernel_length: int = 3,
        num_layers: int = 1,
        enable_skip_connections: bool = False,
        output_conv_out_channels: Sequence[int] = [32, 32],
        preds_dim: int = 256
    ):
        image_channels = 3 if is_rgb else 1
        self.image_channels = image_channels
        self.hidden_dim = hidden_dim
        self.enable_skip_connections = enable_skip_connections
        self.preds_dim = preds_dim

        mask_a = jnp.ones((7, 7, image_channels, hidden_dim * 2))
        mask_a = mask_a.at[3:, 4:, :, :].set(0)
        mask_a = mask_a.at[4:].set(0)
        mask_a = mask_a.at[3, 3].set(0)
        mask_a = nnx.Variable(mask_a)
        # todo: support convolutional masking for rgb images.
        self.input_conv = nnx.Conv(
            in_features=image_channels,
            out_features=hidden_dim * 2,
            kernel_size=(7, 7),
            strides=1,
            padding="SAME",
            mask=mask_a,
            rngs=rngs,
        )

        self.layers = []
        self.num_layers = num_layers
        for _ in range(num_layers):
            layer = RowLSTMCell(
                is_kernel_length,
                ss_kernel_length,
                in_features=hidden_dim * 2,
                out_features=hidden_dim,
                rngs=rngs,
            )
            self.layers.append(layer)

        self.h_init = nnx.Param(
            nnx.initializers.glorot_normal()(rngs.params(), (num_layers, hidden_dim))
        )

        self.output_convs = [
            Conv1x1(hidden_dim * 2, output_conv_out_channels[0], rngs)
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
        b, m, n, c = im_bmnc.shape

        L = self.num_layers
        # h_prev_row = jnp.zeros((L, b, n, self.hidden_dim))
        h_prev_row = jnp.tile(self.h_init.value[:,jnp.newaxis,jnp.newaxis,:], (1, b, n, 1))
        c_prev_row = jnp.zeros((L, b, n, self.hidden_dim))

        x_bmnh = jax.vmap(self.input_conv)(im_bmnc)
        image_logits = []
        for i in range(m):
            layer_in = x_bmnh
            h_curr_row = []
            c_curr_row = []
            h_curr_row_layer, c_curr_row_layer, layer_out = self.layers[0](
                    layer_in[:, i], h_prev_row[0], c_prev_row[0]
                )
            h_curr_row.append(h_curr_row_layer)
            c_curr_row.append(c_curr_row_layer)
            layer_in = layer_out

            final_layer_out = jnp.zeros((b, n, self.hidden_dim * 2))
            if self.enable_skip_connections:
                final_layer_out += layer_out
            else:
                final_layer_out = layer_out

            for layer_idx in range(1, L):
                h_curr_row_layer, c_curr_row_layer, layer_out = self.layers[layer_idx](
                    layer_in, h_prev_row[layer_idx], c_prev_row[layer_idx]
                )
                h_curr_row.append(h_curr_row_layer)
                c_curr_row.append(c_curr_row_layer)
                layer_in = layer_out
                if self.enable_skip_connections:
                    final_layer_out += layer_out
                else:
                    final_layer_out = layer_out

            h_prev_row, c_prev_row = jnp.array(h_curr_row), jnp.array(c_curr_row)

            layer_out = final_layer_out
            for out_conv in self.output_convs:
                layer_out = out_conv(layer_out)

            row_logits = self.head(layer_out)
            image_logits.append(row_logits)

        return jnp.array(image_logits).transpose(1, 0, 2, 3)

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
