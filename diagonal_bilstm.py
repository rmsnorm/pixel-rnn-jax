"""Implements the pixelRNN variant called Diagonal-BiLSTM as described in the paper:
"Pixel Recurrent Neural Networks https://arxiv.org/abs/1601.06759"
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Sequence
import math


def skew_feature_map(im_bmnc):
    b, m, n, c = im_bmnc.shape
    buffer = jnp.zeros((b, m, m + n - 1, c))
    for i in range(m):
        buffer = buffer.at[:, i, i : i + n].set(im_bmnc[:, i])
    return buffer


def unskew_feature_map(im_bm2nc):
    b, m, n2, c = im_bm2nc.shape
    n = n2 - m + 1
    im = jnp.zeros((b, m, n, c))
    for i in range(m):
        im = im.at[:, i].set(im_bm2nc[:, i, i : i + n])
    return im


def create_mask(
    kernel_shape, input_channels, output_channels, image_channels, mask_type
):
    mask = jnp.ones(kernel_shape + (input_channels, output_channels))
    center_h = kernel_shape[0] // 2
    center_w = kernel_shape[1] // 2

    # set rows below center to 0
    mask = mask.at[center_h + 1 :, :, :, :].set(0)

    # set columns on center row after center pixel to 0
    mask = mask.at[center_h, center_w + 1 :, :, :].set(0)

    for i in range(image_channels):
        for j in range(image_channels):
            if (mask_type == "mask_a" and j >= i) or (mask_type == "mask_b" and j > i):
                mask = mask.at[
                    center_h, center_w, j::image_channels, i::image_channels
                ].set(0)
    return mask


class DiagonalLSTMCell(nnx.Module):
    def __init__(self, features: int, image_channels: int, rngs: nnx.Rngs):
        # todo: support conv masking for rgb images
        mask_b = create_mask((1, 1), 2 * features, features, image_channels, "mask_b")
        mask_b = nnx.Variable(mask_b)
        # is_kernels is a stack of 4 kernels instead of 1 kernel with 4*features
        # to maintain the offsets in convolutional masking correctly. Otherwise,
        # it leads to inter-channel leakage.
        self.is_kernels = [
            nnx.Conv(
                in_features=2 * features,
                out_features=features,
                kernel_size=(1, 1),
                rngs=rngs,
                kernel_init=nnx.initializers.he_uniform(),
                mask=mask_b,
            )
            for _ in range(4)
        ]

        self.ss_kernel = nnx.Conv(
            in_features=features,
            out_features=4 * features,
            kernel_size=(1, 2),
            rngs=rngs,
            kernel_init=nnx.initializers.he_uniform(),
            padding=((0, 0), (1, 0)),  # Only pad left, not right
        )

        output_mask_b = nnx.Variable(
            create_mask((1, 1), features, 2 * features, image_channels, "mask_b")
        )
        self.output_1x1_conv = nnx.Conv(
            in_features=features,
            out_features=2 * features,
            kernel_size=(1, 1),
            rngs=rngs,
            kernel_init=nnx.initializers.he_uniform(),
            mask=output_mask_b,
        )

    def __call__(self, x_b1mh, prev_h_b1mh, prev_c_b1mh):
        # is called once per diagonal, i.e. once per column after skewing the inputs
        b, _, m, h = x_b1mh.shape
        i2s_4b1mh = [kernel(x_b1mh) for kernel in self.is_kernels]
        i2s_b1m4h = (
            jnp.stack(i2s_4b1mh).transpose(1, 2, 3, 0, 4).reshape(b, 1, m, 2 * h)
        )
        s2s_b1m4h = self.ss_kernel(prev_h_b1mh)
        z_b1m4h = i2s_b1m4h + s2s_b1m4h

        f, i, o, g = jnp.split(z_b1m4h, 4, axis=-1)

        forget_gate = nnx.sigmoid(f)
        input_gate = nnx.sigmoid(i)
        output_gate = nnx.sigmoid(o)
        cell_gate = nnx.tanh(g)

        c_b1mh = forget_gate * prev_c_b1mh + input_gate * cell_gate
        h_b1mh = output_gate * nnx.tanh(c_b1mh)

        # residual connection
        x_out = self.output_1x1_conv(h_b1mh) + x_b1mh

        return (h_b1mh, c_b1mh), x_out[:, 0, :, :]


class DiagonalLSTMLayer(nnx.Module):
    def __init__(self, features: int, image_channels: int, rngs: nnx.Rngs):
        self.features = features
        self.diagonal_lstm = DiagonalLSTMCell(features, image_channels, rngs)
        self.h0 = nnx.Param(
            nnx.initializers.glorot_normal()(rngs.params(), (1, features))
        )

    def __call__(self, im_bm2nh: jax.Array):
        b, m, n2, h = im_bm2nh.shape

        @nnx.scan
        def scan_diagonals(carry, x):
            h_b1mh, c_b1mh = carry

            x_b1m2h = x[:, jnp.newaxis, :, :]
            carry, x_out = self.diagonal_lstm(x_b1m2h, h_b1mh, c_b1mh)
            return carry, x_out

        h0 = jnp.tile(self.h0.value[jnp.newaxis, jnp.newaxis, ...], (b, 1, m, 1))
        c0 = jnp.zeros((b, 1, m, self.features))

        im_2nbmh = im_bm2nh.transpose(2, 0, 1, 3)
        carry = (h0, c0)

        _, out = scan_diagonals(carry, im_2nbmh)
        x_bm2nh_out = out.transpose(1, 2, 0, 3)
        return x_bm2nh_out


class DiagonalBiLSTMLayer(nnx.Module):
    def __init__(self, features: int, image_channels: int, rngs: nnx.Rngs):
        self.left_layer = DiagonalLSTMLayer(features, image_channels, rngs)
        self.right_layer = DiagonalLSTMLayer(features, image_channels, rngs)

    def __call__(self, im_bm2nh: jax.Array):
        b, m, n2, h = im_bm2nh.shape
        x_bm2nh_left = self.left_layer(im_bm2nh)
        x_bm2nh_right = self.right_layer(im_bm2nh[:, :, ::-1, :])[:, :, ::-1, :]

        x_bm2nh_right = jnp.concat(
            (jnp.zeros((b, 1, n2, h)), x_bm2nh_right[:, :-1, :, :]), axis=1
        )

        return x_bm2nh_left + x_bm2nh_right


class DiagonalBiLSTM(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        features: int = 16,
        num_layers: int = 1,
        enable_skip_connections: bool = False,
        output_conv_out_channels: Sequence[int] = [32, 32],
        is_rgb: bool = False,
        preds_dim: int = 256,
    ):
        image_channels = 3 if is_rgb else 1
        self.image_channels = image_channels
        self.features = features
        self.enable_skip_connections = enable_skip_connections
        self.preds_dim = preds_dim

        mask_a = create_mask(
            kernel_shape=(7, 7),
            input_channels=image_channels,
            output_channels=features * 2,
            image_channels=image_channels,
            mask_type="mask_a",
        )
        mask_a = nnx.Variable(mask_a)
        self.input_conv = nnx.Conv(
            in_features=image_channels,
            out_features=features * 2,
            kernel_size=(7, 7),
            strides=1,
            padding="SAME",
            mask=mask_a,
            rngs=rngs,
            kernel_init=nnx.initializers.he_uniform(),
        )

        self.layers = []
        self.num_layers = num_layers
        for _ in range(num_layers):
            self.layers.append(
                DiagonalBiLSTMLayer(
                    features,
                    image_channels,
                    rngs=rngs,
                )
            )

        self.output_convs = [
            nnx.Conv(
                in_features=2 * features,
                out_features=output_conv_out_channels[0],
                kernel_size=(1, 1),
                rngs=rngs,
                kernel_init=nnx.initializers.he_uniform(),
                mask=nnx.Variable(
                    create_mask(
                        (1, 1),
                        2 * features,
                        output_conv_out_channels[0],
                        image_channels,
                        "mask_b",
                    )
                ),
            )
        ] + [
            nnx.Conv(
                in_features=output_conv_out_channels[i - 1],
                out_features=output_conv_out_channels[i],
                kernel_size=(1, 1),
                rngs=rngs,
                kernel_init=nnx.initializers.he_uniform(),
                mask=nnx.Variable(
                    create_mask(
                        (1, 1),
                        output_conv_out_channels[i - 1],
                        output_conv_out_channels[i],
                        image_channels,
                        "mask_b",
                    ),
                ),
            )
            for i in range(1, len(output_conv_out_channels))
        ]

        if is_rgb:
            s = jnp.arange(output_conv_out_channels[-1])
            r_in = s[0::3].shape[0]
            g_in = s[1::3].shape[0]
            b_in = s[2::3].shape[0]
            self.r_head = nnx.Linear(r_in, 256, rngs=rngs)
            self.g_head = nnx.Linear(g_in, 256, rngs=rngs)
            self.b_head = nnx.Linear(b_in, 256, rngs=rngs)
        else:
            self.head = nnx.Linear(output_conv_out_channels[-1], preds_dim, rngs=rngs)

    def compute_shared_features(self, im_bmnc: jax.Array):
        if self.image_channels == 3:
            im_bmnc = (im_bmnc - 128.0) / 256.0

        x_bmnh = self.input_conv(im_bmnc)
        # skew the input map
        x_bm2nh = skew_feature_map(x_bmnh)

        if self.enable_skip_connections:
            final_out = jnp.zeros(x_bm2nh.shape)
        else:
            final_out = None

        for layer in self.layers:
            x_bm2nh = layer(x_bm2nh)
            if self.enable_skip_connections:
                final_out += x_bm2nh
            else:
                final_out = x_bm2nh

        x_bmnh = unskew_feature_map(final_out)
        x = x_bmnh

        for conv in self.output_convs:
            x = nnx.relu(conv(x))
        return x

    def __call__(self, im_bmnc: jax.Array):
        x = self.compute_shared_features(im_bmnc)

        if self.image_channels == 3:
            r_logits = self.r_head(x[:, :, :, 0::3])
            g_logits = self.g_head(x[:, :, :, 1::3])
            b_logits = self.b_head(x[:, :, :, 2::3])
            logits = jnp.concatenate(
                [
                    r_logits[:, :, :, jnp.newaxis, :],
                    g_logits[:, :, :, jnp.newaxis, :],
                    b_logits[:, :, :, jnp.newaxis, :],
                ],
                axis=3,
            )
        else:
            logits = self.head(x)
        return logits

    def generate(self, im_height: int, im_width: int, batch_size: int, key: int):
        b = batch_size
        m = im_height
        n = im_width
        if self.image_channels == 3:
            image_batch = jnp.zeros((b, m, n, 3), dtype=jnp.int32)
        else:
            image_batch = jnp.zeros((b, m, n, self.image_channels), dtype=jnp.float32)

        @nnx.scan
        def gen_pixels(carry, x):
            image_batch, key = carry
            pixel_idx = x // self.image_channels
            i, j = pixel_idx // n, pixel_idx % n
            channel = x % self.image_channels

            image_batch_logits = self(image_batch)

            if self.preds_dim == 1:
                probs = nnx.sigmoid(image_batch_logits[:, i, j])
                sampled_pixels = jax.random.bernoulli(key, probs)
                image_batch = image_batch.at[:, i, j].set(sampled_pixels)
            else:
                channel_logits = image_batch_logits[:, i, j, channel]
                # sampled_pixels = jax.random.categorical(key, channel_logits)
                sampled_pixels = jnp.argmax(channel_logits, axis=-1)
                image_batch = image_batch.at[:, i, j, channel].set(sampled_pixels)

            key, _ = jax.random.split(key)
            return (image_batch, key), None

        carry = (image_batch, key)
        xs = jnp.arange(m * n * self.image_channels)
        final_carry, _ = gen_pixels(carry, xs)
        image_batch = final_carry[0]
        return image_batch
