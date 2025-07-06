"""Implements a spatial LSTM cell as described in the paper:
"Generative Image Modeling using Spatial LSTMs"
"https://arxiv.org/abs/1506.03478."
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

# Einsum notation shorthand
# H: input_dim + hidden_dim
# h: hidden_dim
# b: batch
# d: input_dim
# m: image height (pixel rows)
# n: image width (pixel cols)
# v: grayscale range [0, 256)

class SpatialLSTMCell(nnx.Module):
    """Spatial LSTM Cell"""
    def __init__(self,
                 hidden_dim: int,
                 rngs: nnx.Rngs):
        input_dim = 256
        H = input_dim + 2 * hidden_dim
        self.gate = nnx.Einsum('bH, Hh -> bh',
                            kernel_shape=(H, 5 * hidden_dim),
                            bias_shape=5 * hidden_dim,
                            param_dtype=jnp.float32,
                            rngs=rngs)
        self.ln = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self,
                 x_bd,
                 h_i_jm1,
                 h_im1_j,
                 c_i_jm1,
                 c_im1_j):
        x_bH = jnp.column_stack([x_bd, h_i_jm1, h_im1_j])
        gate_out = self.gate(x_bH)

        f_c, f_r, i, o, g = jnp.split(gate_out, 5, axis=-1)

        forget_col_gate = nnx.sigmoid(f_c)
        forget_row_gate = nnx.sigmoid(f_r)
        input_gate = nnx.sigmoid(i)
        output_gate = nnx.sigmoid(o)
        cell_gate = nnx.tanh(g)

        # update cell state
        cell_bh = forget_col_gate * c_i_jm1 + forget_row_gate * c_im1_j + \
            input_gate * cell_gate

        # update hidden state
        state_bh = output_gate * nnx.tanh(self.ln(cell_bh))

        return state_bh, cell_bh

class SpatialLSTM(nnx.Module):
    def __init__(self,
                 hidden_dim: int,
                 rngs: nnx.Rngs):
        self.input_dim = 256
        self.hidden_dim = hidden_dim
        self.lstm = SpatialLSTMCell(hidden_dim, rngs)
        self.head = nnx.Einsum('bh, hv -> bv',
                               kernel_shape=(hidden_dim, 256),
                               bias_shape=256,
                               param_dtype=jnp.float32,
                               rngs=rngs)
        self.h_init = nnx.Param(nnx.initializers.glorot_normal()(rngs.params(), (hidden_dim, 1)))

    # only supports gray-scale images
    def __call__(self, x_bmnd):
        assert x_bmnd.shape[-1] == 256
        d = 256
        m = x_bmnd.shape[1]
        n = x_bmnd.shape[2]
        b = x_bmnd.shape[0]
        h = self.hidden_dim

        c_init = jnp.zeros((h, 1))
        prev_row_h_nbh = jnp.tile(self.h_init.value[:, jnp.newaxis],
                                  (1, n, b)).transpose(1, 2, 0)
        prev_row_c_nbh = jnp.tile(c_init[:, jnp.newaxis],
                                  (1, n, b)).transpose(1, 2, 0)

        batch_image_logits = jnp.zeros(shape=(b,m,n,d), dtype=jnp.float32)

        @nnx.scan
        def _scan_col(carry, x):
            h_i_jm1, c_i_jm1 = carry
            x_bd = x[:, :d]
            h_im1_j = x[:, d:d+h]
            c_im1_j = x[:, d+h:d+2*h]

            h_ij, c_ij = self.lstm(x_bd, h_i_jm1, h_im1_j, c_i_jm1, c_im1_j)
            logits = self.head(h_ij)

            return (h_ij, c_ij), (h_ij, c_ij, logits)

        for i in range(m):
            x_nbd = x_bmnd[:,i].transpose(1, 0, 2)
            xs = jnp.concatenate((x_nbd, prev_row_h_nbh, prev_row_c_nbh), axis=-1)
            carry = (jnp.tile(self.h_init.value, (1, b)).T, jnp.tile(c_init, (1, b)).T)
            _, out = _scan_col(carry, xs)
            prev_row_h_nbh = out[0]
            prev_row_c_nbh = out[1]
            row_logits = out[2].transpose(1, 0, 2)
            batch_image_logits = batch_image_logits.at[:, i].set(row_logits)

        return batch_image_logits

    def generate(self,
                 im_height: int,
                 im_width: int,
                 batch_size: int,
                 key: int):
        m = im_height
        n = im_width
        h = self.hidden_dim
        b = batch_size
        d = 256
        c_init = jnp.zeros((h, 1))
        batch_image = jnp.zeros((b,m,n+1), dtype=jnp.uint8)
        prev_row_h_nbh = jnp.tile(self.h_init.value[:, jnp.newaxis],
                                  (1, n, b)).transpose(1, 2, 0)
        prev_row_c_nbh = jnp.tile(c_init[:, jnp.newaxis],
                                  (1, n, b)).transpose(1, 2, 0)

        @nnx.scan
        def _gen_scan_col(carry, x):
            h_i_jm1, c_i_jm1, x_b, key = carry
            x_bd = nnx.one_hot(x_b, d)
            h_im1_j = x[:, :h]
            c_im1_j = x[:, h:2*h]

            h_ij, c_ij = self.lstm(x_bd, h_i_jm1, h_im1_j, c_i_jm1, c_im1_j)

            logits = self.head(h_ij)
            sampled_x_b = jax.random.categorical(key, logits, axis=-1)
            key, _ = jax.random.split(key)

            return (h_ij, c_ij, sampled_x_b, key), (h_ij, c_ij, sampled_x_b)

        x_init = jnp.zeros(b, dtype=jnp.int32)
        for i in range(m):
            xs = jnp.concatenate((prev_row_h_nbh, prev_row_c_nbh), axis=-1)
            carry = (jnp.tile(self.h_init.value, (1, b)).T,
                     jnp.tile(c_init, (1, b)).T,
                     x_init,
                     key)
            final_carry, out = _gen_scan_col(carry, xs)
            prev_row_h_nbh, prev_row_c_nbh, sampled_row_pixels = out
            key, _ = jax.random.split(final_carry[3])
            batch_image = batch_image.at[:, i, 1:].set(sampled_row_pixels.T)
        return batch_image