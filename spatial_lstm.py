"""Implements a spatial LSTM cell as described in the paper:
"Generative Image Modeling using Spatial LSTMs"
"https://arxiv.org/abs/1506.03478."
"""

import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial

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

    def __init__(self, input_dim: int, hidden_dim: int, rngs: nnx.Rngs):
        # 4 for causal neighbourhood of x[i,j] -> x[i,j-1], x[i-1,j-1], x[i-1,j] and x[i-1,j+1] 
        self.hidden_dim = hidden_dim
        H = 4 * input_dim + 2 * hidden_dim
        self.gate = nnx.Einsum(
            "bH, Hh -> bh",
            kernel_shape=(H, 5 * hidden_dim),
            bias_shape=5 * hidden_dim,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.ln = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self, x_b4h, h_b2h, c_b2h):
        x_bH = jnp.column_stack([x_b4h, h_b2h])
        gate_out = self.gate(x_bH)

        f_c, f_r, i, o, g = jnp.split(gate_out, 5, axis=-1)

        forget_col_gate = nnx.sigmoid(f_c)
        forget_row_gate = nnx.sigmoid(f_r)
        input_gate = nnx.sigmoid(i)
        output_gate = nnx.sigmoid(o)
        cell_gate = nnx.tanh(g)

        # update cell state
        h = self.hidden_dim
        cell_bh = (
            forget_col_gate * c_b2h[:, :h]
            + forget_row_gate * c_b2h[:, h:]
            + input_gate * cell_gate
        )

        # update hidden state
        state_bh = output_gate * nnx.tanh(self.ln(cell_bh))

        return state_bh, cell_bh

class SpatialLSTM(nnx.Module):
    def __init__(
        self,
        hidden_dim: int,
        rngs: nnx.Rngs,
    ):
        input_dim = 256
        self.hidden_dim = hidden_dim

        self.embedding = nnx.Embed(input_dim, hidden_dim, rngs=rngs)
        self.lstm = SpatialLSTMCell(hidden_dim, hidden_dim, rngs)
        self.h_init = nnx.Param(
            nnx.initializers.glorot_normal()(rngs.params(), (1, hidden_dim))
        )
        self.head = nnx.Linear(hidden_dim, input_dim, rngs=rngs)

    def _get_causal_neighborhood(self, x_bmnh, index):
        _, n = x_bmnh.shape[1], x_bmnh.shape[2]
        i, j = index//n, index%n
        x_i_jm1 = x_bmnh[:, i, j-1]
        x_im1_jm1 = x_bmnh[:, i-1, j-1]
        x_im1_j = x_bmnh[:, i-1, j]
        x_im1_jp1 = x_bmnh[:, i-1, j+1]
        return jnp.concat((x_i_jm1, x_im1_jm1, x_im1_j, x_im1_jp1), axis=-1)

    def get_causal_neighborhood_for_image_batch(self, x_bmnh):
        f = partial(self._get_causal_neighborhood, x_bmnh=x_bmnh)
        b, m, n, h = x_bmnh.shape
        return jax.vmap(f)(index=jnp.arange(m*n)).reshape(b, m, n, 4*h)


    def get_incoming_hidden_states(self, h_bmnh, index):
        _, n = h_bmnh.shape[1], h_bmnh.shape[2]
        i, j = index//n, index%n
        h_i_jm1 = h_bmnh[:,i,j-1]
        h_im1_j = h_bmnh[:,i-1,j]
        return jnp.concat((h_i_jm1, h_im1_j), axis=-1)
    
    def _get_initial_hidden_state_for_image_batch(self, b, m, n):
        h = self.hidden_dim
        h_bmnh = jnp.zeros((b,m+1,n+2,h), dtype=jnp.float32)
        # fill top padding row with h_init
        h_bmnh = h_bmnh.at[:,0].set(jnp.tile(self.h_init.value, (n+2, 1)))
        # fill left padding column with h_init
        h_bmnh = h_bmnh.at[:,:,0].set(jnp.tile(self.h_init.value, (m+1, 1)))
        # fill right padding column with h_init
        h_bmnh = h_bmnh.at[:,:,-1].set(jnp.tile(self.h_init.value, (m+1, 1)))
        return h_bmnh

    def __call__(self, x_bmn):
        b, m, n = x_bmn.shape
        h = self.hidden_dim
        x_bmn = jnp.pad(x_bmn, pad_width=((0, 0), (1, 0), (1, 1)), mode='constant', constant_values=0)
        x_bmnh = self.embedding(x_bmn)
        h_bmnh = self._get_initial_hidden_state_for_image_batch(b,m,n)
        c_bmnh = jnp.zeros((b,m+1,n+2,h), dtype=jnp.float32)

        @nnx.scan
        def _scan_pixels(carry, x):
            x_bmnh, h_bmnh, c_bmnh = carry
            pixel_index = x

            i, j = pixel_index // (n+2), pixel_index % (n+2)

            neighbor_inputs_4h = self._get_causal_neighborhood(x_bmnh, pixel_index)
            incoming_h_2h = self.get_incoming_hidden_states(h_bmnh, pixel_index)
            incoming_c_2h = self.get_incoming_hidden_states(c_bmnh, pixel_index)

            h_bh, c_bh = self.lstm(neighbor_inputs_4h, incoming_h_2h, incoming_c_2h)

            h_bmnh = h_bmnh.at[:,i,j].set(h_bh)
            c_bmnh = c_bmnh.at[:,i,j].set(c_bh)
            return (x_bmnh, h_bmnh, c_bmnh), h_bh

        carry = (x_bmnh, h_bmnh, c_bmnh)
        xs = []
        for i in range(1, m+1):
            for j in range(1, n+1):
                xs.append(i * (n+2) + j)
        xs = jnp.array(xs)
        _, h_sbh = _scan_pixels(carry, xs)
        h_bmnh = h_sbh.reshape((b, m, n, h))

        logits = self.head(h_bmnh)
        return logits

    def generate(self, im_height: int, im_width: int, batch_size: int, key: int):
        b = batch_size
        m = im_height
        n = im_width
        h = self.hidden_dim

        x_bmn = jnp.zeros((b, m+1, n+2), dtype=jnp.uint8)
        x_bmnh = self.embedding(x_bmn)
        h_bmnh = self._get_initial_hidden_state_for_image_batch(b,m,n)
        c_bmnh = jnp.zeros((b,m+1,n+2,h), dtype=jnp.float32)

        @nnx.scan
        def _scan_pixels(carry, x):
            x_bmnh, h_bmnh, c_bmnh, key = carry
            pixel_index = x

            i, j = pixel_index // (n+2), pixel_index % (n+2)

            neighbor_inputs_4h = self._get_causal_neighborhood(x_bmnh, pixel_index)
            incoming_h_2h = self.get_incoming_hidden_states(h_bmnh, pixel_index)
            incoming_c_2h = self.get_incoming_hidden_states(c_bmnh, pixel_index)

            h_bh, c_bh = self.lstm(neighbor_inputs_4h, incoming_h_2h, incoming_c_2h)

            h_bmnh = h_bmnh.at[:,i,j].set(h_bh)
            c_bmnh = c_bmnh.at[:,i,j].set(c_bh)

            logits = self.head(h_bh)
            sampled_x_b = jax.random.categorical(key, logits, axis=-1)

            sampled_x_bh = self.embedding(sampled_x_b)
            x_bmnh = x_bmnh.at[:,i,j].set(sampled_x_bh)
            key, _ = jax.random.split(key)

            return (x_bmnh, h_bmnh, c_bmnh, key), sampled_x_b

        carry = (x_bmnh, h_bmnh, c_bmnh, key)
        xs = []
        for i in range(1, m+1):
            for j in range(1, n+1):
                xs.append(i * (n+2) + j)
        xs = jnp.array(xs)
        _, sampled_x_sb = _scan_pixels(carry, xs)

        sampled_image_batch = sampled_x_sb.transpose(1, 0).reshape((b, m, n))
        return sampled_image_batch
    
    def complete_image(self, image, coord, b, key: int):
        """
        coord: position of first pixel in the image that needs to be in-painted.
        """
        m = image.shape[0]
        n = image.shape[1]
        h = self.hidden_dim

        x_bmn = jnp.zeros((b, m+1, n+2), dtype=jnp.uint8)
        x_bmn = x_bmn.at[:,1:,1:-1].set(
            jnp.tile(image[jnp.newaxis, :, :], (b, 1, 1))
            )
        x_bmnh = self.embedding(x_bmn)
        h_bmnh = self._get_initial_hidden_state_for_image_batch(b,m,n)
        c_bmnh = jnp.zeros((b,m+1,n+2,h), dtype=jnp.float32)

        @nnx.scan
        def eval_scan_pixels(carry, x):
            x_bmnh, h_bmnh, c_bmnh = carry
            pixel_index = x

            i, j = pixel_index // (n+2), pixel_index % (n+2)

            neighbor_inputs_4h = self._get_causal_neighborhood(x_bmnh, pixel_index)
            incoming_h_2h = self.get_incoming_hidden_states(h_bmnh, pixel_index)
            incoming_c_2h = self.get_incoming_hidden_states(c_bmnh, pixel_index)

            h_bh, c_bh = self.lstm(neighbor_inputs_4h, incoming_h_2h, incoming_c_2h)

            h_bmnh = h_bmnh.at[:,i,j].set(h_bh)
            c_bmnh = c_bmnh.at[:,i,j].set(c_bh)
            return (x_bmnh, h_bmnh, c_bmnh), h_bh

        @nnx.scan
        def gen_scan_pixels(carry, x):
            x_bmnh, h_bmnh, c_bmnh, x_bmn, key = carry
            pixel_index = x

            i, j = pixel_index // (n+2), pixel_index % (n+2)

            neighbor_inputs_4h = self._get_causal_neighborhood(x_bmnh, pixel_index)
            incoming_h_2h = self.get_incoming_hidden_states(h_bmnh, pixel_index)
            incoming_c_2h = self.get_incoming_hidden_states(c_bmnh, pixel_index)

            h_bh, c_bh = self.lstm(neighbor_inputs_4h, incoming_h_2h, incoming_c_2h)

            h_bmnh = h_bmnh.at[:,i,j].set(h_bh)
            c_bmnh = c_bmnh.at[:,i,j].set(c_bh)

            logits = self.head(h_bh)
            sampled_x_b = jax.random.categorical(key, logits, axis=-1)
            x_bmn = x_bmn.at[:,i,j].set(sampled_x_b)

            sampled_x_bh = self.embedding(sampled_x_b)
            x_bmnh = x_bmnh.at[:,i,j].set(sampled_x_bh)
            key, _ = jax.random.split(key)

            return (x_bmnh, h_bmnh, c_bmnh, x_bmn, key), sampled_x_b

        # first build up state for non-masked part of image.
        carry = (x_bmnh, h_bmnh, c_bmnh)
        xs = []
        for i in range(1, m+1):
            for j in range(1, n+1):
                xs.append(i * (n+2) + j)
        xs = jnp.array(xs)

        coord_index_in_padded = (coord[0] + 1) * (n+2) + (coord[1] + 1)
        xs_eval = xs[xs < coord_index_in_padded]
        xs_completion = xs[xs >= coord_index_in_padded]

        carry, _ = eval_scan_pixels(carry, xs_eval)

        carry = (carry[0], carry[1], carry[2], x_bmn, key)
        carry, _ = gen_scan_pixels(carry, xs_completion)

        x_bmn_completed = carry[3]
        return x_bmn_completed
