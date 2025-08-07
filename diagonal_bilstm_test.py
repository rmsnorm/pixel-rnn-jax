from flax import nnx

import unittest
import jax
import jax.numpy as jnp
import diagonal_bilstm


class TestSkewing(unittest.TestCase):
    def test_skew(self):
        x = jnp.arange(1, 10).reshape((1, 3, 3, 1))
        expected_skewed_x = jnp.array(
            [
                [
                    [[1], [2], [3], [0], [0]],
                    [[0], [4], [5], [6], [0]],
                    [[0], [0], [7], [8], [9]],
                ]
            ]
        )
        self.assertTrue(
            jnp.allclose(diagonal_bilstm.skew_feature_map(x), expected_skewed_x)
        )

    def test_unskew(self):
        x = jnp.array(
            [
                [
                    [[1], [2], [3], [0], [0]],
                    [[0], [4], [5], [6], [0]],
                    [[0], [0], [7], [8], [9]],
                ]
            ]
        )
        expected_unskewed_x = jnp.arange(1, 10).reshape((1, 3, 3, 1))
        self.assertTrue(
            jnp.allclose(diagonal_bilstm.unskew_feature_map(x), expected_unskewed_x)
        )


class TestConvMaskGeneration(unittest.TestCase):
    def test_1x1_mask_b(self):
        features = 4
        mask_b = diagonal_bilstm.create_mask(
            (1, 1), 2 * features, 4 * features, 3, "mask_b"
        )

        expected_mask_b = jnp.array(
            [
                [
                    [[1], [0], [0], [1], [0], [0], [1], [0]],
                    [[1], [1], [0], [1], [1], [0], [1], [1]],
                    [[1], [1], [1], [1], [1], [1], [1], [1]],
                    #
                    [[1], [0], [0], [1], [0], [0], [1], [0]],
                    [[1], [1], [0], [1], [1], [0], [1], [1]],
                    [[1], [1], [1], [1], [1], [1], [1], [1]],
                    #
                    [[1], [0], [0], [1], [0], [0], [1], [0]],
                    [[1], [1], [0], [1], [1], [0], [1], [1]],
                    [[1], [1], [1], [1], [1], [1], [1], [1]],
                    #
                    [[1], [0], [0], [1], [0], [0], [1], [0]],
                    [[1], [1], [0], [1], [1], [0], [1], [1]],
                    [[1], [1], [1], [1], [1], [1], [1], [1]],
                    #
                    [[1], [0], [0], [1], [0], [0], [1], [0]],
                    [[1], [1], [0], [1], [1], [0], [1], [1]],
                    [[1], [1], [1], [1], [1], [1], [1], [1]],
                    #
                    [[1], [0], [0], [1], [0], [0], [1], [0]],
                ]
            ]
        ).transpose(0, 3, 2, 1)
        self.assertEqual(mask_b.shape, expected_mask_b.shape)
        self.assertTrue(jnp.allclose(mask_b, expected_mask_b))

    def test_1d_mask_a(self):
        mask_a = diagonal_bilstm.create_mask(
            (1, 3),
            input_channels=3,
            output_channels=5,
            image_channels=3,
            mask_type="mask_a",
        )
        # conv and its mask will be of shape [kernel, input_ch, output_ch],
        # but first I'll construct the expected mask in [output_ch, input_ch, kernel] shape
        # and then transpose it.

        # The way I create the expected mask is to first build up the [5, 3, kernel] shaped matrix from the 1x3 conv kernel
        # for each input channel and output channel. This conv kernel should be [1,1,0] to begin with.
        # Then we zero out the center in the kernel if the output channel shouldn't
        # be looking at that particular input channel.
        expected_mask_a = jnp.array(
            [
                [
                    [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                    [[1, 1, 0], [1, 0, 0], [1, 0, 0]],
                    [[1, 1, 0], [1, 1, 0], [1, 0, 0]],
                    [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
                    [[1, 1, 0], [1, 0, 0], [1, 0, 0]],
                ]
            ]
        ).transpose(0, 3, 2, 1)
        self.assertTrue(jnp.allclose(mask_a, expected_mask_a))
        self.assertEqual(mask_a.shape, expected_mask_a.shape)

    def test_1d_mask_b(self):
        mask_b = diagonal_bilstm.create_mask(
            (1, 3),
            input_channels=3,
            output_channels=5,
            image_channels=3,
            mask_type="mask_b",
        )
        expected_mask_b = jnp.array(
            [
                [
                    [[1, 1, 0], [1, 0, 0], [1, 0, 0]],
                    [[1, 1, 0], [1, 1, 0], [1, 0, 0]],
                    [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
                    [[1, 1, 0], [1, 0, 0], [1, 0, 0]],
                    [[1, 1, 0], [1, 1, 0], [1, 0, 0]],
                ]
            ]
        ).transpose(0, 3, 2, 1)
        self.assertEqual(mask_b.shape, expected_mask_b.shape)
        self.assertTrue(jnp.allclose(mask_b, expected_mask_b))

    def test_2d_mask_a(self):
        mask_a = diagonal_bilstm.create_mask(
            (3, 3),
            input_channels=3,
            output_channels=5,
            image_channels=3,
            mask_type="mask_a",
        )
        basic_2d_mask = jnp.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]])
        expected_mask = jnp.tile(
            basic_2d_mask[jnp.newaxis, jnp.newaxis, :, :], (5, 3, 1, 1)
        )

        # output red channel cannot look at anything in the center pixel
        expected_mask = expected_mask.at[0::3, :, 1, 1].set(0)

        # output green channel cannot look at input green channel and blue channel
        expected_mask = expected_mask.at[1::3, 1, 1, 1].set(0)
        expected_mask = expected_mask.at[1::3, 2, 1, 1].set(0)

        # output blue channel cannot look at input blue channel
        expected_mask = expected_mask.at[2::3, 2, 1, 1].set(0)

        expected_mask = expected_mask.transpose(2, 3, 1, 0)

        self.assertTrue(jnp.allclose(mask_a, expected_mask))

    def test_2d_mask_b(self):
        mask_b = diagonal_bilstm.create_mask(
            (3, 3),
            input_channels=3,
            output_channels=5,
            image_channels=3,
            mask_type="mask_b",
        )
        # construct the autoregressive basic mask.
        basic_2d_mask = jnp.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]])
        expected_mask = jnp.tile(
            basic_2d_mask[jnp.newaxis, jnp.newaxis, :, :], (5, 3, 1, 1)
        )

        # output red channel cannot look at green and blue input channels
        expected_mask = expected_mask.at[0::3, 1, 1, 1].set(0)
        expected_mask = expected_mask.at[0::3, 2, 1, 1].set(0)

        # output green channel cannot look at input blue channel
        expected_mask = expected_mask.at[1::3, 2, 1, 1].set(0)

        expected_mask = expected_mask.transpose(2, 3, 1, 0)

        self.assertTrue(jnp.allclose(mask_b, expected_mask))


class TestModel(unittest.TestCase):
    def setUp(self):
        rngs = nnx.Rngs(params=0)
        self.model = diagonal_bilstm.DiagonalBiLSTM(
            features=128,
            num_layers=4,
            is_rgb=True,
            output_conv_out_channels=[1024, 1024],
            rngs=rngs,
        )

        key = jax.random.key(0)
        b, m, n, c = 1, 4, 4, 3
        self.im_bmnc = jax.random.choice(key, 256, (b, m, n, c))

        # Test positions to predict
        test_positions = [
            (0, 0, 0),  # First R channel
            (0, 0, 1),  # First G channel (should depend on R but not future pixels)
            (0, 1, 0),  # Second pixel R channel
            (1, 0, 0),  # Second row, first pixel
        ]

        self.modified_images = []
        for test_i, test_j, test_c in test_positions:
            img_modified = self.im_bmnc.copy()
            for i in range(m):
                for j in range(n):
                    for c in range(3):
                        is_future = (
                            (i > test_i)
                            or (i == test_i and j > test_j)
                            or (i == test_i and j == test_j and c > test_c)
                        )
                        if is_future:
                            img_modified = img_modified.at[:, i, j, c].set(255)
            self.modified_images.append((test_i, test_j, test_c, img_modified))

    def test_input_conv_autoregressive_masking(self):
        """Test if the model's input conv respects autoregressive ordering if
        outputs change when future pixels are modified."""
        im_bmnc_normed = (self.im_bmnc - 128.0) / 256.0
        input_conv_out_orig = self.model.input_conv(im_bmnc_normed)
        for test_i, test_j, test_c, img_modified in self.modified_images:
            img_modified_normed = (img_modified - 128.0) / 256.0
            input_conv_out_modified = self.model.input_conv(img_modified_normed)

            self.assertTrue(
                jnp.allclose(
                    input_conv_out_modified[:, test_i, test_j, test_c::3],
                    input_conv_out_orig[:, test_i, test_j, test_c::3],
                )
            )

    def test_skewed_inputs(self):
        """Test if skewed inputs are equal after input conv."""
        im_bmnc = (self.im_bmnc - 128.0) / 256.0
        x_bmnh = self.model.input_conv(im_bmnc)
        x_bm2nh = diagonal_bilstm.skew_feature_map(x_bmnh)

        for test_i, test_j, test_c, img_modified in self.modified_images:
            img_modified_normed = (img_modified - 128.0) / 256.0
            x_mod_bmnh = self.model.input_conv(img_modified_normed)
            x_mod_bm2nh = diagonal_bilstm.skew_feature_map(x_mod_bmnh)

            self.assertTrue(
                jnp.allclose(
                    x_mod_bm2nh[:, test_i, test_i + test_j, test_c::3],
                    x_bm2nh[:, test_i, test_i + test_j, test_c::3],
                )
            )

    def test_left_layer_autoregressive_masking(self):
        """Test if the 1st left layer respects autoregressive ordering if
        outputs change when future pixels are modified."""

        im_bmnc = (self.im_bmnc - 128.0) / 256.0
        x_bmnh = self.model.input_conv(im_bmnc)
        x_bm2nh = diagonal_bilstm.skew_feature_map(x_bmnh)

        features_orig = diagonal_bilstm.unskew_feature_map(
            self.model.layers[0].left_layer(x_bm2nh)
        )

        for test_i, test_j, test_c, img_modified in self.modified_images:
            img_modified_normed = (img_modified - 128.0) / 256.0
            x_mod_bmnh = self.model.input_conv(img_modified_normed)
            x_mod_bm2nh = diagonal_bilstm.skew_feature_map(x_mod_bmnh)

            features_modified = diagonal_bilstm.unskew_feature_map(
                self.model.layers[0].left_layer(x_mod_bm2nh)
            )

            self.assertTrue(
                jnp.allclose(
                    features_modified[:, test_i, test_j, test_c::3],
                    features_orig[:, test_i, test_j, test_c::3],
                )
            )

    def test_bilstm_layer_autoregressive_masking(self):
        """Test if the 1st bi-lstm layer respects autoregressive ordering if
        outputs change when future pixels are modified."""

        im_bmnc = (self.im_bmnc - 128.0) / 256.0
        x_bmnh = self.model.input_conv(im_bmnc)
        x_bm2nh = diagonal_bilstm.skew_feature_map(x_bmnh)

        features_orig = diagonal_bilstm.unskew_feature_map(
            self.model.layers[0](x_bm2nh)
        )

        for test_i, test_j, test_c, img_modified in self.modified_images:
            img_modified_normed = (img_modified - 128.0) / 256.0
            x_mod_bmnh = self.model.input_conv(img_modified_normed)
            x_mod_bm2nh = diagonal_bilstm.skew_feature_map(x_mod_bmnh)

            features_modified = diagonal_bilstm.unskew_feature_map(
                self.model.layers[0](x_mod_bm2nh)
            )

            self.assertTrue(
                jnp.allclose(
                    features_modified[:, test_i, test_j, test_c::3],
                    features_orig[:, test_i, test_j, test_c::3],
                )
            )

    def test_model_autoregressive_masking(self):
        logits_orig = self.model(self.im_bmnc)
        for test_i, test_j, test_c, img_modified in self.modified_images:
            logits_modified = self.model(img_modified)
            pred_orig = logits_orig[0, test_i, test_j, test_c]
            pred_modified = logits_modified[0, test_i, test_j, test_c]
            self.assertTrue(jnp.allclose(pred_modified, pred_orig))

    def test_model_channel_dependencies(self):
        logits_orig = self.model(self.im_bmnc)

        # Test 1: R channel should NOT depend on G,B at same position
        img_modified = (
            self.im_bmnc.copy().at[0, 0, 0, 1].set(255).at[0, 0, 0, 2].set(255)
        )  # Set G,B to 255

        logits = self.model(img_modified)
        self.assertTrue(jnp.allclose(logits[0, 0, 0, 0], logits_orig[0, 0, 0, 0]))

        # Test 2: G channel SHOULD depend on R at same position
        img_modified = self.im_bmnc.copy().at[0, 0, 0, 0].set(128)  # Change R channel
        logits = self.model(img_modified)
        self.assertFalse(jnp.allclose(logits[0, 0, 0, 1], logits_orig[0, 0, 0, 1]))


if __name__ == "__main__":
    unittest.main()
