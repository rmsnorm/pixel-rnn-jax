## Pixel RNN implementation in JAX

This repo implements the 3 variants of PixelRNN (van den Oord et. al.) namely, Row-LSTM, Diagonal-BiLSTM and PixelCNN in JAX.

### Implementation Progress

<!-- tick mark - "&#x2611;" -->
<!-- cross - "&#x2612;"  -->

||Row-LSTM | Diagonal-BiLSTM | PixelCNN|
|-|--|--|--|
|Basic impl|&#x2611;|--|--|
|Residual Connections|&#x2611;|--|--|
|Skip Connections|&#x2611;|--|--|
|Supports RGB images (masked conv)|--|--|--|


### Trained Row-LSTM on Binarized MNIST

You can find the training loop in row_lstm_mnist.ipynb . 

<img src="generated_images/row-lstm-binarized-mnist.png" alt="Row-LSTM-Binarized-MNIST" width="300" height="300">

The above images are generated from a Row-LSTM trained with

- model hyper-params:
    - i-s kernel width = 3
    - s-s kernel width = 3
    - num-layers = 16
    - hidden-dim = 16
    - skip-connections & residual connections
- training params:
    - batch-size = 16
    - training-images = 900K
    - on binarized MNIST data