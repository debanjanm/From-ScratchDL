# From-ScratchDL

Deep learning, implemented from scratch. Python + NumPy only — no PyTorch,
no TensorFlow, no autograd. Every gradient is derived and coded by hand.

Matplotlib is used only for plotting results, never for the core math.

## Structure

Each topic is a self-contained, runnable folder — one script (or a few),
numpy-only, with a `__main__` demo that doubles as a sanity check.

```
common/               small reusable MLP framework (optional)
NN_topic_name/        one folder per topic, numbered in learning order (below)
```

## Roadmap

- [x] `01_perceptron` — single-neuron Rosenblatt perceptron, step activation
- [x] `02_backpropagation` — manual forward/backward pass, XOR problem
- [x] `03_optimizers` — SGD w/ momentum, RMSprop, Adam (compare convergence)
- [x] `04_regularization` — L2 weight decay, dropout, batch norm
- [x] `05_cnn` — conv2d, max-pool, im2col; synthetic 24x24 shapes (no dataset download)
- [x] `06_rnn_lstm` — vanilla RNN + LSTM cell, sequence prediction

Topics stay self-contained: each redefines the small layers/losses it needs
(`Dense`, `ReLU`, `Sigmoid`, ...) rather than importing from a shared module,
so any folder can be read and run on its own. `common/framework.py` is a
small reusable MLP framework (used by `03_optimizers` for activations/loss).

## Requirements

`numpy` for all core code; `matplotlib` only for the saved plots.

## Running a topic

```bash
python 02_backpropagation/backpropagation.py   # or any topic folder's script
```
