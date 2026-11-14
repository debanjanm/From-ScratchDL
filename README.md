# From-ScratchDL

Deep learning, implemented from scratch. Python + NumPy only — no PyTorch,
no TensorFlow, no autograd. Every gradient is derived and coded by hand.

Matplotlib is used only for plotting results, never for the core math.
Open any notebook with `jupyter lab`; Run All executes top to bottom.

## Structure

Each topic is a self-contained folder with one Jupyter notebook: theory (with
equations), numpy-only code, an experiment with inline plots, and asserts that
double as sanity checks.

```
common/               reference helpers (framework.py, gradcheck.py); notebooks don't import them
NN_topic_name/        one folder per topic, numbered in learning order (below)
```

## Roadmap

- [x] [`01_perceptron`](01_perceptron/perceptron.ipynb) — single-neuron Rosenblatt perceptron, step activation
- [x] [`02_backpropagation`](02_backpropagation/backpropagation.ipynb) — manual forward/backward pass, XOR problem
- [x] [`03_optimizers`](03_optimizers/optimizers.ipynb) — SGD w/ momentum, RMSprop, Adam (compare convergence)
- [x] [`04_regularization`](04_regularization/regularization.ipynb) — L2 weight decay, dropout, batch norm
- [x] [`05_cnn`](05_cnn/cnn.ipynb) — conv2d, max-pool, im2col; synthetic 24x24 shapes (no dataset download)
- [x] [`06_rnn_lstm`](06_rnn_lstm/rnn_lstm.ipynb) — vanilla RNN + LSTM cell, sequence prediction
- [x] [`07_attention_transformer`](07_attention_transformer/transformer.ipynb) — multi-head attention, positional encoding, LayerNorm, encoder blocks, full gradient check

Topics stay self-contained: each notebook defines every layer, loss and helper
it needs, so any one can be read and run on its own. `common/` holds the
standalone MLP framework and the finite-difference gradient checker that
notebooks 02 and 07 inline.

## Requirements

`numpy` for all core code; `matplotlib` for plots; `jupyter` for the notebooks.

## Running

```bash
jupyter lab 03_optimizers/optimizers.ipynb   # open any topic
./check_all.sh                               # execute every notebook headless; non-zero exit on failure
```
