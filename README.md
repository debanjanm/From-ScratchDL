# From-ScratchDL

Deep learning, implemented from scratch. Python + NumPy only — no PyTorch,
no TensorFlow, no autograd. Every gradient is derived and coded by hand.

Matplotlib is used only for plotting results, never for the core math.

## Structure

Each topic is a self-contained, runnable folder — a Jupyter notebook (theory +
code + plots) or, until converted, a script — numpy-only, with asserts that
double as sanity checks.

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
- [x] `07_attention_transformer` — multi-head attention, positional encoding, LayerNorm, encoder blocks, full gradient check

Topics stay self-contained: each redefines the small layers/losses it needs
(`Dense`, `ReLU`, `Sigmoid`, ...) rather than importing from a shared module,
so any folder can be read and run on its own. `common/framework.py` is a
small reusable MLP framework (used by `03_optimizers` for activations/loss).

## Requirements

`numpy` for all core code; `matplotlib` for plots; `jupyter` for the notebooks.

## Running

```bash
jupyter lab 01_perceptron/perceptron.ipynb   # notebooks (01-02 converted so far)
python 03_optimizers/optimizers.py           # scripts (03-07, pending conversion)
./check_all.sh                               # run everything headless; non-zero exit on failure
```
