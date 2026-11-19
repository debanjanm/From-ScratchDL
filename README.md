# From-ScratchDL

Deep learning, implemented from scratch. Python + NumPy only — no PyTorch,
no TensorFlow, no autograd library. Every gradient is derived and coded by hand
(and notebook 12 then builds an autograd engine to check them).

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

## Topics

Meant to be read in order: later notebooks reuse ideas from earlier ones. Each
ends with a quantitative check (an `assert`), so the claim it makes is tested
every time it runs.

### [01 · Perceptron](01_perceptron/perceptron.ipynb)
- **You'll learn:** the single-neuron model, the mistake-driven Rosenblatt update rule, and what "linearly separable" means.
- **Key result:** learns AND and OR exactly; provably cannot learn XOR, which motivates hidden layers.

### [02 · Backpropagation](02_backpropagation/backpropagation.ipynb)
- **You'll learn:** the forward pass, the chain rule applied layer by layer, why sigmoid + cross-entropy gives a clean output gradient, and how to **verify gradients numerically**.
- **Key result:** a 2-layer net solves XOR; hand-derived gradients match finite differences to ~1e-8.

### [03 · Optimizers](03_optimizers/optimizers.ipynb)
- **You'll learn:** momentum, RMSprop and Adam: what each state variable tracks and why bias correction exists.
- **Key result:** all three converge on XOR from identical starting weights, with different convergence curves.

### [04 · Regularization](04_regularization/regularization.ipynb)
- **You'll learn:** overfitting, and L2 weight decay, (inverted) dropout and batch normalization, including the batch-norm backward pass.
- **Key result:** on a small noisy dataset the plain net memorizes (100% train, 66% test); the regularized net shrinks the gap (95% / 75%).

### [05 · Convolutional networks](05_cnn/cnn.ipynb)
- **You'll learn:** convolution as one matrix multiply via **im2col/col2im**, max-pooling, global pooling and why weight sharing gives translation invariance.
- **Key result:** trained only on shapes on the left of the image, the CNN generalizes to unseen positions (93%) where a dense net does not (62%).

### [06 · RNN and LSTM](06_rnn_lstm/rnn_lstm.ipynb)
- **You'll learn:** backpropagation through time, the LSTM's gates and cell-state highway, and why initialization (forget-gate bias) matters.
- **Key result:** measured gradient flow over 40 steps: the RNN's decays by ~15 orders of magnitude, the LSTM's barely drops. (Both solve the toy task, and the notebook says so.)

### [07 · Attention and the Transformer](07_attention_transformer/transformer.ipynb)
- **You'll learn:** scaled dot-product and multi-head attention, positional encoding, LayerNorm, residual encoder blocks, and the softmax backward pass.
- **Key result:** every parameter's gradient is finite-difference checked; removing positional encoding drops accuracy from 100% to 64%.

### [08 · Autoencoders and VAEs](08_autoencoder/autoencoder.ipynb)
- **You'll learn:** bottleneck compression, the ELBO, the reparameterization trick and the KL term, and why a VAE latent space can be sampled while a plain autoencoder's cannot.
- **Key result:** 100% of VAE prior samples decode to clean images vs 74% for the plain autoencoder.

### [09 · GANs](09_gan/gan.ipynb)
- **You'll learn:** the generator/discriminator game, the non-saturating loss, why training has no single loss to minimize, and how to measure sample **quality and mode coverage**.
- **Key result:** the generator's gradient flowing back *through the discriminator* is verified numerically; 99% clean samples covering all 25 position cells.

### [10 · Word embeddings (word2vec)](10_word2vec/word2vec.ipynb)
- **You'll learn:** skip-gram with negative sampling, why embeddings emerge from context prediction, and how it relates to PMI matrix factorization.
- **Key result:** embeddings cluster by category (88% neighbor purity) and solve gender analogies (100%); a PPMI+SVD baseline does as well, which the notebook reports honestly.

### [11 · Residual networks](11_resnet/resnet.ipynb)
- **You'll learn:** skip connections, how the gradient gets an identity path, initialization of residual blocks, and how to test a claim instead of repeating it.
- **Key result:** at the same settings the deep plain net diverges while the residual net trains at every depth, but a small learning rate rescues the plain net, so the real benefit is robustness.

### [12 · Autograd](12_autograd/autograd.ipynb)
- **You'll learn:** how automatic differentiation works (record the graph, replay it backwards), plus broadcasting-aware gradients and a classic numpy gotcha.
- **Key result:** every op is finite-difference checked, and the engine reproduces notebook 02's hand-derived gradients to ~1e-18.

Topics stay self-contained: each notebook defines every layer, loss and helper
it needs, so any one can be read and run on its own. `common/` holds the
standalone MLP framework and the finite-difference gradient checker that
notebooks 02 and 07-12 inline.

## Requirements

`numpy` for all core code; `matplotlib` for plots; `jupyter` for the notebooks.

```bash
pip install -r requirements.txt
```

## Running

```bash
jupyter lab 03_optimizers/optimizers.ipynb   # open any topic
./check_all.sh                               # execute every notebook headless; non-zero exit on failure
```
