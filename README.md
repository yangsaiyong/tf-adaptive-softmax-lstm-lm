# tf-adaptive-softmax-lstm-lm

This repository shows the experiment result of LSTM language models on PTB (Penn Treebank) and GBW ([Google One Billion Word](https://code.google.com/archive/p/1-billion-word-language-modeling-benchmark/)) using **AdaptiveSoftmax** on TensorFlow. 

## Adaptive Softmax

The adaptive softmax is a faster way to train a softmax classifier over a huge number of classes, and can be used for **both training and prediction**. For example, it can be used for training a Language Model with a very huge vocabulary, and the trained languaed model can be used in speech recognition, text generation, and machine translation very efficiently.

Tha adaptive softmax has been used in the ASR system developed by **Tencent AI Lab**, and achieved about **20x speed up** than full sotfmax in the second pass for rescoing.

See [Efficient softmax approximation for GPUs](https://arxiv.org/pdf/1609.04309v2.pdf) for detail about the adaptive softmax algorithms.

## Implementation

The implementation of AdaptiveSoftmax on TensorFlow can be found here: [TencentAILab/tensorflow](https://github.com/TencentAILab/tensorflow)

## Usage

Train with AdaptiveSoftmax:
```shell
python train_lm.py --data_path=ptb_data --use_adaptive_softmax=1
```
Train with full softmax:
```shell
python train_lm.py --data_path=ptb_data --use_adaptive_softmax=0
```

## Experiment results

### Language Modeling on PTB
With the hyper parameters below, it takes 5min54s to train 20 epochs on PTB corpus, the final perplexity on test set 
is *88.51*. With the same parameters and using full softmax, it takes 6min57s to train 20 epochs, and the final perplexity on test set is *89.00*. 

Since the PTB vocabulary size is only 10K, the speed up is not that significant.


**hyper parameters:**
```python
epoch_num = 20
51,train_batch_size = 128
train_step_size = 20
valid_batch_size = 128
valid_step_size = 20
test_batch_size = 20
test_step_size = 1
word_embedding_dim = 512
lstm_layers = 1
lstm_size = 512
lstm_forget_bias = 0.0
max_grad_norm = 0.25
init_scale = 0.05
learning_rate = 0.2
decay = 0.5
decay_when = 1.0
dropout_prob = 0.5
adagrad_eps = 1e-5
vocab_size = 10001
softmax_type = "AdaptiveSoftmax"
adaptive_softmax_cutoff = [2000, vocab_size]
```
**result:**

| Epoch | Elapse  | Train PPL | Valid PPL | Test PPL |
| ----- | ------  | ----------| --------- | -------- |
|  1    | 0min18s |   376.407 |  169.152  |  164.039 |
|  2    | 0min35s |   154.324 |  132.648  |  127.494 |
|  3    | 0min53s |   117.210 |  118.547  |  113.197 |
|  4    | 1min11s |    98.662 |  111.791  |  106.373 |
|  5    | 1min28s |    87.366 |  107.808  |  102.588 |
|  6    | 1min46s |    79.448 |  105.028  |  100.024 |
|  7    | 2min04s |    73.749 |  103.705  |   98.220 |
|  8    | 2min21s |    69.392 |  102.939  |   96.931 |
|  9    | 2min39s |    62.737 |  100.174  |   94.043 |
|  10   | 2min57s |    59.423 |   99.412  |   93.153 |
|  11   | 3min15s |    56.634 |   97.600  |   91.271 |
|  12   | 3min32s |    55.036 |   97.388  |   91.061 |
|  13   | 3min50s |    54.002 |   96.127  |   89.796 |
|  14   | 4min08s |    53.232 |   96.170  |   89.805 |
|  15   | 4min25s |    52.844 |   95.461  |   89.130 |
|  16   | 4min43s |    52.488 |   95.085  |   88.788 |
|  17   | 5min01s |    52.314 |   94.905  |   88.615 |
|  18   | 5min18s |    52.172 |   94.835  |   88.553 |
|  19   | 5min36s |    52.038 |   94.806  |   88.526 |
|  20   | 5min54s |    51.998 |   94.788  |   88.510 |


### Language Modeling on Google 1Billion Word corpus

**hyper parameters:**
```python
word_embedding_dim = 256
train_batch_size = 256
train_step_size = 20
valid_batch_size = 256
valid_step_size = 20
test_batch_size = 128
test_step_size = 1
lstm_layers = 1
lstm_size = 2048
lstm_forget_bias = 1.0
max_grad_norm = 0.25
init_scale = 0.05
learning_rate = 0.1
decay = 0.5
decay_when = 1.0
dropout_prob = 0.01
adagrad_eps = 1e-5
vocab_size = 793471
softmax_type = "AdaptiveSoftmax"
adaptive_softmax_cutoff = [4000,40000,200000, vocab_size]
```
**result:**

On GBW corpus, we achived a perplexcity of 43.24 after 5 epochs, taking about two days to train on 2 GPUs with synchronous gradient updates.

| Epoch | Elapse | Train PPL | Valid PPL | Test PPL |
| ----- | ------ | --------- | --------- | -------- |
|  1    | 9h56min|  51.428   |  52.727   | 49.553   |
|  2    |19h53min|  45.141   |  48.683   | 45.639   |
|  3    |29h51min|  42.605   |  47.379   | 44.332   |
|  4    |39h48min|  41.119   |  46.822   | 43.743   |
|  5    |49h45min|  38.757   |  46.402   | 43.241   |
|  6    |59h42min|  37.664   |  46.334   | 43.119   |
|  7    |69h40min|  37.139   |  46.337   | 43.101   |
|  8    |79h37min|  36.884   |  46.342   | 43.097   |

##Reference
[1] Grave E, Joulin A, Cissé M, et al. Efficient softmax approximation for GPUs[J]. arXiv preprint arXiv:1609.04309, 2016.

[2] https://github.com/facebookresearch/adaptive-softmax
