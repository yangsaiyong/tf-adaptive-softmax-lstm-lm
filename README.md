# tf-adaptive-softmax-lstm-lm

This repository shows the experiment result of LSTM language models on PTB (Penn Treebank) and GBW ([Google One Billion Word](https://code.google.com/archive/p/1-billion-word-language-modeling-benchmark/)) using **AdaptiveSoftmax** on TensorFlow. 

## Adaptive Softmax

The adaptive softmax is a faster way to train a softmax classifier over a huge number of classes, and can be used for **both training and prediction**. For example, it can be used for training a Language Model with a very huge vocabulary, and the trained languaed model can be used in speech recognition, text generation, and machine translation very efficiently.

Tha adaptive softmax has been used in the ASR system developed by **Tencent AI Lab**, and achieved about **20x speed up** than full sotfmax in the second pass for rescoing.

See [Efficient softmax approximation for GPUs](https://arxiv.org/pdf/1609.04309v2.pdf) for detail about the adaptive softmax algorithms.

## Implementation

The implementation of AdaptiveSoftmax on TensorFlow can be found here: [TencentAILab/tensorflow](https://github.com/TencentAILab/tensorflow/blob/master/tensorflow/python/ops/nn_impl.py)

## Usage
```python
#outputs is a tensor of shape [batch_size, lstm_hidden_size * num_step]
output = tf.reshape(tf.concat(1, outputs), [-1, lstm_hidden_size])

if softmax_type == "AdaptiveSoftmax":
	cutoff = config.adaptive_softmax_cutoff # For example: [2000,10000]
	loss, losses_for_train = tf.nn.adaptive_softmax_loss(output, labels, cutoff)
else: # Full softmax
	softmax_w = tf.get_variable("softmax_w", [lstm_hidden_size, vocab_size])
	softmax_b = tf.get_variable("softmax_b", [vocab_size])
	logits = tf.matmul(output, softmax_w) + softmax_b
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
	losses_for_train = [loss]

if is_training:
	lr = tf.Variable(0.1, trainable=False)
	optimizer = tf.train.AdagradOptimizer(lr, 1e-5)
	tvars = tf.trainable_variables()
	grads = tf.gradients([tf.reduce_sum(loss) / batch_size for loss in losses_for_train], tvars)
	grads = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in grads]
	eval_op = optimizer.apply_gradients(zip(grads, tvars))
else:
	eval_op = tf.no_op()
```

## Language Modeling Result on PTB
**hyper parameters:**
```python
gpu_num = 1
word_embedding_dim = 512
train_batch_size = 128
train_step_size = 20
valid_batch_size = 128
valid_step_size = 20
test_batch_size = 20
test_step_size = 1
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
softmax = AdaptiveSoftmax;2000
vocab_size = 10001
```
**result:**

| Epoch | Elapse  | Train PPL | Valid PPL | Test PPL |
| ----- | ------  | ----------| --------- | -------- |
| 1     | 0min37s | 409.586   | 175.262   | 169.584  |
| 2     | 1min13s | 159.506   | 136.062   | 130.974  |
| 3     | 1min49s | 120.364   | 120.210   | 115.960  |
| 4     | 2min25s | 101.029   | 112.828   | 108.182  |
| 5     | 3min01s | 89.171    | 108.167   | 103.796  |
| 6     | 3min37s | 81.405    | 106.010   | 101.007  |
| 7     | 4min13s | 75.221    | 104.244   | 99.148   |
| 8     | 4min49s | 70.694    | 102.997   | 97.355   |
| 9     | 5min25s | 67.076    | 102.685   | 97.030   |
| 10    | 6min01s | 61.166    | 99.943    | 94.251   |
| 11    | 6min37s | 58.201    | 99.668    | 93.711   |
| 12    | 7min13s | 55.737    | 97.806    | 91.907   |
| 13    | 7min49s | 54.249    | 97.869    | 91.754   |
| 14    | 8min25s | 53.289    | 96.525    | 90.529   |
| 15    | 9min01s | 52.703    | 96.411    | 90.333   |
| 16    | 9min37s | 52.178    | 95.803    | 89.813   |
| 17    | 10min13s| 51.844    | 95.484    | 89.489   |
| 18    | 10min49s| 51.640    | 95.287    | 89.316   | 


## Language Modeling Result on Google 1Billion Word corpus
**hyper parameters:**
```python
gpu_num = 2
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
softmax = AdaptiveSoftmax;4000,40000,200000
vocab_size = 793471
```
**result:**

On GBW corpus, we achived a perplexcity of 43.24 after 5 epochs, taking about two days to train on 2 GPUs with synchronous gradient updates. Detail experiment result and usage demo can be found here [tf-adaptive-softmax-lstm-lm](https://github.com/yangsaiyong/tf-adaptive-softmax-lstm-lm).

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
