import tensorflow as tf
from utils.config import SENTENCE_LENGTH, EMBEDDING_SIZE, TOKEN_NUMBER, OUTPUT_KEEP_PROB, BATCH_SIZE, LEARNING_RATE, \
    Mode


def embedding_layer(input_placeholder):
    embedding = tf.get_variable(initializer=tf.random_uniform((TOKEN_NUMBER, EMBEDDING_SIZE), -1.0, 1.0),
                                name="embedding")
    return tf.nn.embedding_lookup(embedding, input_placeholder)


def rnn(inputs, mode):
    cell = tf.contrib.rnn.BasicLSTMCell(EMBEDDING_SIZE, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=OUTPUT_KEEP_PROB)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * 2)

    if mode == Mode.TRAIN:
        initial_state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    else:
        initial_state = cell.zero_state(1, dtype=tf.float32)
    return initial_state, tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def dense(inputs):
    weights = weight_variable((EMBEDDING_SIZE, TOKEN_NUMBER))
    bias = bias_variable((TOKEN_NUMBER,))
    return tf.nn.bias_add(tf.matmul(inputs, weights), bias=bias)


def build_network(mode=Mode.TRAIN):
    if mode == Mode.TRAIN:
        inputs = tf.placeholder(tf.int32, shape=(BATCH_SIZE, None))
    else:
        inputs = tf.placeholder(tf.int32, shape=(1, None))

    embedding = embedding_layer(inputs)

    initial_state, (rnn_layer, last_state) = rnn(embedding, mode=mode)
    flatten = tf.reshape(rnn_layer, [-1, EMBEDDING_SIZE])

    logits = dense(flatten)

    if mode == Mode.TRAIN:
        expect_tokens = tf.placeholder(tf.int32, shape=(BATCH_SIZE, SENTENCE_LENGTH))
        labels = tf.one_hot(tf.reshape(expect_tokens, [-1, 1]), depth=TOKEN_NUMBER)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        total_loss = tf.reduce_mean(loss)

        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_loss)

        return inputs, expect_tokens, last_state, total_loss, train_op

    else:
        prediction = tf.nn.softmax(logits=logits)

        return inputs, initial_state, last_state, prediction
