import tensorflow as tf
from utils.config import *


def embedding_layer(input_placeholder):
    embedding = tf.get_variable("embedding",
                                initializer=tf.random_uniform([BATCH_SIZE, TOKEN_NUMBER + 1, EMBEDDING_SIZE], -1.0, 1.0))
    return tf.nn.embedding_lookup(embedding, input_placeholder)


def rnn(inputs):
    cell = tf.contrib.rnn.BasicLSTMCell(EMBEDDING_SIZE, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=OUTPUT_KEEP_PROB)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * 2)

    initial_state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    return tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def dense(inputs):
    weights = weight_variable((EMBEDDING_SIZE, TOKEN_NUMBER + 1))
    bias = bias_variable((TOKEN_NUMBER + 1,))
    return tf.nn.bias_add(tf.matmul(inputs, weights), bias=bias)


def build_network():
    input_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE, MAX_LENGTH))
    input_embedding = embedding_layer(input_placeholder)

    outputs, last_state = rnn(input_embedding)
    output = dense(tf.reshape(outputs, [-1, EMBEDDING_SIZE]))

    one_hot = tf.one_hot(tf.reshape(output, [-1]), depth=TOKEN_NUMBER + 1)
    return input_placeholder, one_hot


def get_train_step(output_data):
    y = tf.placeholder(tf.int8, shape=(BATCH_SIZE, TOKEN_NUMBER))
    cost = tf.reduce_mean(tf.square(y - output_data))
    return y, tf.train.AdamOptimizer(1e-6).minimize(cost)
