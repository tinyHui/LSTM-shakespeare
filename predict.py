import logging
import os

import numpy as np
import tensorflow as tf

from model import build_network
from model.generator import get_index_token_map, get_token_index_map
from utils.config import Mode

tf.app.flags.DEFINE_string('checkpoints_dir', os.path.abspath('./checkpoints/'), 'checkpoints save path.')
tf.app.flags.DEFINE_string('model_prefix', 'shakespeare', 'model save prefix.')
FLAGS = tf.app.flags.FLAGS

logging.basicConfig(format="%(levelname)s | %(asctime)s - %(message)s",
                    level=logging.DEBUG)

BEGIN_WORD = "I"


def to_word(prediction, index_token_map):
    # weighted pick
    t = np.cumsum(prediction)
    s = np.sum(prediction)
    index = int(np.searchsorted(t, np.random.rand(1) * s))
    return index_token_map[index]


def to_input_tensor(word, token_index_map):
    input_tensor = np.array([[token_index_map[word]]])
    return input_tensor


def predict(sess):
    inputs, initial_state, last_state, prediction = build_network(Mode.PREDICT)

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    saver = tf.train.Saver(tf.global_variables())

    start_epoch = 0
    checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
    if checkpoint:
        logging.info("Found checkoutpoint, trying to restore from it")
        saver.restore(sess, checkpoint)
        logging.info("Restored successfully")
        start_epoch += int(checkpoint.split('-')[-1])
    else:
        raise IOError("required a saved session")

    token_index_map = get_token_index_map()
    index_token_map = get_index_token_map()

    article = [BEGIN_WORD]

    # generate second word
    input_tensor = to_input_tensor(BEGIN_WORD, token_index_map)
    softmax, last_state_rnn = sess.run([prediction, last_state],
                                       feed_dict={inputs: input_tensor})
    next_word = to_word(softmax, index_token_map)
    article.append(next_word)

    while True:
        input_tensor = to_input_tensor(next_word, token_index_map)
        softmax, last_state_rnn = sess.run([prediction, last_state],
                                           feed_dict={inputs: input_tensor, last_state: last_state_rnn})
        next_word = to_word(softmax, index_token_map)
        article.append(next_word)

        if len(article) % 500 == 0:
            print("Generated Play:", " ".join(article))
            break


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    predict(sess)
    sess.close()
