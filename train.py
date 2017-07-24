from model import build_network, get_train_step, Generator
from utils import get_token_index_map, get_index_token_map
import tensorflow as tf
from utils.config import *
import logging

logging.basicConfig(format="%(levelname)s | %(asctime)s - %(message)s",
                    level=logging.DEBUG)


def get_inputs(next_sentence, index_token_map):
    return [index_token_map[token] for token in next_sentence]


def get_outputs(following_word, index_token_map):
    one_hot = [0] * TOKEN_NUMBER
    token_index = index_token_map[following_word]
    one_hot[token_index] = 1
    return one_hot


def train(input_placeholder, output_data, sess):
    token_index_map = get_token_index_map()
    index_token_map = get_index_token_map()

    assert len(index_token_map.keys()) == TOKEN_NUMBER

    generator = Generator()

    y, train_step = get_train_step(output_data)

    # saver = tf.train.Saver()
    # checkpoint = tf.train.get_checkpoint_state(CHECKOUT_FOLDER)

    while True:
        try:
            next_sentence = generator.next_sentence()
            following_word = generator.following_word()
        except IndexError:
            logging.info("reach the end of the file")
            break

        inputs = get_inputs(next_sentence, index_token_map)
        outputs = get_outputs(following_word, index_token_map)

        # output_data.eval(feed_dict={input_placeholder: inputs})

        train_step.run(feed_dict={
            y: outputs,
            input_placeholder: inputs
        })


def main():
    sess = tf.InteractiveSession()
    input_placeholder, output_data = build_network()
    train(input_placeholder, output_data, sess)


if __name__ == '__main__':
    main()
