from model import build_network, Generator
from utils import get_token_index_map, get_index_token_map
import tensorflow as tf
from utils.config import *


def train(input_placeholder, output_data, sess):
    token_index_map = get_token_index_map()
    index_token_map = get_index_token_map()

    generator = Generator()

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(CHECKOUT_FOLDER)

    while True:
        next_sentence = generator.next_sentence()
        following_word = generator.following_word()

        output_data.eval(feed_dict={})


def main():
    sess = tf.InteractiveSession()
    input_placeholder, output_data = build_network()
    train(input_placeholder, output_data, sess)


if __name__ == '__main__':
    main()
