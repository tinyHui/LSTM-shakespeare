from model import build_network, get_next_token
from utils import get_token_index_map, get_index_token_map
import tensorflow as tf

COMEDY_FULL_TEXT = "./data/full-text.txt"
TOKEN_INDEX_MAP = "./bin/token_index.map"
INDEX_TOKEN_MAP = "./bin/index_token.map"

TOKEN_NUMBER = 15672
SENTENCES = 125986
MAX_LENGTH = 44


def train(input_placeholder, output_data, sess):
    token_index_map = get_token_index_map()
    index_token_map = get_index_token_map()

    saver = tf.train.Saver()

    while True:
        output_data.eval(feed_dict={})
        next_token = get_next_token(output_data)


def main():
    sess = tf.InteractiveSession()
    input_placeholder, output_data = build_network(MAX_LENGTH=MAX_LENGTH)
    train(input_placeholder, output_data, sess)


if __name__ == '__main__':
    main()
