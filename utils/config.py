# COMEDY_FULL_TEXT = "./data/partial-text.txt"
COMEDY_FULL_TEXT = "./data/full-text.txt"
TOKEN_INDEX_MAP = "./bin/token_index.map"
INDEX_TOKEN_MAP = "./bin/index_token.map"
CHECKOUT_FOLDER = "./bin/checkpoint"

TOKEN_NUMBER = 19416
BATCH_SIZE = 64
EMBEDDING_SIZE = 128
SENTENCE_LENGTH = 30
OUTPUT_KEEP_PROB = 0.5
LEARNING_RATE = 0.3


class Mode:
    TRAIN = 'train'
    PREDICT = 'predict'
