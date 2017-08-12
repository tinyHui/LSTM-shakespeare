from utils.config import COMEDY_FULL_TEXT, TOKEN_INDEX_MAP, INDEX_TOKEN_MAP, TOKEN_NUMBER, BATCH_SIZE, SENTENCE_LENGTH
import pickle as pkl
import numpy as np
import logging


def get_token_index_map():
    with open(TOKEN_INDEX_MAP, 'rb') as f:
        return dict(pkl.load(f))


def get_index_token_map():
    with open(INDEX_TOKEN_MAP, 'rb') as f:
        return dict(pkl.load(f))


def _one_hot(dictionary, words):
    one_hot = [0] * TOKEN_NUMBER
    for word in words:
        index = dictionary[word]
        one_hot[index] = 1
    return np.asarray(one_hot, dtype=np.int32)


class Generator:
    def __init__(self):
        with open(COMEDY_FULL_TEXT, 'r') as f:
            full_text = f.read()
            self.full_tokens = full_text.split()
            self.sentences_n = len(self.full_tokens)

        self.__iter_count = 0
        self.token_index_map = get_token_index_map()
        self.chunk_n = self.sentences_n // (BATCH_SIZE * SENTENCE_LENGTH + 1)
        logging.info(f"able to get {self.chunk_n} chunks")

    def next_batch(self):
        start_index = self.__iter_count * BATCH_SIZE * SENTENCE_LENGTH
        end_index = start_index + BATCH_SIZE * SENTENCE_LENGTH

        if self.__iter_count > self.chunk_n:
            logging.info("reach the end of the file")
            return  # end of file, not able to get next batch

        current_batch = [self.full_tokens[start_index:end_index][i:i + SENTENCE_LENGTH]
                         for i in range(0, BATCH_SIZE * SENTENCE_LENGTH, SENTENCE_LENGTH)]
        current_batch = [_one_hot(self.token_index_map, current_tokens) for current_tokens in current_batch]
        # fold to batches
        self.__iter_count += 1
        yield np.asarray(current_batch)

    def following_tokens(self):
        following_words = []
        for i in range(SENTENCE_LENGTH, (BATCH_SIZE + 1) * SENTENCE_LENGTH, SENTENCE_LENGTH):
            following_words.append(_one_hot(self.token_index_map, [self.full_tokens[i]]))
        return np.asarray(following_words)

    def __len__(self):
        return self.sentences_n
