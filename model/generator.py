from tokens import get_tokens
from utils import config
import pickle as pkl
import numpy as np
import logging


def get_token_index_map():
    with open(config.TOKEN_INDEX_MAP, 'rb') as f:
        return dict(pkl.load(f))


def get_index_token_map():
    with open(config.INDEX_TOKEN_MAP, 'rb') as f:
        return dict(pkl.load(f))


class Generator:
    def __init__(self, file, batch_size, sentence_length):
        with open(file, 'r') as f:
            full_text = f.read()
            self.full_tokens = get_tokens(full_text)
            self.sentences_n = len(self.full_tokens)

        self.__iter_count = 0
        self.token_index_map = get_token_index_map()
        self.BATCH_SIZE = batch_size
        self.SENTENCE_LENGTH = sentence_length

        self.chunk_n = self.sentences_n // (self.BATCH_SIZE * self.SENTENCE_LENGTH)
        logging.info(f"Able to get {self.chunk_n} chunks")

    def next_batch(self):
        start_index = self.__iter_count * self.BATCH_SIZE * self.SENTENCE_LENGTH
        end_index = start_index + self.BATCH_SIZE * self.SENTENCE_LENGTH

        current_batch = self.__get_tokens(start_index, end_index)
        self.__iter_count += 1
        return np.asarray(current_batch)

    def following_tokens(self):
        start_index = (self.__iter_count - 1) * self.BATCH_SIZE * self.SENTENCE_LENGTH + 1
        end_index = start_index + self.BATCH_SIZE * self.SENTENCE_LENGTH + 1
        current_batch = self.__get_tokens(start_index, end_index)
        return np.asarray(current_batch)

    def have_next(self):
        has_next = self.__iter_count < self.chunk_n
        if has_next:
            return True
        else:
            logging.info("Reached the end of the document")
            return False

    def get_iter_count(self):
        return self.__iter_count

    def reset_iterator(self):
        self.__iter_count = 1

    def __get_tokens(self, start_index, end_index):
        current_batch = [self.full_tokens[start_index:end_index][i:i + self.SENTENCE_LENGTH]
                         for i in range(0, self.BATCH_SIZE * self.SENTENCE_LENGTH, self.SENTENCE_LENGTH)]
        current_batch = [[self.token_index_map[current_token] for current_token in current_sentence]
                         for current_sentence in current_batch]
        return current_batch

    def __len__(self):
        return self.sentences_n
