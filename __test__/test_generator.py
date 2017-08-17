from unittest import TestCase

from model import Generator
from model.generator import get_index_token_map
from utils import config

config.TOKEN_INDEX_MAP = "../bin/token_index.map"
config.INDEX_TOKEN_MAP = "../bin/index_token.map"
index_token_map = get_index_token_map()


def get_tokens_in_word(tokens_in_index):
    return [[index_token_map[index] for index in batch] for batch in tokens_in_index]


class TestGenerator(TestCase):
    def setUp(self):
        self.test_file = "./test_file"
        self.batch_size = 2
        self.sentence_length = 3

    def test_get_next_batch(self):
        generator = Generator(self.test_file, self.batch_size, self.sentence_length)

        tokens_in_index = generator.next_batch()
        self.assertEqual(get_tokens_in_word(tokens_in_index),
                         [["<head>", "SCENE", "V"], [".", "Paris", "."]])

        tokens_in_index = generator.next_batch()
        self.assertEqual(get_tokens_in_word(tokens_in_index),
                         [["The", "KING's", "palace"], [".", "</head>", "<blockquote>"]])

        tokens_in_index = generator.next_batch()
        self.assertEqual(get_tokens_in_word(tokens_in_index),
                         [["Enter", "LAFEU", "and"], ["BERTRAM", "</blockquote>", "<a>"]])

        tokens_in_index = generator.next_batch()
        self.assertEqual(get_tokens_in_word(tokens_in_index),
                         [["LAFEU", "</a>", "<blockquote>"], ["But", "I", "hope"]])

    def test_get_next_token(self):
        generator = Generator(self.test_file, self.batch_size, self.sentence_length)

        generator.next_batch()
        next_tokens_in_index = generator.following_tokens()
        self.assertEqual(get_tokens_in_word(next_tokens_in_index),
                         [["SCENE", "V", "."], ["Paris", ".", "The"]])

        generator.next_batch()
        next_tokens_in_index = generator.following_tokens()
        self.assertEqual(get_tokens_in_word(next_tokens_in_index),
                         [["KING's", "palace", "."], ["</head>", "<blockquote>", "Enter"]])

        generator.next_batch()
        next_tokens_in_index = generator.following_tokens()
        self.assertEqual(get_tokens_in_word(next_tokens_in_index),
                         [["LAFEU", "and", "BERTRAM"], ["</blockquote>", "<a>", "LAFEU"]])

        generator.next_batch()
        next_tokens_in_index = generator.following_tokens()
        self.assertEqual(get_tokens_in_word(next_tokens_in_index),
                         [["</a>", "<blockquote>", "But"], ["I", "hope", "your"]])

    def test_determine_has_next(self):
        generator = Generator(self.test_file, self.batch_size, self.sentence_length)

        generator.next_batch()
        self.assertTrue(generator.have_next())

        generator.next_batch()
        self.assertTrue(generator.have_next())

        generator.next_batch()
        self.assertTrue(generator.have_next())

        generator.next_batch()
        self.assertFalse(generator.have_next())
