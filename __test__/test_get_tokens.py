from unittest import TestCase

from tokens import get_tokens


class TestTokens(TestCase):
    def test_get_tokens(self):
        self.assertEqual(get_tokens("I, with a troop of Florentines, will suddenly surprise him;"),
                         ["I", ",", "with", "a", "troop", "of", "Florentines", ",", "will", "suddenly", "surprise", "him", ";"])
        self.assertEqual(get_tokens("The KING's palace."), ["The", "KING's", "palace", "."]),
        self.assertEqual(get_tokens("I like re. You like re."), ["I", "like", "re", ".", "You", "like", "re", "."])
        self.assertEqual(get_tokens("I like re.You like re."), ["I", "like", "re", ".", "You", "like", "re", "."])
        self.assertEqual(get_tokens("<p> I like re. </p>"), ["<p>", "I", "like", "re", ".", "</p>"])
        self.assertEqual(get_tokens("<p>I like re.</p>"), ["<p>", "I", "like", "re", ".", "</p>"])
        self.assertEqual(get_tokens("A pox on't, let it go"), ["A", "pox", "on't", ",", "let", "it", "go"])
        self.assertEqual(get_tokens("<blockquote> [Aside to BERTRAM]"),
                         ["<blockquote>", "[", "Aside", "to", "BERTRAM", "]"])
        self.assertEqual(get_tokens("O, for the 0 love of laughter"),
                         ["O", ",", "for", "the", "0", "love", "of", "laughter"])
        self.assertEqual(get_tokens("'But a drum'!"),
                         ["'", "But", "a", "drum'", "!"])
        self.assertEqual(get_tokens("'tis pride: but why, why?"),
                         ["'", "tis", "pride", ":", "but", "why", ",", "why", "?"])
        self.assertEqual(get_tokens("All's Well That Ends Well "),
                         ["All's", "Well", "That", "Ends", "Well"])
