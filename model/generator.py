from utils.config import COMEDY_FULL_TEXT, MAX_LENGTH, SKIP_STEP


class Generator:
    def __init__(self):
        self.__current_index = 0
        with open(COMEDY_FULL_TEXT, 'r') as f:
            full_text = f.read()
            self.full_tokens = full_text.split()
            self.length = len(self.full_tokens)

    def next_sentence(self):
        if self.__current_index + MAX_LENGTH - 1 >= self.length:
            raise IndexError("this is the end of the text")

        sentence = self.full_tokens[self.__current_index:self.__current_index + MAX_LENGTH]
        self.__current_index += SKIP_STEP
        return sentence

    def following_word(self):
        return self.full_tokens[self.__current_index + MAX_LENGTH]
