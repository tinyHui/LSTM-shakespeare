import pickle as pkl
import sys
import re

from utils.config import COMEDY_FULL_TEXT


def get_lines():
    count = 1
    with open(COMEDY_FULL_TEXT, 'r') as f:
        print()
        while True:
            line = f.readline()
            if not line:
                break
            sys.stdout.flush()
            sys.stdout.write(f"\b\rProcessing: {count}")
            count += 1

            yield get_tokens(line)


def get_tokens(line):
    tokens = re.findall(r"</?[a-z]+>|[a-zA-Z]+\'[a-zA-Z]*|[a-zA-Z]+|[0-9]+|[.,;\[\]!?:]+|\'", line)
    return list(map(lambda x: x.strip(), filter(lambda x: x.strip(), tokens)))


if __name__ == '__main__':
    token_set = []

    max_length = -1
    max_length_sentence = []

    length = 0
    sentence = []

    for tokens in get_lines():
        token_set += tokens
        sentence += tokens
        length += len(tokens)
        if length > max_length:
            max_length = length
            max_length_sentence = sentence
            length = 0
            sentence = []

    token_set = set(token_set)
    print("\ntoken number:", len(token_set))
    print("Max sentence length:", max_length, " - ", " ".join(max_length_sentence))

    token_index_map = zip(token_set, range(len(token_set)))
    with open("./bin/token_index.map", 'wb') as f:
        pkl.dump(token_index_map, f, protocol=4)

    index_token_map = zip(range(len(token_set)), token_set)
    with open("./bin/index_token.map", 'wb') as f:
        pkl.dump(index_token_map, f, protocol=4)

