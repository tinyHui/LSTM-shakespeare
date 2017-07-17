import pickle as pkl
import sys
import re


COMEDY_FULL_TEXT = "./data/full-text.txt"


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

            yield filter(lambda x: x.strip() != "", re.findall(r"\w+|[^a-z0-9]", line.lower()))


if __name__ == '__main__':
    token_set = []

    max_length = -1
    max_length_sentence = []

    length = 0
    sentence = []

    for tokens in get_lines():
        for token in tokens:
            if re.match("^[a-z0-9]+$", token):
                length += 1
                sentence.append(token)
            else:
                if length > max_length:
                    max_length = length
                    max_length_sentence = sentence

                sentence = []
                length = 0

            token_set.append(token)

    token_set = set(token_set)
    print("\ntoken number:", len(token_set))
    print("Max sentence length:", max_length, " - ", " ".join(max_length_sentence))

    token_index_map = zip(token_set, range(len(token_set)))
    with open("./bin/token_index.map", 'wb') as f:
        pkl.dump(token_index_map, f, protocol=4)

    index_token_map = zip(range(len(token_set)), token_set)
    with open("./bin/index_token.map", 'wb') as f:
        pkl.dump(index_token_map, f, protocol=4)

