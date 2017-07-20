def get_token_index_map():
    with open(TOKEN_INDEX_MAP, 'rb') as f:
        return pkl.load(f)


def get_index_token_map():
    with open(TOKEN_INDEX_MAP, 'rb') as f:
        return pkl.load(f)
