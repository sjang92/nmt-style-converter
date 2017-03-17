import numpy as np

MTRX_FILE_NAME = "all_embed.npy"
ALL_TOKENS_FILE_NAME = "./data/all.snt.tokens"
SRC_TOKENS_FILE_NAME = "./data/all_modern.snt.aligned.tokens"
DST_TOKENS_FILE_NAME = "./data/all_original.snt.aligned.tokens"
SRC_EMBED_FILE_NAME = "modern.npy"
DST_EMBED_FILE_NAME = "original.npy"
DIMENSION = 300

def load_tokens(file_name):
    tokens = []
    with open(file_name, 'r') as f:
        for line in f:
            tokens.append(line.strip('\n'))

        f.close()
    return tokens

def list_to_dict(arr):
    dictionary = dict()
    for word in arr:
        dictionary[word] = len(dictionary.keys())

    return dictionary

# Step 1 : load combined embedding matrix and tokens
all_embeddings = np.load(MTRX_FILE_NAME)
all_tokens = load_tokens(ALL_TOKENS_FILE_NAME)
all_dictionary = list_to_dict(all_tokens)

# Step 2 : load src and dst tokens
src_tokens = load_tokens(SRC_TOKENS_FILE_NAME)
dst_tokens = load_tokens(DST_TOKENS_FILE_NAME)
src_dictionary = list_to_dict(src_tokens)
dst_dictionary = list_to_dict(dst_tokens)
src_vocab_size = len(src_tokens)
dst_vocab_size = len(dst_tokens)

# Step 3 : create src / dst embedding mtrx 
src_embedding = np.zeros([src_vocab_size, DIMENSION])
dst_embedding = np.zeros([dst_vocab_size, DIMENSION])

not_in_src = []
not_in_dst = []

for word in src_tokens:
    src_idx = src_dictionary[word]
    if word in all_dictionary:
        all_idx = all_dictionary[word]
        src_embedding[src_idx] = all_embeddings[all_idx]
    else:
        not_in_src.append(word)

for word in dst_tokens:
    dst_idx = dst_dictionary[word]
    if word in  all_dictionary:
        all_idx = all_dictionary[word]
        dst_embedding[dst_idx] = all_embeddings[all_idx]
    else:
        not_in_dst.append(word)

print len(not_in_src)
print len(not_in_dst)
np.save("not_in_src.npy", not_in_src)
np.save("not_in_dst.npy", not_in_dst)
np.save(SRC_EMBED_FILE_NAME, src_embedding)
np.save(DST_EMBED_FILE_NAME, dst_embedding)


