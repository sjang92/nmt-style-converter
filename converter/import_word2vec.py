import gensim
import numpy as np

print "Loading GoogleNews vectors..."

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

print "Loading our vocab file..."

SYMBS = {'_UNK' : 'UNK', '_PAD' : 'PAD', '_EOS': 'EOS', '_GO': 'GO'}
vocabs = {}

vocabs = []
indices = []
counter = 0

with open('./data/all_modern.vocab', 'r') as vocab_file:
    for char in vocab_file:
        if char in SYMBS:
            char = SYMBS[char]
        vocabs.append(char.strip('\n').strip('.'))
        indices.append(counter)
        counter += 1
        #vocabs[char.strip('\n').strip('.')] = len(vocabs.keys()) # ignore new_line

    vocab_file.close()


embedding_matrix = np.zeros(shape = [len(vocabs), 300])

unincluded = []

included_count = 0
counter = 0

# loop through the entire word<->vector map
for word in vocabs:
    if word in model:
        included_count += 1
        vec = model[word]
        embedding_matrix[indices[counter]] = vec
    else :
        unincluded.append(word)
    counter += 1

print embedding_matrix.shape

print included_count
np.save('unincluded.npy', np.array(unincluded))
np.save('w2v.npy', embedding_matrix)
