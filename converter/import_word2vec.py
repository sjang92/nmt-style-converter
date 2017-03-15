import gensim
import numpy as np

print "Loading GoogleNews vectors..."

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

print "Loading our vocab file..."

vocabs = {}

with open('./data/all_modern.vocab', 'r') as vocab_file:
    for char in vocab_file:
        vocabs[filter(str.isalpha, char)] = len(vocabs.keys())

    vocab_file.close()


embedding_matrix = np.zeros(shape = [len(vocabs.keys()), 300])

unincluded = []

included_count = 0

# loop through the entire word<->vector map
for word in vocabs.keys():
    if word in model:
        included_count += 1
        vec = model[word]
        embedding_matrix[vocabs[word]] = vec
    else :
        unincluded.append(word)


print included_count
np.save('unincluded.npy', np.array(unincluded))
np.save('w2v.npy', embedding_matrix)
