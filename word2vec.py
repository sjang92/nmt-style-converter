import tensorflow as tf
import numpy as np
import colletions
import math


class VectorExtractor(object):

    def __init__(self, vocab_size = 50000, window_size = 5, dimension=300, batch_size=128, num_skips=2, file_name):
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.dimension = dimension
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.file_name = file_name

    def build_dataset(self):
        """
        Reads the given file, and generates a 
        """
        assert self.file_name is not None, "you must set the file name first"

        print "Building the corpus dataset..."

        data = [] 
        f = open(file_name, 'r')
        for line in f:
            data.extend(line.split())

        # Count the number of each words
        counts = [['UNK', -1]]
        counts.extend(collections.Counter(data).most_common(self.vocab_size - 1))

        # Construct dictionary of words along with their unique index
        word_dictionary = dict()
        for word, c in counts:
            word_dictionary[word] = len(word_dictionary)

        words = list()
        unk_count = 0
        for word in data:
            if word in word_dictionary:
                index = word_dictionary[word]
            else:
                  index = 0  # dictionary['UNK']
                  unk_count += 1
            words.append(index)

        counts[0][1] = unk_count
        reverse_dictionary = dict(zip(word_dictionary.values(), word_dictionary.keys()))

        del data # reduce memory usage. Big corpus
        return words, counts, word_dictionary, reverse_dictionary

    def generate_batch(self, words):

        # Initialize
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * self.window_size + 1



    def train(self, file_name):
        f = open(file_name, 'r')

