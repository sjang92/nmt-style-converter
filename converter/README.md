data: put training / vocab / speech data here.
vocab.from = put src language tokens here. Each token is put in a single line. Line number = token-id

vocab.to = same. 

NOTE : _PAD / _GO / _EOS / _UNK = token_ids 1 ~ 4


speeches = make this file aligned. Each line should correspond to each other

We referred to some of the tutorials from tensorflow to write the codes for word2vec_extractor.py and nmt_seq2seq.py,
and the link to the tutorials are here:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
https://github.com/tensorflow/models/tree/master/tutorials/rnn/translate

We also referred to an example here to implement beam search in nmt_seq2seq.py
https://github.com/tensorflow/tensorflow/issues/654