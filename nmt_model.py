import tensorflow as tf


class NMT_Model(object):

    def __init__(self, src_vocab_size, dst_vocab_size, buckets, size,
                 num_layers, max_grad_norm, batch_size, lr, lr_decay, 
                 use_lstm=False, num_samples=512, forward_only=False):

        """
        Initialize the NMT Model
        """

        # since we're dealing with the same language, vocab size should be the same
        assert src_vocab_size == dst_vocab_size, "vocab size should be the same"
        self.src_vocab_size = src_vocab_size
        self.dst_vocab_size = dst_vocab_size

        self.buckets = buckets
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lr = tf.Variable(float(lr), trainable=False)

        # TODO : use sampled softmax?

    def createCell
