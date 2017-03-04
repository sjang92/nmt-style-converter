import tensorflow as tf

import NMT_Cell_Generator from nmt_cell


class NMT_Model(object):

    def __init__(self, src_vocab_size, dst_vocab_size, buckets,
                 num_layers, max_grad_norm, batch_size, lr, lr_decay, 
                 use_lstm=True, num_samples=512, forward_only=False):

        """
        Initialize the NMT Model
        """

        # since we're dealing with the same language, vocab size should be the same
        # TODO : See if we need to make them different
        assert src_vocab_size == dst_vocab_size, "vocab size should be the same"
        self.src_vocab_size = src_vocab_size
        self.dst_vocab_size = dst_vocab_size

        self.buckets = buckets
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lr = tf.Variable(float(lr), trainable=False)

        # TODO : use sampled softmax?


    """
    ====================== LIST OF CONFIG METHODS ==============================
    Use the methods below to configure our nmt model. Since we might want to 
    test multiple configurations (cell type, func type, etc.) it would be more
    efficient to be able to change the cell / seq func from outside the object

    Methods:
        define_nmt_cell
        define_nmt_seq_func
    """

    def define_embedding_mtrx(self, src_eb_mtrx=None, dst_eb_mtrx=None, trainable=True):
        """
        takes in an initial embedding mtrx. It could be initialized as 
        the default one to be used for training. To do so, just pass None
        """
        assert self.src_vocab_size == len(src_eb_mtrx), 'embedding row size must equal src.|V|'# #row == |V|
        assert self.dst_vocab_size == len(dst_eb_mtrx), 'embedding row size msut equal dst.|V|'

        self.src_embedding_mtrx = tf.Variable(src_eb_mtrx, trainable=trainable)
        self.dst_embedding_mtrx = tf.Variable(dst_eb_mtrx, trainable=trainable)

    def define_nmt_cell(self, cell_type, size):
        """
        Use this function to define the rnn cell type for our NMT Model.
        Args: 
            cell_type: string. ['gru', 'lstm', 'custom']
            size: number of units in each layer of the model
        """
        self.size = size
        self.cell = NMT_Cell_Generator(cell_type, self.num_layers, size)

    def define_nmt_seq_func(self, func_type):
        """
        Defines the sequence2sequence method for our NMT Model.
        The function set by this method takes the following parameters
            [encoder_inputs, decoder_inputs, cell, dtype=dtypes.float32, scope=None]

        TODO : should we use different embeddings for src and dst??
        Args:
            func_type : string, ['basic', 'attention','custom']
        """
        if func_type == 'basic':
            print "Configured seq func as basic"
            self.seq_func = nmt.basic_rnn_seq2seq
        elif func_type == 'attention':
            print "Configured seq func as attention"
        else:

