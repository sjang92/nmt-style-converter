import tensorflow as tf
#from tensorflow.nn import seq2seq as tf_seq2seq

import nmt_seq2seq as seq2seq
import random
import numpy as np
from six.moves import xrange

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

class NMT_Model(object):

    def __init__(self,
                             source_vocab_size,
                             target_vocab_size,
                             buckets,
                             size,
                             num_layers,
                             max_gradient_norm,
                             batch_size,
                             learning_rate,
                             learning_rate_decay_factor,
                             num_samples=512,
                             forward_only=False,
                             dtype=tf.float32,
                             beam_search=False,
                             beam_size=0):

        """
        Initialize the NMT Model
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size

        self.buckets = buckets
        self.size = size

        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_samples = num_samples
        self.dtype = dtype
        self.beam_search = beam_search

        self.beam_size = beam_size

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.forward_only = forward_only

        self.max_gradient_norm = max_gradient_norm
        self.output_projection=None
        self.loss_func = None

        self.src_embedding_mtrx = None
        self.dst_embedding_mtrx = None

    """
    ====================== LIST OF CONFIG METHODS ==============================
    Use the methods below to configure our nmt model. Since we might want to
    test multiple configurations (cell type, func type, etc.) it would be more
    efficient to be able to change the cell / seq func from outside the object.

    MAKE SURE THSES CONFIG METHODS ARE CALLED IN ORDER

    Methods:
        define_embedding_mtrx
        define_nmt_cell
        define_nmt_buckets
        define_nmt_seq_func
        define_loss_func
        define_train_ops

    """

    def define_embedding_mtrx(self, src_eb_mtrx=None, dst_eb_mtrx=None, trainable=True):
        """
        takes in an initial embedding mtrx. It could be initialized as
        the default one to be used for training. To do so, just pass None
        """
        print(self.source_vocab_size)
        print(len(src_eb_mtrx))
        assert self.source_vocab_size == len(src_eb_mtrx), 'embedding row size must equal src.|V|'# #row == |V|

        self.src_embedding_mtrx = src_eb_mtrx
        self.dst_embedding_mtrx = dst_eb_mtrx


    def define_nmt_cell(self, size):
        """
        Use this function to define the rnn cell type for our NMT Model.
        Args:
            cell_type: string. ['gru', 'lstm', 'custom']
            size: number of units in each layer of the model
        """
        if self.size is None:
            self.size = size

        self.cell = tf.contrib.rnn.LSTMCell(300, num_proj=self.size)

        if self.num_layers > 1:
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell for _ in range(self.num_layers)])

    def define_nmt_buckets(self, buckets):
        """
        Defines the buckets that we're gonna use for our nmt system.
        Once the buckets are read, [(5,10), (15, 20), ...] sets the input
        length for encoder/decoders appropriately
        """
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        for i in xrange(self.buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

        for i in xrange(self.buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(self.dtype, shape=[None], name="weight{0}".format(i)))

    def define_loss_func(self, loss_type='sampled'):
        w_t = tf.get_variable("proj_w", [self.target_vocab_size, self.size], dtype=self.dtype)
        w = tf.transpose(w_t)
        b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=self.dtype)
        self.output_projection = (w, b)

        def sampled_loss(labels, inputs):
            with tf.device("/cpu:0"):
                return tf.cast(
                    tf.nn.sampled_softmax_loss(weights=tf.cast(w_t, tf.float32), biases=tf.cast(b, tf.float32),labels=tf.reshape(labels, [-1, 1]),
                                                inputs=tf.cast(inputs, tf.float32), num_sampled=self.num_samples, num_classes=self.target_vocab_size),
                                                self.dtype)
        self.loss_func = sampled_loss

    def define_nmt_seq_func(self, func_type):
        """
        Defines the sequence2sequence method for our NMT Model.
        """
        print "Configured seq func as attention"
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return seq2seq.seq2seq(encoder_inputs, decoder_inputs, self.cell, num_encoder_symbols=self.source_vocab_size,num_decoder_symbols=self.target_vocab_size,
                    embedding_size=300, output_projection=self.output_projection, feed_previous=do_decode, dtype=self.dtype, beam_search=self.beam_search,
                    target_vocab_size=self.target_vocab_size, beam_size=self.beam_size, encoder_embeddings=self.src_embedding_mtrx,decoder_embeddings=self.dst_embedding_mtrx)
        self.seq2seq_f = seq2seq_f

    def define_train_ops(self):
        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

        if self.forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, self.buckets, lambda x, y: self.seq2seq_f(x, y, True),
                    softmax_loss_function=self.loss_func)

            # If we use output projection, we need to project outputs for decoding.
            if self.output_projection is not None:
                for b in xrange(len(self.buckets)):
                    self.outputs[b] = [
                            tf.matmul(output, self.output_projection[0]) + self.output_projection[1]
                            for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, self.buckets,
                    lambda x, y: self.seq2seq_f(x, y, False),
                    softmax_loss_function=self.loss_func)

        params = tf.trainable_variables()
        if not self.forward_only:
            self.gradient_norms = []
            self.updates = []
            #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            #opt = tf.train.AdamOptimizer(self.learning_rate)

            for b in xrange(len(self.buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

        # Define saver for our parameters
        self.saver = tf.train.Saver(tf.global_variables())


    """
    ================================= LIST OF MAIN METHODS =====================
    """

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):

        encoder_size, decoder_size = self.buckets[bucket_id]

        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            output_feed = [self.updates[bucket_id],
                            self.gradient_norms[bucket_id],
                            self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in xrange(decoder_size): 
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None 
        else:
            return None, outputs[0], outputs[1:]

    def get_batch(self, data, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input +
                                    [PAD_ID] * decoder_pad_size)

        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
