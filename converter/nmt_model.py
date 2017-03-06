import tensorflow as tf
#from tensorflow.nn import seq2seq as tf_seq2seq

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
                             use_lstm=False,
                             num_samples=512,
                             forward_only=False,
                             dtype=tf.float32):

        """
        Initialize the NMT Model
        """

        # since we're dealing with the same language, vocab size should be the same
        # TODO : See if we need to make them different
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size

        self.buckets = buckets
        self.size = size

        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_samples = num_samples
        self.use_lstm = use_lstm
        self.dtype = dtype

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.forward_only = forward_only

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
        assert self.source_vocab_size == len(src_eb_mtrx), 'embedding row size must equal src.|V|'# #row == |V|
        assert self.target_vocab_size == len(dst_eb_mtrx), 'embedding row size msut equal dst.|V|'

        self.src_embedding_mtrx = tf.Variable(src_eb_mtrx, trainable=trainable)
        self.dst_embedding_mtrx = tf.Variable(dst_eb_mtrx, trainable=trainable)

    def define_nmt_cell(self, size):
        """
        Use this function to define the rnn cell type for our NMT Model.
        Args: 
            cell_type: string. ['gru', 'lstm', 'custom']
            size: number of units in each layer of the model
        """
        self.size = size
				
				# Create the internal multi-layer cell for our RNN.
        def single_cell():
            return tf.contrib.rnn.GRUCell(self.size)
        if self.use_lstm:
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(self.size)
        self.cell = single_cell()
        if self.num_layers > 1:
            self.cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])

    def define_nmt_buckets(self, buckets):
        """
        Defines the buckets that we're gonna use for our nmt system. 
        Once the buckets are read, [(5,10), (15, 20), ...] sets the input
        length for encoder/decoders appropriately
        """
        # self.buckets = buckets
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(self.dtype, shape=[None], name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        self.targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]


    def define_loss_func(self, loss_type='sampled'):
        #if loss_type == 'sampled':
        if self.num_samples > 0 and self.num_samples < self.target_vocab_size:
            #assert self.num_samples < self.target_vocab_size, '# samples should be less than |V|'
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, self.size], dtype=self.dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=self.dtype)


            self.output_projection = (w, b)
            # TODO : figure out if we need to use CPU
            def sampled_loss(inputs, labels):
                with tf.device("/cpu:0"):
                    local_w_t = tf.cast(w_t, tf.float32)
                    local_w = tf.cast(w, tf.float32)
                    local_b = tf.cast(b, tf.float32) 
                    local_inputs = tf.cast(inputs, tf.float32)
                    labels = tf.reshape(labels, [-1, 1])
                    local_inputs = tf.cast(inputs, tf.float32)
                    return tf.cast(
                        tf.nn.sampled_softmax_loss(
                                weights=local_w_t,
                                biases=local_b,
                                labels=labels,
                                inputs=inputs,
                                num_sampled=num_samples,
                                num_classes=self.target_vocab_size),
                        self.dtype)
            self.loss_func = sampled_loss
        else:
            assert False, "sampled"


        #self.loss_func = sampled_loss



    def define_nmt_seq_func(self, func_type):
        """
        Defines the sequence2sequence method for our NMT Model.
        The function set by this method takes the following parameters
            [encoder_inputs, decoder_inputs, cell, dtype=dtypes.float32, scope=None]

        TODO : should we use different embeddings for src and dst??
        Args:
            func_type : string, ['basic', 'attention','custom']
        """
        self.func_type = func_type

        if func_type == 'basic':
            print "Configured seq func as basic"
            #self.seq_func = nmt.basic_rnn_seq2seq
        elif func_type == 'attention':
            print "Configured seq func as attention"
            #seq_func = tf_seq2seq.embedding_attention_seq2seq
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(self.encoder_inputs, self.decoder_inputs, self.cell, num_encoder_symbols=self.source_vocab_size, num_decoder_symbols=self.target_vocab_size, embedding_size=self.size, output_projection=self.output_projection, feed_previous=do_decode, dtype=self.dtype)
            self.seq2seq_f = seq2seq_f
        else:
            pass
            # print "Custom"
            # seq_func = attention_seq2seq

            # self.encoder_dim = self.size
            # self.decoder_dim = self.size
            # self.src_embedding_mtrx = None
            # self.dst_embedding_mtrx = None

            # def seq2seq_func(encoder_inputs, decoder_inputs, do_decode):
            #     return seq_func(self.encoder_inputs, self.decoder_inputs, self.cell, 
            #                     self.source_vocab_size, self.target_vocab_size, self.encoder_dim, 
            #                     self.decoder_dim, src_embedding_init=self.src_embedding_mtrx, 
            #                     dst_embedding_init=self.dst_embedding_mtrx, 
            #                     output_projection=None, feed_previous=do_decode)

    def define_train_ops(self):
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
            #import pdb
            #pdb.set_trace()

            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, self.targets,
                    self.target_weights, self.buckets,
                    lambda x, y: self.seq2seq_f(x, y, False),
                    softmax_loss_function=self.loss_func)
        

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(self.buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                        zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())
        # if self.forward_only is True:
        #     print "implement this part when we're ready for results"
        #     if self.func_type == "attention":
        #         self.outputs, self.losses = tf_seq2seq.model_with_buckets(self.encoder_inputs, self.decoder_inputs, targets, self.target_weights, self.buckets, self.target_vocab_size, lambda x, y: self.seq_func(x, y, True), softmax_loss_function=self.loss_func)
        #     # Custom
        #     else:
        #         print "implement this part for custom seq2seq"
        #         self.outputs, self.losses = bucket_model(self.encoder_inputs, self.decoder_inputs, self.targets, self.target_weights, self.buckets, lambda x, y: self.seq_func(x, y, True), softmax_loss_function=self.loss_func)

        # else:
        #     if self.func_type == "attention":
        #         self.outputs, self.losses = tf_seq2seq.model_with_buckets(self.encoder_inputs, self.decoder_inputs, self.targets, self.target_weights, self.buckets, lambda x, y: self.seq_func(x, y, False), softmax_loss_function=self.loss_func)
        #     # Custom
        #     else:
        #         print "implement this part for custom seq2seq"
        #         self.outputs, self.losses = bucket_model(self.encoder_inputs, self.decoder_inputs, self.targets, self.target_weights, self.buckets, lambda x, y: self.seq_func(x, y, False), softmax_loss_function=self.loss_func)


        # # Train op is configured only when we do backprop
        # params = tf.trainable_variables()
        # if self.forward_only is not True:
        #     self.grad_norms = []
        #     self.updates = []
        #     optimizer = tf.train.AdamOptimizer(self.lr)

        #     for b in xrange(len(self.buckets)):
        #         grads = tf.gradients(self.losses[b], params)
        #         clipped_grads, norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        #         self.grad_norms.append(norm)
        #         self.updates.append(optimizer.apply_gradients(zip(clipped_grads, params), global_step=self.global_step))
        # self.saver = tf.train.Saver(tf.global_variables())



    """
    ================================= LIST OF MAIN METHODS =====================
    """

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
        """
        Run a single step of our nmt model.

        Args:
            session : tf session
            encoder_inputs, decoder_inputs : numpy array of token ids
            bucket_id: which bucket to use
        """
            # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket," 
                            " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                            " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                            " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                            self.gradient_norms[bucket_id],  # Gradient norm.
                            self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def get_batch(self, data, bucket_id):
        """
        Get a random batch of data from the specified step.
        """
            # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                    [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
                batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
