import tensorflow as tf
from tensorflow.contrib.layers import linear

def decode(decoder_inputs, initial_state, attention_states, cell,
            output_size=None, num_heads=1, loop_function=None, 
            dtype=tf.float32, scope=None, initial_state_attention=False):
    assert decoder_inputs is not None, "we need embedded input vectors"
    if output_size is None: 
        output_size = cell.output_size

    with tf.variable_scope(scope or 'decode function') as scope:
        # batch_size = # rows of our inputs
        batch_size = decoder_inputs.get_shape()[0].value
        attn_length = attention_states.get_shape()[1].value
        attn_size = attension_states.get_shape()[2].value

        hidden = tf.reshape(attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attn_vec_size = attn_size

        for a in xrange(num_heads):
            k = tf.get_variable("AttnW_%d" % a, [1, 1, attn_size, attn_vec_size])
            hidden_features.append(tf.nn.conv2d(hidden, k, [1,1,1,1], "SAME"))
            v.append(tf.get_variable("AttnV_%d" % a, [attn_vec_size]))

        state = initial_state


        def attention(query):
            """
            Put attention masks on hidden using hidden_features and query
            """
            ds = [] # Results of attention reads 
            for a in xrange(num_heads):
                with tf.variable_scope("Attention_%d" % a):
                    y = (query, attn_vec_size, True)
                    y = tf.reshape(y, [-1, 1, 1, attn_vec_size])

                    # Attention mask = softmax of v^T * tanh(..)
                    s = tf.reduce_sum(v[a] * tf.nn.tanh(hidden_features[a] + y), [2,3])
                    a = tf.nn.softmax(s)

                    # Calculate the attention weighted vector 
                    d = tf.reduce_sum(tf.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1,2])
                    ds.append(tf.reshape(d, [-1, attn_size]))

        outputs = []
        prev = None
        batch_attn_size = tf.pack([batch_size, attn_size])
        attns = [tf.zeros(batch_attn_size, dtype=dtype) for _ in xrange(num_heads)]

        for a in attns:
            a.set_shape([None, attn_size])

        if initial_state_attention:
            attns = attention(initial_state) # TODO: implement this func

        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                scope.reuse_variables()

            if loop_function is not None:
                print "we don't support this at this point."

            # Merge input and previous attentions into one vector of the right size.
            # TODO : Does this mean we have to add them up? 
            x = linear([inp] + attns, cell.input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)

            # Run the attention mechanism
            if i == 0 and initial_state_attention:
                # I didn't really understand this part of the code. Look into it
                attns = attention(state)
            else:
                attns = attention(state)

            with tf.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + attns, output_size, True)

            outputs.append(output)

        return outputs, state

def attention_decoder(decoder_inputs, initial_state, attention_states, cell, 
                    num_symbols, num_heads=1, output_size=None, output_projection=None, 
                    feed_previous=False, dtype=tf.float32, scope=None, 
                    initial_state_attention=False, embedding_init=None):

    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        print "output_projection is not none. Check if you want this"

    with tf.variable_scope(scope or 'attention_decoder'):
        # Initialize the word embedding tensor. 
        embedding = tf.get_variable("embedding", shape=[num_symbols, cell.input_size], initializer=embedding_init)

        # For now, only consider feed_previous = False
        assert feed_previous is False, "Only consider feed_previous = False"

        emb_inp = [tf.nn.embedding_lookup(embedding, i) for i in decoder_inputs]
        assert emb_inp.get_shape()[0] == len(decoder_inputs), "# rows of embedded batch must equal num inputs"



def attention_seq2seq(encoder_inputs, decoder_inputs, cell, 
                      num_encoder_symbols, num_decoder_symbols, encoder_dim, decoder_dim,
                      num_heads=1, src_embedding_init=None, dst_embedding_init=None,output_projection=None,feed_previous=False, dtype=tf.float32, scope=None, initial_state_attention=None):
    """
    embedding seq2seq function with attention

    Returns:
        A tuple (outputs, state)
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                     shape [batch_size x num_decoder_symbols] containing the generated
                     outputs.
            state: The state of each decoder cell at the final time-step.
                   It is a 2D Tensor of shape [batch_size x cell.state_size].
    """

    with tf.variable_scope(scope or 'attention_seq2seq'):
        # Run Encoder first
        encoder_cell = tf.nn.rnn_cell.EmbeddingWrapper(cell, num_encoder_symbols, encoder_dim, src_embedding_init)
        encoder_ouputs, encoder_state = tf.nn.rnn(encoder_cell, encoder_inputs, dtype=dtype)

        # Concatenate encoder outputs for attention
        top_states = [tf.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs]
        attention_states = tf.concat(1, top_states)

        # Decoding part
        return attention_decoder(decoder_inputs, encoder_state, attention_states, 
                                cell, num_decoder_symbols, num_heads=num_heads, 
                                output_size=None, output_projection=output_projection, 
                                feed_previous=feed_previous, 
                                initial_state_attention=initial_state_attention, 
                                embedding_init=dst_embedding_init)


def sequence_loss_by_example(logits, targets, weights, softmax_loss_function=None):
    assert len(logits) == len(targets), 'some problem'
    assert len(weights) == len(logits), 'more problem'
    assert softmax_loss_function is not None, 'even though the function accepts None, force it'

    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
        crossent = softmax_loss_function(logit, target)
        log_perp_list.append(crossent * weight)
    log_perps = tf.add_n(log_perp_list)
    total_size = tf.add_n(weights) + 1e-12 # avoid division by 0
    log_perps = log_perps / total_size
    return log_perps

def sequence_loss(logits, targets, weights, softmax_loss_function=None):
    cost = tf.reduce_sum(sequence_loss_by_example(logits, targets, weights, softmax_loss_function=softmax_loss_function))
    batch_size = targets.get_shape()[0].value
    return cost/tf.cast(batch_size, tf.float32)

def bucket_model(encoder_inputs, decoder_inputs, targets, weights, buckets, seq2seq, softmax_loss_function=None, per_example_loss=None, name=None):
    """
    Returns a seq2seq model with bucketing. This should be the only entry into 
    this module from outside. All other functions should never be called 
    explicitly. (attention_seq2seq2) can be passed in as a param
    """

    assert len(encoder_inputs) >= buckets[-1][0], "encoder input length must be longer"
    assert len(targets) >= buckets[-1][1], "same"

    losses = []
    outputs = []
    # TODO : Do we need to define operation scope here?
    for j, bucket in enumerate(buckets):
        with tf.variable_scope(tf.get_variable_scope(), reuse=True if j>0 else None):
            bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]], decoder_inputs[:bucket[1]])
            outputs.append(bucket_outputs)

            losses.append(sequence_loss(outputs[-1], targets[:bucket[1]], weights[:bucket[1]], softmax_loss_function=softmax_loss_funcion))
