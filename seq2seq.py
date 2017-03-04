import tensorflow as tf



def attention_decoder(decoder_inputs, initial_state, attention_states, cell, num_symbols, num_heads=1, output_size=None, output_projection=None, feed_previous=False, dtype=tf.float32, scope=None, initial_state_attention=False, embedding_init=None):

    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        print "output_projection is not none. Check if you want this"

    with tf.variable_scope(scope or 'attention_decoder')
        embedding = tf.get_variable("embedding", shape=[num_symbols, cell.input_size], initializer=embedding_init)

        # For now, only consider feed_previous = False
        if not feed_previous:
            



def attention_seq2seq(encoder_inputs, decoder_inputs, cell, 
                      num_encoder_symbols, num_decoder_symbols, encoder_dim, decoder_dim,
                      num_heads=1, embedding_init=None, output_projection=None, 
                      feed_previous=False, dtype=tf.float32, scope=None, initial_state_attention=None):
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
        encoder_cell = tf.nn.rnn_cell.EmbeddingWrapper(cell, num_encoder_symbols, encoder_dim, embedding_init)
        encoder_ouputs, encoder_state = tf.nn.rnn(encoder_cell, encoder_inputs, dtype=dtype)

        # Concatenate encoder outputs for attention
        top_states = [tf.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs]
        attention_states = tf.concat(1, top_states)

        # Decoding part
        # TODO : Implement Decoding
        return attention_decoder(decoder_inputs, encoder_state, attention_states, cell, num_decoder_symbols, num_heads=num_heads, output_size=None, output_projection=output_projection, feed_previous=feed_previous, initial_state_attention=initial_state_attention, embedding_init=embedding_init)

