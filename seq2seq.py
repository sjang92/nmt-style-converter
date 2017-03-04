import tensorflow as tf

def attention_seq2seq(encoder_inputs, decoder_inputs, cell, 
                      num_encoder_symbols, num_decoder_symbols, encoder_dim,
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

