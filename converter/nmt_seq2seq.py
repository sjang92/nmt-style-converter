from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

import tensorflow as tf

# Apparently we need to import this function since it's going to be deprecated
linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access

g_log_beam_probs, g_beam_path = [], []

def get_symbol_lookup_function(embedding,
                               output_projection=None,
                               update_embedding=True,
                               beam_search=False,
                               target_vocab_size=0,
                               beam_size=0):

  def lookup_function(prev, i):

    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])

    prev_symbol = math_ops.argmax(prev, 1)
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  def beam_search_function(prev, i):
      global g_log_beam_probs
      global g_beam_path

      if i == 1:
          g_log_beam_probs, g_beam_path = [], []

      log_beam_probs = g_log_beam_probs
      beam_path = g_beam_path

      if output_projection is not None:
        prev = tf.matmul(prev, output_projection[0]) + output_projection[1]

      probs = tf.log(tf.nn.softmax(prev))

      if i > 1:
          probs = tf.reshape(probs + log_beam_probs[-1],
                             [-1, beam_size * target_vocab_size])

      best_probs, indices = tf.nn.top_k(probs, beam_size)
      indices = tf.reshape(indices, [-1, 1])
      best_probs = tf.reshape(best_probs, [-1, 1])


      symbols = indices % target_vocab_size
      beam_prev = indices // target_vocab_size

      beam_path.append(beam_prev)
      log_beam_probs.append(best_probs)

      symbols = symbols[:,0]

      output = embedding_ops.embedding_lookup(embedding, symbols)
      if not update_embedding:
          output = array_ops.stop_gradient(output)
      return output

  if beam_search:
      return beam_search_function

  return lookup_function


def decode_with_attention(decoder_inputs,initial_state,attention_states,cell,output_size=None,
                          num_heads=2,symbol_lookup=None,dtype=None,scope=None,beam_search=False,
                          beam_size=0):

  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(
      scope or "decode_with_attention", dtype=dtype) as scope:
    dtype = scope.dtype

    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    if attn_length is None:
      attn_length = array_ops.shape(attention_states)[1]
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = array_ops.reshape(attention_states,
                               [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    for a in xrange(num_heads):
      k = variable_scope.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size])
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(
          variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

    state = initial_state

    # We adopted this from Tensorflow's seq2seq attention example. 
    def attention(query):
      ds = []  # Results of attention reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(query_list, 1)
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):
          y = linear(query, attention_vec_size, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                  [2, 3])
          a = nn_ops.softmax(s)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))
      return ds

    outputs = []
    prev = None
    batch_attn_size = array_ops.stack([batch_size, attn_size])
    attns = [
        array_ops.zeros(
            batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
        ]

    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])
  
    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()

      # If loop_function is set, we use it instead of decoder_inputs.
      if symbol_lookup is not None and prev is not None:
        with variable_scope.variable_scope("symbol_lookup", reuse=True):
          inp = symbol_lookup(prev, i)
          if i == 1 and beam_search:
            attns = [tf.concat([attns[j]] * beam_size, axis=0) for j in xrange(num_heads)]
            new_state = []
            for j in xrange(len(state)):
              c, m = state[j]
              c = tf.concat([c] * beam_size, axis=0)
              m = tf.concat([m] * beam_size, axis=0)
              concat = tf.contrib.rnn.LSTMStateTuple(c, m)
              new_state.append(concat)
            state = new_state


      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)

      x = linear([inp] + attns, input_size, True)
      # Run the RNN.

      cell_output, state = cell(x, state)
      # Run the attention mechanism.

      if i == 1 and beam_search:
          attns = [array_ops.zeros(batch_attn_size * beam_size, dtype=dtype) for _ in xrange(num_heads)]

      attns = attention(state)

      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + attns, output_size, True)
      if symbol_lookup is not None:
        prev = output
      outputs.append(output)  # 3 x |output_size|

  if beam_search:
    beam_outputs = []
    # first choose the beam path with highest score

    col = g_log_beam_probs[-1] # last column
    max_row = tf.reshape(tf.to_int32(tf.argmax(col, axis=0)), shape=())


    # Handle last output first
    beam_outputs.append(tf.gather(outputs[-1], [max_row]))

    for i, path_tensor in enumerate(g_beam_path[::-1]):
        # first set max_row'th elem of the 

        val = tf.reshape(path_tensor[max_row], shape=()) # extract scalar

        output_idx = len(outputs) - i - 2
        true_output = tf.gather(outputs[output_idx], [val])
        beam_outputs.append(true_output)

        max_row = val

    outputs = beam_outputs[::-1]
  return outputs, state

def embedding_decode_with_attention(decoder_inputs, initial_state, attention_states, cell, num_symbols, embedding_size, num_layers=1,
                                    num_heads=1, output_size=None, output_projection=None, feed_previous=False, update_embedding_for_previous=True,
                                    dtype=None, scope=None, beam_search=False, target_vocab_size=0, beam_size=0, 
                                    embedding=None):

  # Set up cells for the decoder
  cell = tf.contrib.rnn.LSTMCell(embedding_size, num_proj=output_size, state_is_tuple=True)
  cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

  # Set another variable scope for this function
  with variable_scope.variable_scope(scope or "embedding_decode_with_attention", dtype=dtype) as scope:

    # Wrap our data with embedding matrices
    embedding = variable_scope.get_variable("embedding", shape=[num_symbols, 300], initializer=tf.constant_initializer(embedding), dtype=dtype, trainable=False)
    
    symbol_lookup = get_symbol_lookup_function(embedding, output_projection,update_embedding_for_previous, beam_search, target_vocab_size, beam_size) if feed_previous else None
    return decode_with_attention(
        [embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs],
        initial_state,
        attention_states,
        cell,
        output_size=output_size,
        num_heads=num_heads,
        symbol_lookup=symbol_lookup,
        beam_search=beam_search,
        beam_size=beam_size)

def seq2seq(encoder_inputs, decoder_inputs, cell, num_encoder_symbols, num_decoder_symbols, embedding_size, num_heads=1, output_projection=None,
            feed_previous=False, dtype=None, scope=None, encoder_embeddings=None, decoder_embeddings=None,
            beam_search=False, target_vocab_size=0, beam_size=0):  # CHANGES - adding embeddings for input/output

  # Set all variable scope to this function to avoid name collision.
  with variable_scope.variable_scope(scope or "seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype

    if encoder_embeddings is not None:
      print("feeding encoder_embeddings")
      embedding = variable_scope.get_variable("embedding", [num_encoder_symbols, embedding_size], initializer = tf.constant_initializer(encoder_embeddings), dtype=dtype, trainable=False)

      inputs = []

      for inp in encoder_inputs:
        curr_input = tf.nn.embedding_lookup(embedding, inp)
        inputs.append(curr_input)

      encoder_inputs = inputs

      print(encoder_inputs)
      encoder_cell = cell

    # If no embedding matrix is given, we need to create it and train it as a parameter
    else:
      encoder_cell = core_rnn_cell.EmbeddingWrapper(cell, embedding_classes = num_encoder_symbols, embedding_size = embedding_size)

    # code block for running uni-directional LSTM
    #encoder_outputs, encoder_state = core_rnn.static_rnn(encoder_cell, encdoer_inputs, dtype=dtype)

    # run bidirectional LSTM as encoder
    encoder_outputs, fw_state, bw_state = core_rnn.static_bidirectional_rnn(encoder_cell, encoder_cell, encoder_inputs, dtype=dtype)
    encoder_state = fw_state

    # Calculate attention candidates from the bidirectional LSTM outputs per timestep
    attention_candidates = [array_ops.reshape(opt, [-1, 1, 2 * encoder_cell.output_size]) for opt in encoder_outputs]
    attention_states = array_ops.concat(attention_candidates, 1)

    # prepare parameters for decoder
    output_size = cell.output_size

    return embedding_decode_with_attention(decoder_inputs,encoder_state, attention_states, None, num_decoder_symbols, embedding_size,
                                            num_layers=len(encoder_state), num_heads=num_heads, output_size=output_size,
                                            output_projection=output_projection, feed_previous=feed_previous,
                                            beam_search=beam_search, target_vocab_size=target_vocab_size, beam_size=beam_size,embedding=decoder_embeddings)
