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

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access

g_log_beam_probs, g_beam_symbols, g_beam_path = [], [], []

def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True,
                              beam_search=False,
                              target_vocab_size=0,
                              beam_size=0):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev, i):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  def beam_search_function(prev, i):
      global g_log_beam_probs
      global g_beam_symbols
      global g_beam_path

      if i == 1:
          g_log_beam_probs, g_beam_symbols, g_beam_path = [], [], []

      log_beam_probs = g_log_beam_probs
      beam_path = g_beam_path
      beam_symbols = g_beam_symbols

      if output_projection is not None:
        prev = tf.nn.xw_plus_b(
        prev, output_projection[0], output_projection[1])

      probs = tf.log(tf.nn.softmax(prev))

      if i > 1:
          probs = tf.reshape(probs + log_beam_probs[-1],
                             [-1, beam_size * target_vocab_size])

      best_probs, indices = tf.nn.top_k(probs, beam_size)
      indices = tf.reshape(indices, [-1, 1])
      best_probs = tf.reshape(best_probs, [-1, 1])

      symbols = indices % target_vocab_size  # Which word in vocabulary.
      beam_parent = indices // target_vocab_size  # Which hypothesis it came from.

      beam_symbols.append(symbols)
      beam_path.append(beam_parent)
      log_beam_probs.append(best_probs)

      symbols = symbols[:,0]

      output = embedding_ops.embedding_lookup(embedding, symbols)
      if not update_embedding:
          output = array_ops.stop_gradient(output)
      return output

  if beam_search:
      return beam_search_function

  return loop_function


def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=2,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False,
                      beam_search=False,
                      beam_size=0):
  """RNN decoder with attention for the sequence-to-sequence model.
  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.
  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: core_rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.
  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
  
  ##########################
  # modify the cell type
  # cell.output_size = d right now. make our decoder cell to output 2d.
  """
  embed_size = cell.output_size
  num_layers = len(initial_state) # initial state is a tuple of size num_layers
  print(embed_size)
  # Each LSTM cell should get prev_state of 2d and input of d
  layer1_cell = tf.contrib.rnn.LSTMCell(embed_size, num_proj=2 * embed_size)
  #layern_cell = tf.contrib.rnn.LSTMCell(2*embed_size)
  cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)])
  cell = core_rnn_cell.EmbeddingWrapper(cell, embedding_classes = num_encoder_symbols, embedding_size = embedding_size)
"""
  ########################


  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if attention_states.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states must be known: %s" %
                     attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(
      scope or "attention_decoder", dtype=dtype) as scope:
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

    def attention(query):
      """Put attention masks on hidden using hidden_features and query."""
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
    if initial_state_attention:
      attns = attention(initial_state)
    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()

      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
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

      #inp = tf.Print(inp, [tf.shape(inp), i, tf.shape(attns[0]), tf.shape(state[0])], "input_dimesnion: ")

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

      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(
            variable_scope.get_variable_scope(), reuse=True):
          attns = attention(state)
      else:
        attns = attention(state)

      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + attns, output_size, True)
      if loop_function is not None:
        prev = output
      outputs.append(output)  # 3 x |output_size|

  if beam_search:
    beam_outputs = []
    # first choose the beam path with highest score

    col = g_log_beam_probs[-1] # last column
    max_row = tf.reshape(tf.to_int32(tf.argmax(col, axis=0)), shape=())

    print(max_row)


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
    print(len(outputs))

  return outputs, state

def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False,
                                beam_search=False,
                                target_vocab_size=0,
                                beam_size=0):
  """RNN decoder with embedding and attention and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: core_rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_size: Size of the output vectors; if None, use output_size.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    dtype: The dtype to use for the RNN initial states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  if output_size is None:
    output_size = cell.output_size
  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with variable_scope.variable_scope(
      scope or "embedding_attention_decoder", dtype=dtype) as scope:

    embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous, beam_search, target_vocab_size, beam_size) if feed_previous else None
    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs
    ]
    return attention_decoder(
        emb_inp,
        initial_state,
        attention_states,
        cell,
        output_size=output_size,
        num_heads=num_heads,
        loop_function=loop_function,
        initial_state_attention=initial_state_attention,
        beam_search=beam_search,
        beam_size=beam_size)

def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False,
                                encoder_embeddings=None,
                                decoder_embeddings=None,
                                beam_search=False,
                                target_vocab_size=0,
                                beam_size=0):  # CHANGES - adding embeddings for input/output
  """
  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: core_rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial RNN state (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_seq2seq".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  # CHANGE

  # EMBEDDINGS ARE NP MATRICES
  if encoder_embeddings:
    assert num_encoder_symbols == encoder_embeddings.shape[0]
  if decoder_embeddings:
    assert num_decoder_symbols == decoder_embeddings.shape[0]
  
  with variable_scope.variable_scope(scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype

    # Make sure all variables are as we want them to be
    if encoder_embeddings:
      assert embedding_size == encoder_embeddings.shape[1], "embedding size is wrong..."

    # Wrap EMBEDDING AS A TENSOR
    """
    # DEFINE LAYER1 CELLS FOR BIDIRECTIONAL LSTM
    lstm_layer_1 = tf.contrib.rnn.BasicLSTMCell(embedding_size) # forget bias = 1.0 by default
    lstm_layer_2 = tf.contrib.rnn.LSTMCell(2 * embedding_size, num_proj=embedding_size) # starting layer2, we get double the size
    lstm_layer_3 = tf.contrib.rnn.BasicLSTMCell(embedding_size, state_is_tuple=True)

    cell = tf.contrib.rnn.BasicLSTMCell(embedding_size)
    #cell = tf.contrib.rnn.MultiRNNCell([cell, cell, cell, cell])

    # Look up embedding vectors for each input. 
    if encoder_embeddings:
      print("xx")
    # If embedding matrix is not provided, train that as well. 
    else: 
      # Only embed the inputs to the first layer. 
      lstm_layer_1 = core_rnn_cell.EmbeddingWrapper(lstm_layer_1, embedding_classes=num_encoder_symbols, embedding_size=embedding_size)

    # Run Layer 1 : Bidirectional RNN
    layer1_outputs, layer1_fw_state, layer1_bw_state = core_rnn.static_bidirectional_rnn(lstm_layer_1, lstm_layer_1, encoder_inputs, dtype=dtype)

    # Run Layer 2 : Uni directional RNN 
    layer2_outputs, layer2_state = core_rnn.static_rnn(lstm_layer_2, layer1_outputs, dtype=dtype)

    # Run Layer 3 : 
    layer3_outputs, layer3_state = core_rnn.static_rnn(lstm_layer_3, layer2_outputs, dtype=dtype)

    # TEST
    #test_cell = tf.contrib.rnn.LSTMCell(2 * embedding_size, num_proj = )
    """
    #############################################################
    if encoder_embeddings:
      print("should not come here. we're not feeding encoder_embeddings")
    else:
      encoder_cell = core_rnn_cell.EmbeddingWrapper(cell, embedding_classes = num_encoder_symbols, embedding_size = embedding_size)

    encoder_outputs, fw_state, bw_state = core_rnn.static_bidirectional_rnn(encoder_cell, encoder_cell, encoder_inputs, dtype=dtype)

    encoder_state = []
    for i in range(0, len(fw_state)):
      fw_c, fw_m = fw_state[i]
      bw_c, bw_m = bw_state[i]
      concat_c = tf.concat([fw_c, bw_c], axis=1)
      concat_m = tf.concat([fw_m, bw_m], axis=1)

      concat_state = tf.contrib.rnn.LSTMStateTuple(concat_c, concat_m)
      # each state at each layer is shape (batch_size, embedding_size), so concat at axis=1
      encoder_state.append(concat_state)

    encoder_state = tuple(encoder_state)
    assert len(encoder_state) == len(fw_state), "length should be the same after concat"""
    #############################################################

    #encoder_outputs = layer3_outputs # length of this should equal the bucket[0]
    #encoder_state = layer3_state # This should be a single tensor

    # First calculate a concatenation of encoder outputs to put attention on.
    # IMPORTANT : since we're using bidirectional, 2*embeddingsize
    top_states = [
        array_ops.reshape(e, [-1, 1, 2*embedding_size]) for e in encoder_outputs
    ]
    attention_states = array_ops.concat(top_states, 1)

    #encoder_state = encoder_outputs[-1]
    encoder_state = encoder_state # only pass in the last state for each layer

    # Decoder.
    output_size = None
    if output_projection is None:
        # TODO : let decoding language be 2d
      cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
      output_size = num_decoder_symbols
    cell = tf.contrib.rnn.LSTMCell(embedding_size * 2, num_proj=embedding_size*2, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * len(encoder_state))


    return embedding_attention_decoder(
        decoder_inputs,
         encoder_state,
         attention_states,
         cell,
         num_decoder_symbols,
         embedding_size*2,
         num_heads=num_heads,
         output_size=output_size,
         output_projection=output_projection,
         feed_previous=feed_previous,
         initial_state_attention=initial_state_attention,
         beam_search=beam_search,
         target_vocab_size=target_vocab_size,
         beam_size=beam_size)

def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (labels-batch, inputs-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                      logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logit)
      else:
        crossent = softmax_loss_function(target, logit)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.name_scope(name, "sequence_loss", logits + targets + weights):
    cost = math_ops.reduce_sum(
        sequence_loss_by_example(
            logits,
            targets,
            weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, cost.dtype)
    else:
      return cost

def model_with_buckets(encoder_inputs,
                       decoder_inputs,
                       targets,
                       weights,
                       buckets,
                       seq2seq,
                       softmax_loss_function=None,
                       per_example_loss=False,
                       name=None): 
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(
      x, y, core_rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be
      a scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to "model_with_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors. The shape of output tensors can be either
        [batch_size x output_size] or [batch_size x num_decoder_symbols]
        depending on the seq2seq model used.
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputsut, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(
              sequence_loss_by_example(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function))
        else:
          losses.append(
              sequence_loss(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function))

  return outputs, losses
