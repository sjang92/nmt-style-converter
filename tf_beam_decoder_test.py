import tensorflow as tf
from tensorflow.python.platform import test
import numpy as np
from tf_beam_decoder import beam_decoder, BeamSearchHelper

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
# %%

sess = tf.InteractiveSession()

# %%

class MarkovChainCell(core_rnn_cell.RNNCell):
    """
    This cell type is only used for testing the beam decoder.

    It represents a Markov chain characterized by a probability table p(x_t|x_{t-1},x_{t-2}).
    """
    def __init__(self, table):
        """
        table[a,b,c] = p(x_t=c|x_{t-1}=b,x_{t-2}=a)
        """
        assert len(table.shape) == 3 and table.shape[0] == table.shape[1] == table.shape[2]
        with np.errstate(divide='ignore'): # ignore warning for log(0)
            self.log_table = np.log(np.asarray(table, dtype=np.float32))
        self.log_table_var = None
        self._output_size = table.shape[0]

    def __call__(self, inputs, state, scope=None):
        """
        inputs: [batch_size, 1] int tensor
        state: [batch_size, 1] int tensor
        """
        # Simulate variable creation, to ensure scoping works correctly
        log_table = tf.get_variable('log_table',
            shape=(3,3,3),
            dtype=tf.float32,
            initializer=tf.constant_initializer(self.log_table))

        if self.log_table_var is None:
            self.log_table_var = log_table
        else:
            assert self.log_table_var == log_table

        logits = tf.reshape(log_table, [-1, self.output_size])
        indices = state[0] * self.output_size + inputs
        return tf.gather(logits, tf.reshape(indices, [-1])), (inputs,)

    @property
    def state_size(self):
        return (1,)

    @property
    def output_size(self):
        return self._output_size


class BeamSearchTest(test.TestCase):
    def test1(self):
        """
        test correct decode in sequence
        """
        with self.test_session() as sess:
            table = np.array([[[0.0, 0.6, 0.4],
                               [0.0, 0.4, 0.6],
                               [0.0, 0.0, 1.0]]] * 3)

            for cell_transform in ['default', 'flatten', 'replicate']:
                cell = MarkovChainCell(table)
                initial_state = cell.zero_state(1, tf.int32)
                initial_input = initial_state[0]

                with tf.variable_scope('test1_{}'.format(cell_transform)):
                    best_sparse, best_logprobs = beam_decoder(
                        cell=cell,
                        beam_size=7,
                        stop_token=2,
                        initial_state=initial_state,
                        initial_input=initial_input,
                        tokens_to_inputs_fn=lambda x:tf.expand_dims(x, -1),
                        max_len=5,
                        cell_transform=cell_transform,
                        output_dense=False,
                        )

                    tf.variables_initializer([cell.log_table_var]).run()
                    assert all(best_sparse.eval().values == [2])
                    assert np.isclose(np.exp(best_logprobs.eval())[0], 0.4)

    def test2(self):
        """
        test correct intermediate beam states
        """
        with self.test_session() as sess:
            table = np.array([[[0.9, 0.1, 0],
                               [0, 0.9, 0.1],
                               [0, 0, 1.0]]] * 3)

            for cell_transform in ['default', 'flatten', 'replicate']:
                cell = MarkovChainCell(table)
                initial_state = cell.zero_state(1, tf.int32)
                initial_input = initial_state[0]

                with tf.variable_scope('test2_{}'.format(cell_transform)):
                    helper = BeamSearchHelper(
                        cell=cell,
                        beam_size=10,
                        stop_token=2,
                        initial_state=initial_state,
                        initial_input=initial_input,
                        tokens_to_inputs_fn=lambda x:tf.expand_dims(x, -1),
                        max_len=3,
                        cell_transform=cell_transform
                        )

                    _, _, final_loop_state = tf.nn.raw_rnn(helper.cell, helper.loop_fn)
                    _, _, beam_symbols, beam_logprobs = final_loop_state

                tf.variables_initializer([cell.log_table_var]).run()
                candidates, candidate_logprobs = sess.run((beam_symbols, beam_logprobs))

                assert all(candidates[0,:] == [0,0,0])
                assert np.isclose(np.exp(candidate_logprobs[0]), 0.9 * 0.9 * 0.9)
                # Note that these three candidates all have the same score, and the sort order
                # may change in the future
                assert all(candidates[1,:] == [0,0,1])
                assert np.isclose(np.exp(candidate_logprobs[1]), 0.9 * 0.9 * 0.1)
                assert all(candidates[2,:] == [0,1,1])
                assert np.isclose(np.exp(candidate_logprobs[2]), 0.9 * 0.1 * 0.9)
                assert all(candidates[3,:] == [1,1,1])
                assert np.isclose(np.exp(candidate_logprobs[3]), 0.1 * 0.9 * 0.9)
                assert all(np.isclose(np.exp(candidate_logprobs[4:]), 0.0))

    def test3(self):
        """
        test that variable reuse works as expected
        """
        with self.test_session() as sess:
            table = np.array([[[0.0, 0.6, 0.4],
                               [0.0, 0.4, 0.6],
                               [0.0, 0.0, 1.0]]] * 3)

            for cell_transform in ['default', 'flatten', 'replicate']:
                cell = MarkovChainCell(table)
                initial_state = cell.zero_state(1, tf.int32)
                initial_input = initial_state[0]

                with tf.variable_scope('test3_{}'.format(cell_transform)) as scope:
                    best_sparse, best_logprobs = beam_decoder(
                        cell=cell,
                        beam_size=7,
                        stop_token=2,
                        initial_state=initial_state,
                        initial_input=initial_input,
                        tokens_to_inputs_fn=lambda x:tf.expand_dims(x, -1),
                        max_len=5,
                        cell_transform=cell_transform,
                        output_dense=False,
                        scope=scope
                        )

                tf.variables_initializer([cell.log_table_var]).run()

                with tf.variable_scope(scope, reuse=True) as varscope:
                    best_sparse_2, best_logprobs_2 = beam_decoder(
                        cell=cell,
                        beam_size=7,
                        stop_token=2,
                        initial_state=initial_state,
                        initial_input=initial_input,
                        tokens_to_inputs_fn=lambda x:tf.expand_dims(x, -1),
                        max_len=5,
                        cell_transform=cell_transform,
                        output_dense=False,
                        scope=varscope
                        )

                assert all(sess.run(tf.equal(best_sparse.values, best_sparse_2.values)))
                assert np.isclose(*sess.run((best_logprobs, best_logprobs_2)))

    def test4(self):
        """
        test batching, with statically unknown batch size
        """
        with self.test_session() as sess:
            table = np.array([[[0.9, 0.1, 0],
                               [0, 0.9, 0.1],
                               [0, 0, 1.0]]] * 3)

            for cell_transform in ['default', 'flatten', 'replicate']:
                cell = MarkovChainCell(table)
                initial_state = (tf.constant([[2],[0]]),)
                initial_input = initial_state[0]
                initial_input._shape = tf.TensorShape([None, 1])

                with tf.variable_scope('test4_{}'.format(cell_transform)):
                    helper = BeamSearchHelper(
                        cell=cell,
                        beam_size=10,
                        stop_token=2,
                        initial_state=initial_state,
                        initial_input=initial_input,
                        tokens_to_inputs_fn=lambda x:tf.expand_dims(x, -1),
                        max_len=3,
                        cell_transform=cell_transform
                        )

                    _, _, final_loop_state = tf.nn.raw_rnn(helper.cell, helper.loop_fn)
                    _, _, beam_symbols, beam_logprobs = final_loop_state

                tf.variables_initializer([cell.log_table_var]).run()
                candidates, candidate_logprobs = sess.run((beam_symbols, beam_logprobs))

                assert all(candidates[10,:] == [0,0,0])
                assert np.isclose(np.exp(candidate_logprobs[10]), 0.9 * 0.9 * 0.9)
                # Note that these three candidates all have the same score, and the sort order
                # may change in the future
                assert all(candidates[11,:] == [0,0,1])
                assert np.isclose(np.exp(candidate_logprobs[11]), 0.9 * 0.9 * 0.1)
                assert all(candidates[12,:] == [0,1,1])
                assert np.isclose(np.exp(candidate_logprobs[12]), 0.9 * 0.1 * 0.9)
                assert all(candidates[13,:] == [1,1,1])
                assert np.isclose(np.exp(candidate_logprobs[13]), 0.1 * 0.9 * 0.9)
                assert all(np.isclose(np.exp(candidate_logprobs[14:]), 0.0))

if __name__ == '__main__':
    test.main()
