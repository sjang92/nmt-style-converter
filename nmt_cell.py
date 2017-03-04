import tensorflow as tf

class Custom_Cell(object):
    def __init__(self, size):
        """
        This is an empty capsule for our custom cells. Only use when we're trying
        to define our own RNN cell. Otherwise, specify cell_type="gru" or "lstm"
        when initializing the NMT_Cell_Generator
        """

class NMT_Cell_Generator(object):
    def __init__(self, cell_type, num_layers=1, size):
        """
        cell_type = gru, lstm, custom
        size = number of units in each layer of the model
        """
        self.cell_type=cell_type
        self.num_layers = num_layers
        self.size = size

    def get_cell(self):
        if self.cell_type == "gru":
            cell = tf.nn.rnn_cell.GRUCell(self.size)
        elif self.cell_type == "lstm":
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        else:
            cell = Custom_Cell(self.size)

        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers)

        return cell
