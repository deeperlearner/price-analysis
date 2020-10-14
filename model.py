import tensorflow as tf

def RNN_cell(RNN_type, size_layer):
    if RNN_type == 'BasicRNN':
        cell = tf.nn.rnn_cell.BasicRNNCell(size_layer)
    elif RNN_type == 'LSTM':
        cell = tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)
    elif RNN_type == 'GRU':
        cell = tf.nn.rnn_cell.GRUCell(size_layer)
    return cell
