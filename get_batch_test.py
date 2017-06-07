import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
    num_words_per_batch = batch_size * seq_length
    num_batches = len(int_text) // num_words_per_batch

    batches = []
    for batch_num in range(num_batches):
        starting_offset = batch_num * seq_length
        input_batch = [int_text[starting_offset + index * num_batches *seq_length:
                                starting_offset + seq_length + index * num_batches * seq_length]
                                for index in range(batch_size)]
        starting_offset += 1
        targets_batch = [int_text[starting_offset + index * num_batches * seq_length:
                                starting_offset + seq_length + index * num_batches * seq_length]
                                for index in range(batch_size)]
        batch = [input_batch, targets_batch]
        batches.append(batch)
    batches[len(batches)-1][1][batch_size-1][seq_length-1] = int_text[0]

    return np.array(batches)

if __name__ is '__main__':
    with tf.Graph().as_default():
        test_batch_size = 128
        test_seq_length = 5
        test_int_text = list(range(1000 * test_seq_length))
        batches = get_batches(test_int_text, test_batch_size, test_seq_length)

        # Check type
        assert isinstance(batches, np.ndarray), \
            'Batches is not a Numpy array'

        # Check shape
        assert batches.shape == (7, 2, 128, 5), \
            'Batches returned wrong shape.  Found {}'.format(batches.shape)

        for x in range(batches.shape[2]):
            assert np.array_equal(batches[0, 0, x], np.array(range(x * 35, x * 35 + batches.shape[3]))), \
                'Batches returned wrong contents. For example, input sequence {} in the first batch was {}'.format(
                    x, batches[0, 0, x])
            assert np.array_equal(batches[0, 1, x], np.array(range(x * 35 + 1, x * 35 + 1 + batches.shape[3]))), \
                'Batches returned wrong contents. For example, target sequence {} in the first batch was {}'.format(
                    x, batches[0, 1, x])

        last_seq_target = (test_batch_size - 1) * 35 + 31
        last_seq = np.array(range(last_seq_target, last_seq_target + batches.shape[3]))
        last_seq[-1] = batches[0, 0, 0, 0]

        assert np.array_equal(batches[-1, 1, -1], last_seq), \
            'The last target of the last batch should be the first input of the first batch. Found {} but expected {}'.format(
                batches[-1, 1, -1], last_seq)

    _print_success_message()