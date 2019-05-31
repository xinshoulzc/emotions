import tensorflow as tf

from constants import *

class Batcher(object):
    def __init__(self, data_paths, hps):
        self._data_paths = data_paths
        self._hps = hps
        self._dataset = self._read_dataset()

    def _read_dataset(self):
        dataset = tf.data.TextLineDataset(self._data_paths)

        # split string
        dataset = dataset.map(lambda x: (
            tf.split(tf.string_split([x], delimiter=b'\t').values, 2)
        ))
        dataset = dataset.map(lambda x, y: (
            tf.string_split(x, delimiter=b' ').values[:self._hps["max_length"]], 
            tf.string_to_number(y, out_type=tf.int32)
        ))
        dataset = dataset.map(lambda x, y:(
            x, y, tf.shape(x)[-1]
        ))
        # convert to idx
        # all_words = self._vocab._word_to_id.keys()
        # all_words = [word.encode("utf-8") for word in all_words]
        # mapping_strings = tf.constant(all_words, dtype=tf.string)
        # vocab_table = tf.contrib.lookup.index_table_from_tensor(
        #     mapping=mapping_strings, num_oov_buckets=1, default_value=self._vocab.WordToId(UNKNOWN_TOKEN))
        # dataset = dataset.map(lambda x, y:(vocab_table.lookup(x), y))

        # pad to a specific size
        dataset = dataset.map(lambda x, y, l:(
            tf.pad([x], [[0, 0], [0, self._hps["max_length"] - l]], constant_values=PAD_MARK.encode("utf-8")), y, l
        ))

        # squeeze x
        dataset = dataset.map(lambda x, y, l:(
            tf.squeeze(x), tf.squeeze(y), l
        ))

        return dataset
    
    def get_batch(self, shuffle=True):
        if shuffle:
            dataset = self._dataset.shuffle(buffer_size=self._hps["default_batch_size"]*500)
        else:
            dataset = self._dataset
        dataset = dataset.batch(self._hps["default_batch_size"])
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        return next_element