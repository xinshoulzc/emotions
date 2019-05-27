import sys
import time
import tensorflow as tf

from src import transformer
from constants import *
from features import vocab

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path',)


def model_fn(features, labels, mode, params):
  """Defines how to train, evaluate and predict from the transformer model."""
  with tf.variable_scope("model"):
    # inputs [batch_size, input_length] 
    # targets [batch_size]
    inputs, targets = features, labels

    # Create model and get output logits.
    model = transformer.Transformer(params, mode == tf.estimator.ModeKeys.TRAIN)

    logits = model(inputs, targets)

    # When in prediction mode, the labels/targets is None. The model output
    # is the prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
      if params["use_tpu"]:
        raise NotImplementedError("Prediction is not yet supported on TPUs.")
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=logits,
          export_outputs={
              "translate": tf.estimator.export.PredictOutput(logits)
          })

def add_placeholder():
  placeholders = {
    "inputs": tf.placeholder(tf.int32, shape=[None, params["max_length"]], name="inputs")
    "labels": tf.placeholder(tf.int32, shape=[None], name="labels")
    "is_training": tf.placeholder(tf.bool, shape=(), name="is_training")
  }
  return placeholders

def convert_to_idx(voc, inputs):
  # convert to idx
  all_words = voc._word_to_id.keys()
  mapping_strings = tf.constant(all_words, dtype=tf.string)
  vocab_table = tf.contrib.lookup.index_table_from_tensor(
      mapping=mapping_strings, num_oov_buckets=1, default_value=voc.WordToId(UNK_MARK))
  return vocab_table.lookup(inputs)

def train(model):
  # adam optimizer
  # support multi gpus
  pass

def main():
  reader = batch_reader.Batcher(TRAIN_INPUT_FILEPATH, params)
  word_voc = vocab.Vocab(WORD_VOC_FILEPATH, MAX_VOCAB_SIZE, MIN_F)
  word_voc_size = word_voc.NumIds()
  # update vocab size in params
  params.update(vocab_size=word_voc_size)

  # need checking
  placeholders = add_placeholder()
  inputs = convert_to_idx(word_voc, placeholders["inputs"])
  model = transformer.Transformer(params, placeholders["is_training"])
  logits = model(inputs, targets)
  


if __name__ == "__main__":
  main()