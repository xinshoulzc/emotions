import sys
import time
import tensorflow as tf

from src import transformer
from constants import *
from features import vocab
from src import transformer
from src import batch_reader

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('data_path',)


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
    "inputs": tf.placeholder(tf.string, shape=[None, params["max_length"]], name="inputs"),
    "labels": tf.placeholder(tf.int32, shape=[None], name="labels"),
    "is_training": tf.placeholder(tf.bool, shape=(), name="is_training"),
  }
  return placeholders

def convert_to_idx(voc, inputs):
  # convert to idx
  all_words = list(voc._word_to_id.keys())
  # print("words", len(all_words))
  mapping_strings = tf.constant(all_words, dtype=tf.string)
  vocab_table = tf.contrib.lookup.index_table_from_tensor(
      mapping=mapping_strings, num_oov_buckets=0, default_value=voc.WordToId(UNK_MARK))
  return vocab_table.lookup(inputs)

def predict(logits):
  eval_op = tf.argmax(logits, axis=1)
  return eval_op

def train(logits, targets):
  # adam params
  opt = tf.train.AdamOptimizer(LR, beta1=0.9, beta2=0.999, epsilon=0.1)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=targets, logits=logits, name='cross_entropy')
  
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  updates_op = tf.group(*update_ops)

  with tf.control_dependencies([updates_op]):
      grads = opt.compute_gradients(cross_entropy)

  # whether need clip

  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  loss = tf.reduce_sum(cross_entropy)
  train_op = [opt.apply_gradients(grads), loss]
  return train_op

def get_error(predicted_labels, true_labels, threshold):
  assert len(predicted_labels) == len(true_labels)
  predicted_labels = map(lambda x: 1 if x >= threshold else 0, predicted_labels)
  # print(predicted_labels, true_labels)
  TT, TF, FF, FT = 0, 0, 0, 0
  for p, t in zip(predicted_labels, true_labels):
    if p == t and t == 1: TT += 1
    elif p == t and t == 0: FF += 1
    elif p != t and t == 1: TF += 1
    else: FT += 1
  
  print("recall: ", TT / (TF + TT), " precision: ", (TT + FF) / (TT + TF + FT + FF))

def main():
  train_reader = batch_reader.Batcher(TRAIN_INPUT_FILEPATH, params)
  eval_reader = batch_reader.Batcher(EVAL_INPUT_FILEPATH, params)
  sentences, labels, lengths = train_reader.get_batch()
  word_voc = vocab.Vocab(WORD_VOC_FILEPATH, MAX_VOCAB_SIZE, MIN_F)
  word_voc_size = word_voc.NumIds()
  # update vocab size in params
  params.update(vocab_size=word_voc_size)

  placeholders = add_placeholder()
  inputs = placeholders["inputs"]
  targets = placeholders["labels"]
  is_training = placeholders["is_training"]

  inputs_idx = convert_to_idx(word_voc, inputs)
  model = transformer.Transformer(params, placeholders["is_training"])
  logits = model(inputs_idx, targets)
  train_op = train(logits, targets)
  eval_op = predict(logits)

  bn_moving_vars = [g for g in tf.global_variables() if 'moving_mean' in g.name]
  bn_moving_vars += [g for g in tf.global_variables() if 'moving_variance' in g.name]
  saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars, max_to_keep=3)
  
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
  init_op = tf.group([tf.initialize_all_variables(), tf.initialize_local_variables(), tf.tables_initializer()])
  model_file = os.path.join(MODEL_PATH, MODEL_NAME_00)

  with sess.as_default():
    sess.run(init_op)
    print("Running training")
    epoch = 0
    while epoch < MAX_EPOCH:
      
      # get batch
      try:
        sentences_epoch, label_epoch, length_epoch = sess.run([sentences, labels, lengths])
      except tf.errors.OutOfRangeError:
        # end of files in dataset
        sentences, labels, lengths = train_reader.get_batch()
        sentences_epoch, label_epoch, length_epoch = sess.run([sentences, labels, lengths])

      _, loss = sess.run(train_op, {
          inputs: sentences_epoch,
          targets: label_epoch,
          is_training: True,
      })
      if epoch % 10 == 0: print("epoch : ", epoch, " loss : ", loss)
      epoch += 1
      if epoch % 10 == 0:
        eval_sentences, eval_labels, eval_lengths = eval_reader.get_batch(shuffle=False)
        predicted_labels, true_labels = [], []
        while True:
          try:
              sentences_epoch, label_epoch, length_epoch = sess.run([eval_sentences, eval_labels, eval_lengths])
          except tf.errors.OutOfRangeError:
              break
          ans = sess.run(eval_op, {
            inputs: sentences_epoch,
            targets: label_epoch,
            is_training: False,
          })
          predicted_labels.extend(ans.tolist())
          true_labels.extend(label_epoch.tolist())
        
        get_error(predicted_labels, true_labels, 0.9)
        saver.save(sess, model_file, global_step=epoch)


if __name__ == "__main__":
  main()
