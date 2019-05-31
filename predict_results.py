from constants import *
from main import *
from features import utils, filter_symbol, vocab
from tqdm import tqdm

def Gen_test():
    filepath = RAW_TEST_FILEPATH     
    data = list(utils.load_test_raw(filepath))
    #  print(data)
    test_out = open(TEST_INPUT_FILEPATH, "w", encoding="utf-8")
    test_first = 0
    all_len =[]
    for idx, (sentences, label) in tqdm(enumerate(data)):
        cur_len = 0
        sentences = filter_symbol.special_marks(sentences)
        for sentence in sentences:
            cur_len += len(sentence)
        if cur_len < 1: 
          print("Bad lines: ", sentences)
          continue
        all_len.append(cur_len)
        # TODO: comma symbol should be removed ?
        line = " ".join([" ".join(sentence) for sentence in sentences]) + "\t" + str(label)
        if test_first != 0: 
            test_out.write("\n")
        test_out.write(line)
        test_first += 1

    all_len = sorted(all_len, reverse=False)
    print(len(all_len), all_len[:50])
    test_out.close()

def predict_main(filepath):
  print(filepath)
  reader = batch_reader.Batcher(filepath, params)
  sentences, labels, lengths = reader.get_batch(shuffle=False)
  word_voc = vocab.Vocab(WORD_VOC_FILEPATH, MAX_VOCAB_SIZE, MIN_F)
  word_voc_size = word_voc.NumIds()
  params.update(vocab_size=word_voc_size)

  placeholders = add_placeholder()
  inputs = placeholders["inputs"]
  targets = placeholders["labels"]
  is_training = placeholders["is_training"]

  inputs_idx = convert_to_idx(word_voc, inputs)
  model = transformer.Transformer(params, placeholders["is_training"])
  logits = model(inputs_idx, targets)
  logits = tf.nn.softmax(logits)
  
  bn_moving_vars = [g for g in tf.global_variables() if 'moving_mean' in g.name]
  bn_moving_vars += [g for g in tf.global_variables() if 'moving_variance' in g.name]
  saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
  init_op = tf.group([tf.initialize_all_variables(), tf.initialize_local_variables(), tf.tables_initializer()])

  with sess.as_default():
    sess.run(init_op)
    model_file = model_file = tf.train.latest_checkpoint(MODEL_PATH)
    saver.restore(sess, model_file)
    print("Running Predicting")
    predicted_labels, true_labels, all_sentences = [], [], []
    while True:
      try:
        sentences_epoch, label_epoch, length_epoch = sess.run([sentences, labels, lengths])
        print(sentences_epoch.shape)
      except tf.errors.OutOfRangeError:
        break

      ans = sess.run(logits, {
        inputs: sentences_epoch,
        targets: label_epoch,
        is_training: False
      })
      all_sentences.extend(sentences_epoch.tolist())
      predicted_labels.extend(ans.tolist())
      print(len(predicted_labels))
      true_labels.extend(label_epoch.tolist())
  
  predicted_labels = [i[1] for i in predicted_labels]

  return predicted_labels, true_labels, all_sentences

if __name__ == "__main__":
  # Gen_test()
  predict_main(TEST_INPUT_FILEPATH)
