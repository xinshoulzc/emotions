from tqdm import tqdm
import tensorflow as tf
import numpy as np
import codecs

from features import utils, filter_symbol, vocab
from constants import *
from src import batch_reader
from main import *
from predict_results import *

def test_batch():
    reader = batch_reader.Batcher(TRAIN_INPUT_FILEPATH, params)
    sentence, label, length = reader.get_batch()
    with tf.Session() as sess:
        for idx, x in enumerate(range(2000)):
            try:
                sentence_, label_, length_ = sess.run([sentence, label, length])
            except tf.errors.OutOfRangeError:
                sentence, label, length = reader.get_batch()
            print(idx, sentence_.shape, label_.shape)
        # print(sentence.dtype)
        # print(batch, len(batch))


def test_vocab():
    voc = vocab.Vocab(WORD_VOC_FILEPATH, MAX_VOCAB_SIZE, MIN_F)
    print(voc._word_to_id)
    print(voc._id_to_word)
    # pass

def check_word():
    voc = vocab.Vocab(WORD_VOC_FILEPATH, MAX_VOCAB_SIZE, MIN_F)
    for w in voc._word_to_id.keys():
        w = w.decode("utf-8")
        ans = [c for c in w if not ('A' <= c <= 'Z' or 'a' <= c <= 'z')]
        if len(ans) > 0: print(ans)


def test_main():
    reader = batch_reader.Batcher(TRAIN_INPUT_FILEPATH, params)
    sentences, labels, lengths = reader.get_batch()
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
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    init_op = tf.group([tf.initialize_all_variables(), tf.initialize_local_variables(), tf.tables_initializer()])

    with sess.as_default():
        sess.run(init_op)
        print("Running training")
        epoch = 0
        while epoch < MAX_EPOCH:
            print("epoch", epoch)
            sentences_epoch, label_epoch, length_epoch = sess.run([sentences, labels, lengths])
            sess.run(train_op, {
                inputs: sentences_epoch,
                targets: label_epoch,
                is_training: True,
            })
            epoch += 1
            if epoch % 10 == 0:
                sentences_epoch, label_epoch, length_epoch = sess.run([sentences, labels, lengths])
                sess.run(eval_op, {
                    inputs: sentences_epoch,
                    targets: label_epoch,
                    is_training: False,
                })

def Gen_voc():
    filepath = RAW_TRAIN_FILEPATH
    data = list(utils.load_raw(filepath))
    np.random.seed(SEED)
    voc = {
        PAD_MARK: MAX_F,
        UNK_MARK: MIN_F,
        QUESTION_MARK: MIN_F,
        EXCLAMATORY_MARK: MIN_F,
        NUMBER_MARK: MIN_F,
        URL_MARK: MIN_F,
        START_MARK: MIN_F,
        END_MARK: MIN_F,
        ENTER_MARK: MIN_F
    }
    train_cout = open(TRAIN_INPUT_FILEPATH, "w", encoding="utf-8")
    eval_cout = open(EVAL_INPUT_FILEPATH, "w", encoding="utf-8")
    train_first, eval_first = 0, 0
    all_len =[]
    for idx, (sentences, label) in tqdm(enumerate(data)):
        cur_len = 0
        is_training = np.random.randint(0, 9) != 0
        sentences = filter_symbol.special_marks(sentences)
        for sentence in sentences:
            cur_len += len(sentence)
            if is_training: voc = vocab.CreateVocab(sentence, voc)
        if cur_len < 1: continue
        all_len.append(cur_len)
        # TODO: comma symbol should be removed ?
        line = " ".join([" ".join(sentence) for sentence in sentences]) + "\t" + str(label)
        if is_training:
            if train_first != 0: 
                train_cout.write("\n")
            train_cout.write(line)
            train_first += 1
        else:
            if eval_first != 0: 
                eval_cout.write("\n")
            eval_cout.write(line)
            eval_first += 1

    all_len = sorted(all_len, reverse=False)
    print(len(all_len), all_len[:50])
    train_cout.close()
    eval_cout.close()
    vocab.WriteVocab(voc, WORD_VOC_FILEPATH)

def test_predict_main(output_filepath):
    data = predict_main(TEST_INPUT_FILEPATH)
    cout = codecs.open(output_filepath, "w", encoding="utf-8")
    for idx, line in enumerate(data):
        cout.write(str(idx + 1) + "," + "%.8f" % line)
        if data[-1] != line: cout.write("\n")
    cout.close()

if __name__ == "__main__":
    # test_main()
    # Gen_voc()
    # Gen_test()
    # check_word()
    # np.set_printoptions(threshold=np.inf)
    # test_batch()
    # test_vocab()
    test_predict_main(PREDICT_FILEPATH)
