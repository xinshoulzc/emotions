from tqdm import tqdm
import tensorflow as tf
import numpy as np

from features import utils, filter_symbol, vocab
from constants import *
from src import batch_reader

def test_batch():
    reader = batch_reader.Batcher(TRAIN_INPUT_FILEPATH, params)
    with tf.Session() as sess:
        batch = sess.run(reader.get_batch())
        print(batch)


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
    filepath = RAW_TRAIN_FILEPATH
    data = list(utils.load_raw(filepath))
    voc = {
        PAD_MARK: MIN_F,
        UNK_MARK: MIN_F,
        QUESTION_MARK: MIN_F,
        EXCLAMATORY_MARK: MIN_F,
        NUMBER_MARK: MIN_F,
        URL_MARK: MIN_F,
        START_MARK: MIN_F,
        END_MARK: MIN_F,
        ENTER_MARK: MIN_F
    }
    cout = open(TRAIN_INPUT_FILEPATH, "w", encoding="utf-8")
    line = ""
    all_len =[]
    for idx, (sentences, label) in tqdm(enumerate(data)):
        cur_len = 0
        sentences = filter_symbol.special_marks(sentences)
        for sentence in sentences:
            cur_len += len(sentence)
            voc = vocab.CreateVocab(sentence, voc)
        if cur_len == 306: print(idx, sentences)
        all_len.append(cur_len)
        # TODO: comma symbol should be removed ?
        line += " ".join([" ".join(sentence) for sentence in sentences]) + "\t" + str(label)
        cout.write(line)
        line = "\n"

    all_len = sorted(all_len, reverse=True)
    print(len(all_len), all_len[:50])
    cout.close()
    vocab.WriteVocab(voc, WORD_VOC_FILEPATH)


if __name__ == "__main__":
    test_main()
    check_word()
    # np.set_printoptions(threshold=np.inf)
    # test_batch()
    # test_vocab()
