import os
from src import model_params

# filepath
RAW_DIR = os.path.join("data", "raw")
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "20190520_test.csv"
RAW_TRAIN_FILEPATH = os.path.join(RAW_DIR, TRAIN_FILENAME)
WORD_VOC_FILEPATH = os.path.join("data", "train_voc")

TRAIN_INPUT_FILEPATH = os.path.join("data", "train", "train")

ENC_LEN = 100
# sample size: 6328
# top_len50: [306, 201, 185, 177, 163, 148, 142, 138, 128, 122, 115, 110, 109, 106, 98, 96, 93, 92, 91, 90, 90, 89, 88, 87, 85, 85, 84, 84, 83, 82, 82, 81, 81, 80, 80, 80, 79, 79, 79, 78, 78, 76, 76, 76, 76, 76, 75, 75, 74, 74]

# marks
QUESTION_MARK = "<Q>"
EXCLAMATORY_MARK = "<E>"
NUMBER_MARK = "<N>"
URL_MARK = "<URL>"
UNK_MARK = "<UNK>"
START_MARK = "<s>"
END_MARK = "</s>"
ENTER_MARK = "<ENTER>" # \n
PAD_MARK = "<PAD>"

FILTER_MARKS = ". : \" \' | \\ [ ] ( ) * @ # $ % ^ & + = \n ” “ \t".split(" ") + [" "]

params = model_params.TINY_PARAMS

# adding transformer params
params.update(max_length=ENC_LEN)
# params.update(default_batch_size=4)

# minimal frequence of words, words' frequence < MIN_F are treated unk
MIN_F = 2

# max frequence of words, words' frequence must < MIN_F
MAX_F = 1000000

# max vocab size
MAX_VOCAB_SIZE = 10000

# params in ml

# max epoch
MAX_EPOCH = 1000
# learning rate
LR = 1e-3

# print(FILTER_MARKS)
