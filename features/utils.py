import os
import codecs
from tqdm import tqdm
from functools import wraps
from constants import *

def load_raw(filepath, cnt=None):
    with open(filepath, "rb") as cin:
        next(cin)
        cur = b""
        for idx, line in tqdm(enumerate(cin)):
            if cnt is not None and idx >= int(cnt): break
            cur += line
            if len(line) > 8:
                if line[-9:-1] == b"Positive":
                    cur = cur.split(b',')[1:-1]
                    cur = [line.decode("utf-8") for line in cur]
                    yield (cur, 1)
                    cur = b""
                elif line[-9:-1] == b"Negative":
                    cur = cur.split(b',')[1:-1]
                    cur = [line.decode("utf-8") for line in cur]
                    yield (cur, 0)
                    cur = b""

def load_test_raw(filepath, cnt=None):
    cur_idx = 2
    with open(filepath, "rb") as cin:
        next(cin)
        cur = b""
        for idx, line in tqdm(enumerate(cin)):
            if cnt is not None and idx >= int(cnt): break
            # print(str(cur_idx).encode("utf-8") + b",")
            if line.startswith(str(cur_idx).encode("utf-8") + b","):
                cur = cur.split(b',')[1:]
                cur = [line.decode("utf-8") for line in cur]
                yield (cur, 0)
                cur = b""
                cur_idx += 1
            cur += line
        if len(cur) != 0:
            cur = cur.split(b',')[1:]
            cur = [line.decode("utf-8") for line in cur]
            yield (cur, 0)
                    

def support_list(func):
    @wraps(func)
    def wrapper(x):
        if isinstance(x, list):
            ret = [func(i) for i in x]
        else: ret = func(x)
        return ret
    return wrapper

def main():
    filepath = RAW_TRAIN_FILEPATH
    data = list(load_raw(filepath))
    print(data)
    # data = [(x[0].decode("utf-8"), y.decode("utf-8")) for (x,y) in data]
    # print(data)

if __name__ == "__main__":
    main()