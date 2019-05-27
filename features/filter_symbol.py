import os
import tqdm
import re
import numpy as np
import traceback
import emoji
from constants import *
from features.utils import *

num_pattern = re.compile(r"[-+]?[0-9]\d*\.\d+|[-+]?\d+")
url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

@support_list
def special_marks(sentence):
    # nums question exclamatory emoji filter_marks
    ans = []
    # s = sentence
    s = sentence.lower()
    # url > nums == english > symbol
    url_span = [i.span() for i in url_pattern.finditer(s)]
    nums_span = [i.span() for i in num_pattern.finditer(s)]
    num_cur, english_cur, url_cur, cur, pre = 0, 0, 0, 0, 0

    while cur < len(s):

        # nums may be defined as http marks
        while url_cur < len(url_span) and cur > url_span[url_cur][0]: url_cur += 1
        while num_cur < len(nums_span) and cur > nums_span[num_cur][0]: num_cur += 1

        if url_cur < len(url_span) and cur == url_span[url_cur][0]:
            ans.append(URL_MARK)
            cur += url_span[url_cur][1] - url_span[url_cur][0]
            url_cur += 1
            pre = cur
        elif num_cur < len(nums_span) and cur == nums_span[num_cur][0]:
            ans.append(NUMBER_MARK)
            cur += nums_span[num_cur][1] - nums_span[num_cur][0]
            num_cur += 1
            pre = cur
        else:
            if s[cur] in emoji.UNICODE_EMOJI or s[cur] == "?" or s[cur] == "!":
                if cur - pre > 0:
                    ans.append(s[pre: cur])
                    pre = cur
                if s[cur] in emoji.UNICODE_EMOJI: ans.append(s[cur: cur + 1])
                elif s[cur] == "?": ans.append(QUESTION_MARK)
                elif s[cur] == "!": ans.append(EXCLAMATORY_MARK)
                cur += 1
                pre = cur
            elif s[cur] in FILTER_MARKS:
                if cur - pre > 0: ans.append(s[pre: cur])
                cur += 1
                pre = cur
            else: cur += 1

    if cur > pre: ans.append(s[pre: cur])
    return ans

def main():
    pass

if __name__ == "__main__":
    main()
