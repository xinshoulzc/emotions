import sys
import codecs
from constants import *

class Vocab(object):

  def __init__(self, vocab_file, max_size, min_f):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0

    with open(vocab_file, 'rb') as vocab_f:
      for line in vocab_f:
        pieces = line.rstrip().split(b"\t")
        if len(pieces) != 2:
          sys.stderr.write('Bad line: %s\n' % line.decode("utf-8"))
          continue
        if int(pieces[1]) < min_f: continue
        if pieces[0] in self._word_to_id:
          raise ValueError('Duplicated word: %s.' % pieces[0].decode("utf-8"))
        self._word_to_id[pieces[0]] = self._count
        self._id_to_word[self._count] = pieces[0]
        self._count += 1
        if self._count > max_size:
          raise ValueError('Too many words: >%d.' % max_size)
    
    # padding value must be 0 which may be same as the default padding value in transformer 
    assert self._word_to_id[PAD_MARK.encode("utf-8")] == 0

  def CheckVocab(self, word):
    if word not in self._word_to_id:
      return None
    return self._word_to_id[word]
  
  def WordToId(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNK_MARK.encode("utf-8")]
    return self._word_to_id[word]

  def IdToWord(self, word_id):
    if word_id not in self._id_to_word:
      raise ValueError('id not found in vocab: %d.' % word_id)
    return self._id_to_word[word_id]

  def NumIds(self):
    return self._count

def Pad(ids, pad_id, length):
  assert pad_id is not None
  assert length is not None

  if len(ids) < length:
    a = [pad_id] * (length - len(ids))
    return ids + a
  else:
    return ids[:length]

def CreateVocab(sentence, words=None):
    assert isinstance(sentence, list) 

    if words is None: words = {}
    for w in sentence:
        words[w] = words.get(w, 0) + 1
    
    return words

def WriteVocab(v, out_file):
    assert isinstance(v, dict)

    v = sorted(v.items(), key=lambda x: x[1], reverse=True)
    with open(out_file, "w", encoding="utf-8") as cout:
        for word, f in v:
            cout.write(word + "\t" + str(f) + "\n")     
