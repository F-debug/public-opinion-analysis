import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

jieba.setLogLevel('WARN')


class DataPreprocess():
    def __init__(self, tokenizer=None,
                 label_set=None):
        self.tokenizer = tokenizer
        self.num_words = None
        self.label_set = label_set
        self.sentence_len = None
        self.word_len = None

    def cut_texts(self, texts=None, word_len=1):

        if word_len > 1:
            texts_cut = [[word for word in jieba.lcut(text) if len(word) >= word_len] for text in texts]
        else:
            texts_cut = [jieba.lcut(one_text) for one_text in texts]

        self.word_len = word_len

        return texts_cut

    def train_tokenizer(self,
                        texts_cut=None,
                        num_words=2000):

        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(texts=texts_cut)
        num_words = min(num_words, len(tokenizer.word_index) + 1)
        self.tokenizer = tokenizer
        self.num_words = num_words

    def text2seq(self,
                 texts_cut,
                 sentence_len=30):

        tokenizer = self.tokenizer
        texts_seq = tokenizer.texts_to_sequences(texts=texts_cut)
        del texts_cut

        texts_pad_seq = pad_sequences(texts_seq,
                                      maxlen=sentence_len,
                                      padding='post',
                                      truncating='post')
        self.sentence_len = sentence_len
        return texts_pad_seq

    def creat_label_set(self, labels):

        label_set = set()
        for i in labels:
            label_set = label_set.union(set(i))

        self.label_set = np.array(list(label_set))

    def creat_label(self, label):

        label_set = self.label_set
        label_zero = np.zeros(len(label_set))
        label_zero[np.in1d(label_set, label)] = 1
        return label_zero

    def creat_labels(self, labels=None):

        label_set = self.label_set
        labels_one_hot = [self.creat_label(label) for label in labels]

        return np.array(labels_one_hot)