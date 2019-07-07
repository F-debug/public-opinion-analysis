import json
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import numpy as np

sess = tf.InteractiveSession()

data = pd.read_csv("Ebusiness.csv", encoding='utf-8')
x = data['evaluation']
y = [[i] for i in data['label']]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


from TextClassification import TextClassification

clf = TextClassification()
texts_seq, texts_labels = clf.get_preprocess(x_train, y_train,
                                             word_len=1,
                                             num_words=2000,
                                             sentence_len=50)
clf.fit(texts_seq=texts_seq,
        texts_labels=texts_labels,
        output_type=data_type,
        epochs=10,
        batch_size=64,
        model=None)


with open('./%s.pkl' % data_type, 'wb') as f:
    pickle.dump(clf, f)




with open('./%s.pkl' % data_type, 'rb') as f:
    clf = pickle.load(f)
y_predict = clf.predict(x_test)
y_predict = [[clf.preprocess.label_set[i.argmax()]] for i in y_predict]
score = sum(y_predict == np.array(y_test)) / len(y_test)
print(score)