# Run to get LSTM predictions

import numpy as np
import csv
from sklearn.decomposition import PCA
import itertools
from librosa.display import specshow
from librosa.feature import mfcc
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

# Open file, save as reader object
f = open('train.csv', 'r')
reader = csv.reader(f)

print('READING IN DATA')
# Read in the data, save dataframe to x array
# Save true y values to y array
x = np.zeros((1, 88200))
y = []
for row in reader:
    y.append(int(float(row[-1])))
    x = np.append(
        x, 
        np.array(map(float, row[:-1])).reshape((1, 88200)), 
        axis = 0
    )

x = np.delete(x, 0, 0)
y = to_categorical(y)

# input_shape is (batch_size, timesteps, input_dim)
data_dim = 88200
timesteps = 1
num_classes = 10

x = x.reshape((n, timesteps, data_dim))

print('FITTING LSTM')
def lstm_f():
    model = Sequential()
    model.add(LSTM(100, return_sequences = True, input_shape = (timesteps, data_dim)))
    model.add(LSTM(100, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(100))  # return a single vector of dimension 32
    model.add(Dense(10, activation = 'softmax'))
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = 'adam', 
        metrics = ['accuracy']
    )
    return model

estimator = KerasClassifier(build_fn = lstm_f, epochs = 20, verbose = 1)
estimator.fit(x, y)

def write_predictions(predictions, ids, outfile):
    """
    assumes len(predictions) == len(ids), and that predictions[i] is the
    index of the predicted class with the malware_classes list above for 
    the executable corresponding to ids[i].
    outfile will be overwritten
    """
    with open(outfile,"w+") as f:
        # write header
        f.write("Id,Prediction\n")
        for i, history_id in enumerate(ids):
            f.write("%s,%d\n" % (history_id, predictions[i]))

# Read in test data
del x
del y

print('READING IN TEST DATA')

f = open('test.csv', 'r')
reader = csv.reader(f)

xtest = np.zeros((1, 88200))
for row in reader:
    xtest = np.append(
        xtest, 
        np.array(map(float, row[1:])).reshape((1, 88200)), 
        axis = 0
    )

xtest = np.delete(xtest, 0, 0)

xtest = xtest.reshape(1000, 1, 88200)
preds = estimator.predict(xtest)
write_predictions(preds, range(1000), 'lstm_preds.csv')