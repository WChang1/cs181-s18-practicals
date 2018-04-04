import numpy as np
import csv
from sklearn.decomposition import PCA
import itertools

# Open file, save as reader object
f = open('train.csv', 'r')
reader = csv.reader(f)

# Get desired columns and rows from csv, 
# Each row is a sublist inside of lst
# Right now it is taking rows 0 to 49
# and columns 0:4
lst = []
for row in itertools.islice(reader, 0, 70):
    lst.append(map(float, row[0:10]))

# Turn into array
array = np.array(lst)
print(array.shape)

# Apply PCA
#
# Documentation: 
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
pca = PCA(n_components=3)
pca_vectors_array = pca.fit_transform(array)