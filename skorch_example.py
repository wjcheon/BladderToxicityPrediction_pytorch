import skorch
import numpy as np
from sklearn.datasets import make_classification
from torch import nn
from skorch import NeuralNetClassifier

X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)   # 1000 x 20
y = y.astype(np.int64) # 1000

class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=nn.ReLU()):
        super(MyModule, self).__init__()
        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.softmax(self.output(X))
        return X

model = NeuralNetClassifier(
    MyModule,
    max_epochs=10,
    lr=0.1,

    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)

model.fit(X, y)

y_proba = model.predict_proba(X)

##
import skorch
import numpy as np
from sklearn.datasets import make_classification
from torch import nn
from skorch import NeuralNetClassifier

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
import openpyxl

import datetime
import copy
import os

# Time for saving data
currentDT = datetime.datetime.now()
# Load data
df = pd.read_excel(r"C:\Users\admin\Dropbox\Research\개인연구\23_Brachy_BladderTocixity\GU toxicity_DB_python-balance-147-age.xlsx",engine='openpyxl') # Best performance data set
#df = pd.read_excel(r"C:\Users\admin\Dropbox\Research\개인연구\23_Brachy_BladderTocixity\GU toxicity_DB_python.xlsx",engine='openpyxl') # Best performance data set
# df = pd.read_excel(r"C:\Users\admin\Dropbox\Research\개인연구\23_Brachy_BladderTocixity\GU toxicity_DB_python-balance.xlsx",engine='openpyxl')
# df = pd.read_excel(r"C:\Users\admin\Dropbox\Research\개인연구\23_Brachy_BladderTocixity\GU toxicity_DB_python.xlsx",engine='openpyxl')
#df = pd.read_excel(r"C:\Users\admin\Dropbox\Research\개인연구\23_Brachy_BladderTocixity\GU toxicity_DB_python_norm-F3.xlsx",engine='openpyxl')
df.head()
df = df.sample(frac=1, random_state=1).reset_index(drop=True) # shffle and index reset
print(df)

input_original = df.iloc[:, 1:-1]
output_original = df.iloc[:, -1]
scaler = StandardScaler()
input_original = scaler.fit_transform(input_original)
patientID = df.iloc[:, 0]
patientID = patientID.to_numpy()
#input_original = input_original.to_numpy()

iter1= 1
print("Fold {} is under the training".format(iter1))
kFoldParameters = 5
currentFold = iter1
output_original_length = output_original.__len__()
rn_output_original = range(0, output_original_length)
kf5 = KFold(n_splits=kFoldParameters, shuffle=False) # 5 is default

training_indexes = {}
test_indexes = {}
counter = 1
for train_index, test_index in kf5.split(rn_output_original):
    training_indexes["trainIndex_CV{0}".format(counter)] = train_index
    test_indexes["testIndex_CV{0}".format(counter)] = test_index
    counter = counter + 1
# print("trainIndex:{}".format(train_index))
# print("testIndex:{}".format(test_index))

trainingDicKey_list = list(training_indexes.keys())
trainingIndex_CV_F = training_indexes.get(trainingDicKey_list[currentFold - 1])

testDicKey_list = list(test_indexes.keys())
testIndex_CV_F = test_indexes.get(testDicKey_list[currentFold - 1])

X_train = np.array(input_original[trainingIndex_CV_F])
y_train = np.array(output_original[trainingIndex_CV_F])
X_val = np.array(input_original[testIndex_CV_F])
y_val = np.array(output_original[testIndex_CV_F])
patientID_val = np.array(patientID[testIndex_CV_F])
y_val_original = copy.deepcopy(y_val)

y_train[y_train>0]=1
y_val[y_val>0]=1

X_train = X_train.astype(np.float32)   # 1000 x 20
y_train = y_train.astype(np.int64) # 1000
X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.int64)



class MulticlassClassification(nn.Module):
    def __init__(self, num_feature=19, num_class=2):
    #def __init__(self, num_units=10, nonlin=nn.ReLU()):
        super(MulticlassClassification, self).__init__()
        # model 3
        self.biasOp = False
        self.layer_1 = nn.Linear(num_feature, 36, self.biasOp)
        self.layer_2 = nn.Linear(36, 72, self.biasOp)
        self.layer_3 = nn.Linear(72, 144, self.biasOp)
        self.layer_4 = nn.Linear(144, 72, self.biasOp)
        self.layer_5 = nn.Linear(72, 36, self.biasOp)
        self.layer_out = nn.Linear(36, num_class, self.biasOp)

        self.relu = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(36)
        self.batchnorm2 = nn.BatchNorm1d(72)
        self.batchnorm3 = nn.BatchNorm1d(144)
        self.batchnorm4 = nn.BatchNorm1d(72)
        self.batchnorm5 = nn.BatchNorm1d(36)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # model 1
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.LeakyReLU(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.LeakyReLU(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.LeakyReLU(x)
        x = self.dropout(x)

        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.LeakyReLU(x)
        x = self.dropout(x)

        x = self.layer_5(x)
        x = self.batchnorm5(x)
        x = self.LeakyReLU(x)
        x = self.dropout(x)

        x = self.layer_out(x)
        x = self.sigmoid(x)

        return x

model = NeuralNetClassifier(
    MulticlassClassification,
    max_epochs=1000,
    lr=0.1,

    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)


# class MyModule(nn.Module):
#     def __init__(self, num_units=10, nonlin=nn.ReLU()):
#         super(MyModule, self).__init__()
#         self.dense0 = nn.Linear(19, num_units)
#         self.nonlin = nonlin
#         self.dropout = nn.Dropout(0.5)
#         self.dense1 = nn.Linear(num_units, num_units)
#         self.output = nn.Linear(num_units, 2)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, X, **kwargs):
#         X = self.nonlin(self.dense0(X))
#         X = self.dropout(X)
#         X = self.nonlin(self.dense1(X))
#         X = self.softmax(self.output(X))
#         return X
#
# model = NeuralNetClassifier(
#     MyModule,
#     max_epochs=1000,
#     lr=0.1,
#
#     # Shuffle training data on each epoch
#     iterator_train__shuffle=True,
# )

model.fit(X_train, y_train)

y_proba = model.predict_proba(X_train)
y_proba

##
from sklearn.inspection import permutation_importance

pi_T = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=0)

# 시각화
fig =  plt.figure()
ax = fig.add_subplot(111)

title = "Neural Net"
font_title = {
    'fontsize': 16,
    'fontweight': 'bold'
}
people = ('Pathology', 'FIGO stage', 'Pelvic LN', 'Para-aortic LN', 'TNM category', 'SCL', 'Tumor size', 'CCRT', 'concurrent chemotherapy regimen',
          'Number of concurrent chemotherapy cycle', 'Adjuvant chemotherapy', 'EBRT total dose EQD2(3) (Gy)', 	'GTV D100 (cGy)',	'BPICRU EQD2(3) (Gy)',
          'BD0.1cc EQD2(3) (Gy)', 'BD1cc EQD2(3) (Gy)' ,'BD2cc EQD2(3) (Gy)', 'BD5cc EQD2(3) (Gy)', 'Age')

y_pos = np.arange(len(people))
ax.barh(y_pos, pi_T.importances_mean, xerr=pi_T.importances_std, color="orange")
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()
ax.set_xlim(0, )
ax.set_title(title, fontdict=font_title, pad=16)