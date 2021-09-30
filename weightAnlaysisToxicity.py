import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
#
from sklearn.model_selection import train_test_split
#
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

# machine learning models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#
from sklearn import set_config
#
import datetime
import copy
import os
from sklearn.model_selection import KFold


#
def plot_parity(model, y_true, y_pred=None, X_to_pred=None, ax=None, **kwargs):
    if not ax:
        fig, ax = plt.subplots(figsize=(5, 5))

    if y_pred is None:
        y_pred = model.predict(X_to_pred)
    ax.scatter(y_true, y_pred, **kwargs)
    xbound = ax.get_xbound()
    xticks = [x for x in ax.get_xticks() if xbound[0] <= x <= xbound[1]]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.0f}" for x in xticks])
    ax.set_yticks(xticks)
    ax.set_yticklabels([f"{x:.0f}" for x in xticks])
    dxbound = 0.05 * (xbound[1] - xbound[0])
    ax.set_xlim(xbound[0] - dxbound, xbound[1] + dxbound)
    ax.set_ylim(xbound[0] - dxbound, xbound[1] + dxbound)

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    ax.text(0.95, 0.1, f"RMSE = {rmse:.2f}\nR2 = {r2:.2f}", transform=ax.transAxes,
            fontsize=14, ha="right", va="bottom", bbox={"boxstyle": "round", "fc": "w", "pad": 0.3})

    ax.grid(True)

    return ax

##

# sns.set_context("talk")
# sns.set_style("white")
#
# # Linux 한글 사용 설정
# plt.rcParams['font.family']=['NanumGothic', 'sans-serif']
# plt.rcParams['axes.unicode_minus'] = False
#
# # # 펭귄 데이터셋 불러오기
# df_peng = sns.load_dataset("penguins")
# df_peng.dropna(inplace=True)
# df_peng.isna().sum()
#
# y = df_peng["body_mass_g"]
# X = df_peng.drop("body_mass_g", axis=1)
# X.head(3)
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Toxicity set 불러오기
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
iter1 =0
print("Fold {} is under the training".format(iter1))

kFoldParameters = 5
currentFold = 1
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
X_test = np.array(input_original[testIndex_CV_F])
y_test = np.array(output_original[testIndex_CV_F])
patientID_val = np.array(patientID[testIndex_CV_F])
y_val_original = copy.deepcopy(y_test)

print("Data is successfully loaded !!")
print("Train input:{}, Train gt:{}".format(np.shape(X_train), np.shape(y_train)))
print("Test input:{}, Test gt:{}".format(np.shape(X_test), np.shape(y_test)))

# Binary classification
y_train[y_train>0]=1
y_test[y_test>0]=1

##
from torch import optim
from torch.optim.lr_scheduler import CyclicLR

import torch
import torch.nn as nn


class RegressorModule(nn.Module):
    def __init__(self, ninput=11, init_weights=True):
        super(RegressorModule, self).__init__()

        self.model = nn.Sequential(nn.Linear(ninput, 16),
                                   nn.ReLU(),
                                   nn.Linear(16, 16),
                                   nn.ReLU(),
                                   nn.Linear(16, 12),
                                   nn.ReLU(),
                                   nn.Linear(12, 8),
                                   nn.ReLU(),
                                   nn.Linear(8, 1),
                                   )
        if init_weights:
            self._initialize_weights()

    def forward(self, X, **kwargs):
        return self.model(X)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
       #model 3
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




X_train_tensor = torch.Tensor(pd.get_dummies(X_train).astype(np.float32).values)
y_train_tensor = torch.Tensor(y_train.astype(np.float32).values)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


#loss_func = RMSELoss()
loss_func = nn.BCELoss()
##
from skorch import NeuralNetRegressor
from skorch import NeuralNetBinaryClassifier
from sklearn.base import BaseEstimator, TransformerMixin


def get_model_T(X_cols, degree=1, method="lr"):
    X_cols_ = deepcopy(X_cols)

    # 1-1.categorical feature에 one-hot encoding 적용
    cat_features = list(set(X_cols) & set(["species", "island", "sex"]))
    cat_transformer = OneHotEncoder(sparse=False, handle_unknown="ignore")

    # 1-2.numerical feature는 Power Transform과 Scaler를 거침
    num_features = list(set(X_cols) - set(cat_features))
    num_features.sort()
    num_transformer = Pipeline(steps=[("polynomial", PolynomialFeatures(degree=degree)),
                                      ("scaler", RobustScaler())
                                      ])

    # 1. 인자 종류별 전처리 적용
    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),
                                                   ("cat", cat_transformer, cat_features)])

    # 2. float64를 float32로 변환
    class FloatTransformer(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, x):
            return np.array(x, dtype=np.float32)

    # 3. 전처리 후 머신러닝 모델 적용
    if method == "lr":
        ml = LinearRegression(fit_intercept=True)
    elif method == "rf":
        ml = RandomForestRegressor()
    elif method == "torch":
        ninput = len(num_features) + 1
        if "species" in cat_features:
            ninput += 3
        if "island" in cat_features:
            ninput += 3
        if "sex" in cat_features:
            ninput += 2

        net = NeuralNetBinaryClassifier(MulticlassClassification(ninput=ninput, init_weights=False),
                                 max_epochs=1000, verbose=0,
                                 warm_start=True,
                                 #                          device='cuda',
                                 criterion=RMSELoss,
                                 optimizer=optim.Adam,
                                 optimizer__lr=0.01
                                 )
        ml = net

    # 3. Pipeline
    model = Pipeline(steps=[("preprocessor", preprocessor),
                            ("float64to32", FloatTransformer()),
                            ("ml", ml)])

    return model


model_T = get_model_T(list(X_train.columns), degree=1, method="torch")
model_T.fit(X_train, y_train.astype(np.float32).values.reshape(-1, 1))
model_T

fig, axs = plt.subplots(ncols=2, figsize=(8, 4), constrained_layout=True, sharey=True)
plot_parity(model_T, y_train, X_to_pred=X_train, ax=axs[0], c="g", s=10, alpha=0.5)
plot_parity(model_T, y_test, X_to_pred=X_test, ax=axs[1], c="m", s=10, alpha=0.5)


##
from sklearn.inspection import permutation_importance

# Neural Network
pi_T = permutation_importance(model_T, X_test, y_test, n_repeats=30, random_state=0)

# 시각화
fig, axs = plt.subplots(ncols=3, figsize=(15, 5), constrained_layout=True, sharey=True)

for ax, pi, title in zip(axs, [pi_0, pi_1, pi_T], ["Linear Reg.", "Random Forest", "Neural Net"]):
    ax.barh(X_test.columns, pi.importances_mean, xerr=pi.importances_std, color="orange")
    ax.invert_yaxis()
    ax.set_xlim(0, )
    ax.set_title(title, fontdict=font_title, pad=16)