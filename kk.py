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

#df = pd.read_excel(r"C:\Users\admin\Dropbox\Research\개인연구\23_Brachy_BladderTocixity\GU toxicity_DB_python-balance-2.xlsx",engine='openpyxl')
df = pd.read_excel(r"C:\Users\admin\Dropbox\Research\개인연구\23_Brachy_BladderTocixity\GU toxicity_DB_python.xlsx",engine='openpyxl')
df.head()
#df.sample(frac=1) # shuffle option turn off


#X = df.iloc[:, 1:-1]
X = df.iloc[:, 12:-1]
y = df.iloc[:, -1]
# Z-score normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_nonZero = X[214:-1]
y_nonZero = np.array(y[214:-1])
y_nonZero = y_nonZero.reshape((np.shape(y_nonZero)[0],1))
df_nonZero = np.concatenate((X_nonZero, y_nonZero), axis=1)
np.random.seed(2021)
df_nonZero = np.take(df_nonZero,np.random.permutation(df_nonZero.shape[0]),axis=0)

x_zero = X[1:214]
y_zero = np.array(df.iloc[1:214, -1])
x_nonZero = df_nonZero[:, 0:-1]
y_nonZero = df_nonZero[:, -1]
y_nonZero[y_nonZero>0]=1
#

kFoldParameters = 5
currentFold =  5
# ZERO
output_original_length_zero = y_zero.__len__()
rn_output_original = range(0, output_original_length_zero)
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

X_train_zero = np.array(x_zero[trainingIndex_CV_F])
y_train_zero = np.array(y_zero[trainingIndex_CV_F])
X_val_zero = np.array(x_zero[testIndex_CV_F])
y_val_zero = np.array(y_zero[testIndex_CV_F])

# NON-ZERO
output_original_length_nonZero = y_nonZero.__len__()
rn_output_original = range(0, output_original_length_nonZero)
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

X_train_nonZero = np.array(x_nonZero[trainingIndex_CV_F])
y_train_nonZero = np.array(y_nonZero[trainingIndex_CV_F])
X_val_nonZero = np.array(x_nonZero[testIndex_CV_F])
y_val_nonZero = np.array(y_nonZero[testIndex_CV_F])


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
         x_data = self.X_data[index] + (torch.rand(self.X_data[index].size()[0]) * 0.1)
         return x_data, self.y_data[index]
        #return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_dataset_zero = ClassifierDataset(torch.from_numpy(X_train_zero).float(), torch.from_numpy(y_train_zero).long())
val_dataset_zero = ClassifierDataset(torch.from_numpy(X_val_zero).float(), torch.from_numpy(y_val_zero).long())
train_dataset_nonZero = ClassifierDataset(torch.from_numpy(X_train_nonZero).float(), torch.from_numpy(y_train_nonZero).long())
val_dataset_nonZero = ClassifierDataset(torch.from_numpy(X_val_nonZero).float(), torch.from_numpy(y_val_nonZero).long())

EPOCHS = 300
BATCH_SIZE_zero = 8
BATCH_SIZE_nonZero = 16

LEARNING_RATE = 0.0001
NUM_FEATURES = np.shape(X)[1]
NUM_CLASSES = 2

train_loader_zero = DataLoader(dataset=train_dataset_zero,
                          batch_size=BATCH_SIZE_zero)
val_loader_zero = DataLoader(dataset=val_dataset_zero, batch_size=1)
train_loader_nonZero = DataLoader(dataset=train_dataset_nonZero,
                          batch_size=BATCH_SIZE_nonZero)
val_loader_nonZero = DataLoader(dataset=val_dataset_nonZero, batch_size=1)




class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        # #model 1
        # self.biasOp = False
        # self.layer_1 = nn.Linear(num_feature, 512, self.biasOp)
        # self.layer_2 = nn.Linear(512, 128, self.biasOp)
        # self.layer_3 = nn.Linear(128, 64, self.biasOp)
        # self.layer_out = nn.Linear(64, num_class, self.biasOp)
        #
        # self.relu = nn.ReLU()
        # self.LeakyReLU = nn.LeakyReLU()
        # self.dropout = nn.Dropout(p=0.2)
        # self.batchnorm1 = nn.BatchNorm1d(512)
        # self.batchnorm2 = nn.BatchNorm1d(128)
        # self.batchnorm3 = nn.BatchNorm1d(64)
        #
        # torch.nn.init.xavier_uniform_(self.layer_1.weight)
        # torch.nn.init.xavier_uniform_(self.layer_2.weight)
        # torch.nn.init.xavier_uniform_(self.layer_3.weight)


        # # model 2
        # self.biasOp = False # False option is fixed.
        # self.layer_1 = nn.Linear(num_feature, 36, self.biasOp)
        # self.layer_2 = nn.Linear(36, 64, self.biasOp)
        # self.layer_3 = nn.Linear(64, 128, self.biasOp)
        # self.layer_4 = nn.Linear(128, 256, self.biasOp)
        # self.layer_5 = nn.Linear(256, 512, self.biasOp)
        # self.layer_6 = nn.Linear(512, 256, self.biasOp)
        # self.layer_7 = nn.Linear(256, 128, self.biasOp)
        # self.layer_8 = nn.Linear(128, 64, self.biasOp)
        # self.layer_9 = nn.Linear(64, 32, self.biasOp)
        # self.layer_out = nn.Linear(32, num_class, self.biasOp)
        #
        # self.relu = nn.ReLU()
        # self.LeakyReLU = nn.LeakyReLU()
        # self.dropout = nn.Dropout(p=0.2)
        # self.batchnorm1 = nn.BatchNorm1d(36)
        # self.batchnorm2 = nn.BatchNorm1d(64)
        # self.batchnorm3 = nn.BatchNorm1d(128)
        # self.batchnorm4 = nn.BatchNorm1d(256)
        # self.batchnorm5 = nn.BatchNorm1d(512)
        # self.batchnorm6 = nn.BatchNorm1d(256)
        # self.batchnorm7 = nn.BatchNorm1d(128)
        # self.batchnorm8 = nn.BatchNorm1d(64)
        # self.batchnorm9 = nn.BatchNorm1d(32)
        #
        # torch.nn.init.xavier_normal_(self.layer_1.weight)
        # torch.nn.init.xavier_normal_(self.layer_2.weight)
        # torch.nn.init.xavier_normal_(self.layer_3.weight)
        # torch.nn.init.xavier_normal_(self.layer_4.weight)
        # torch.nn.init.xavier_normal_(self.layer_5.weight)
        # torch.nn.init.xavier_normal_(self.layer_6.weight)
        # torch.nn.init.xavier_normal_(self.layer_7.weight)
        # torch.nn.init.xavier_normal_(self.layer_8.weight)
        # torch.nn.init.xavier_normal_(self.layer_9.weight)

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

        # kaiming_uniform_
        # xavier_normal_
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_3.weight)
        torch.nn.init.xavier_uniform_(self.layer_4.weight)
        torch.nn.init.xavier_uniform_(self.layer_5.weight)

    def forward(self, x):
        # # model 1
        # x = self.layer_1(x)
        # x = self.batchnorm1(x)
        # x = self.LeakyReLU(x)
        #
        # x = self.layer_2(x)
        # x = self.batchnorm2(x)
        # x = self.LeakyReLU(x)
        # x = self.dropout(x)
        #
        # x = self.layer_3(x)
        # x = self.batchnorm3(x)
        # x = self.LeakyReLU(x)
        # x = self.dropout(x)
        #
        # x = self.layer_out(x)

        # # # model 2
        # x = self.layer_1(x)
        # x = self.batchnorm1(x)
        # x = self.LeakyReLU(x)
        #
        # x = self.layer_2(x)
        # x = self.batchnorm2(x)
        # x = self.LeakyReLU(x)
        # x = self.dropout(x)
        #
        # x = self.layer_3(x)
        # x = self.batchnorm3(x)
        # x = self.LeakyReLU(x)
        # x = self.dropout(x)
        #
        # x = self.layer_4(x)
        # x = self.batchnorm4(x)
        # x = self.LeakyReLU(x)
        # x = self.dropout(x)
        #
        # x = self.layer_5(x)
        # x = self.batchnorm5(x)
        # x = self.LeakyReLU(x)
        # x = self.dropout(x)
        #
        # x = self.layer_6(x)
        # x = self.batchnorm6(x)
        # x = self.LeakyReLU(x)
        # x = self.dropout(x)
        #
        # x = self.layer_7(x)
        # x = self.batchnorm7(x)
        # x = self.LeakyReLU(x)
        # x = self.dropout(x)
        #
        # x = self.layer_8(x)
        # x = self.batchnorm8(x)
        # x = self.LeakyReLU(x)
        # x = self.dropout(x)
        #
        # x = self.layer_9(x)
        # x = self.batchnorm9(x)
        # x = self.LeakyReLU(x)
        # x = self.dropout(x)
        #
        # x = self.layer_out(x)

        # model 3
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


        return x



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



model = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
#optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
print(model)


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}




print("Begin training.")
for e in tqdm(range(1, EPOCHS + 1)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    train_len_counter = 0
    for i, data in enumerate(zip(train_loader_zero, train_loader_nonZero)):

        X_train_batch_zero = data[0][0].to(device)
        y_train_batch_zero = data[0][1].to(device)
        X_train_batch_nonZero = data[1][0].to(device)
        y_train_batch_nonZero = data[1][1].to(device)
        X_train_batch = torch.cat((X_train_batch_zero, X_train_batch_nonZero))
        y_train_batch = torch.cat((y_train_batch_zero, y_train_batch_nonZero))

        optimizer.zero_grad()

        y_train_pred = model(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)

        train_loss.backward()
        optimizer.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        train_len_counter += 1

# VALIDATION
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            predSet = []
            gtSet = []
            val_len_coutner = 0
            model.eval()
            for i, data in enumerate(zip(val_loader_zero, val_loader_nonZero)):
                X_val_batch_zero = data[0][0].to(device)
                y_val_batch_zero = data[0][1].to(device)
                X_val_batch_nonZero = data[1][0].to(device)
                y_val_batch_nonZero = data[1][1].to(device)
                X_val_batch = torch.cat((X_val_batch_zero, X_val_batch_nonZero))
                y_val_batch = torch.cat((y_val_batch_zero, y_val_batch_nonZero))

                #X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
                val_len_coutner += 1

                # save the predicted and gt value
                y_pred_softmax = torch.log_softmax(y_val_pred, dim=1)
                _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                y_val_pred_np = y_pred_tags.cpu().detach().numpy()
                y_val_batch_np = y_val_batch.cpu().detach().numpy()
                predSet.extend(y_val_pred_np)
                gtSet.extend(y_val_batch_np)


    # len_train = len(train_loader_zero) + len(train_loader_nonZero)
    # len_val = len(val_loader_zero) + len(val_loader_nonZero)
    len_train = train_len_counter
    len_val = val_len_coutner
    loss_stats['train'].append(train_epoch_loss / len_train)
    loss_stats['val'].append(val_epoch_loss / len_val)
    accuracy_stats['train'].append(train_epoch_acc / len_train)
    accuracy_stats['val'].append(val_epoch_acc / len_val)

    print(f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len_train:.5f} | Val Loss: {val_epoch_loss / len_val:.5f} | Train Acc: {train_epoch_acc / len_train:.3f}| Val Acc: {val_epoch_acc / len_val:.3f}')

plt.figure()
plt.plot(predSet, 'r')
plt.plot(gtSet)


# Create dataframes
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# Plot the dataframes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
plt.show()