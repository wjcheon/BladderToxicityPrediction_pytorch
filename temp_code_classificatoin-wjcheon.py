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

df = pd.read_excel(r"C:\Users\admin\Dropbox\Research\개인연구\23_Brachy_BladderTocixity\GU toxicity_DB_python-balance-2.xlsx",engine='openpyxl')
#df = pd.read_excel(r"C:\Users\admin\Dropbox\Research\개인연구\23_Brachy_BladderTocixity\GU toxicity_DB_python.xlsx",engine='openpyxl')
df.head()
df.sample(frac=1)


#X = df.iloc[:, 1:-1]
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

scaler = StandardScaler()
# X_train = scaler.fit_transform(X_trainval)
# X_val = scaler.transform(X_test)
X_train, y_train = np.array(X_trainval), np.array(y_trainval)
X_val, y_val = np.array(X_test), np.array(y_test)

# Binary classification
y_train[y_train>0]=1
y_val[y_val>0]=1

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


train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

target_list = []
for _, t in train_dataset:
    target_list.append(t)

target_list = torch.tensor(target_list)
target_list = target_list[torch.randperm(len(target_list))]




EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_FEATURES = np.shape(X)[1]
NUM_CLASSES = 2


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)


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


        return x



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



model = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()

        y_train_pred = model(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)

        train_loss.backward()
        optimizer.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

# VALIDATION
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            predSet = []
            gtSet = []

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

                # save the predicted and gt value
                y_pred_softmax = torch.log_softmax(y_val_pred, dim=1)
                _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                y_val_pred_np = y_pred_tags.cpu().detach().numpy()
                y_val_batch_np = y_val_batch.cpu().detach().numpy()
                predSet.append(y_val_pred_np)
                gtSet.append(y_val_batch_np)

    loss_stats['train'].append(train_epoch_loss / len(train_loader))
    loss_stats['val'].append(val_epoch_loss / len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

    print(f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(val_loader):.3f}')

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






