import torch
from torchvision import transforms, utils
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import albumentations
import albumentations.pytorch

from sklearn.model_selection import KFold
import numpy as np

class AlbumentationsDataset(Dataset):
    """TensorDataset with support of transforms."""
    #def __init__(self, tensors, batchSize=4, shuffle=False, transform=None):
    def __init__(self, tensors, transform=None):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform


    def __len__(self):
        return len(self.tensors[0])


    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        #start_t = time.time()
        #print('transform start !')
        sample = {"image": x, "mask": y}

        if self.transform:
            augmented = self.transform(**sample)
            x = augmented['image']
            y = augmented['mask']

        #total_time = (time.time()-start_t)
        #print('Calculation time: {}'.format(total_time))
        return x, y


# Custom dataset
class BladderToxicityDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform


    def __len__(self):
        return len(self.tensors[0])


    def __getitem__(self, index):
        x = self.tensors[0][index]   # numpy
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x)

        x= torch.from_numpy(x).float()  # tensor
        #y = torch.from_numpy(y).long()
        y = torch.from_numpy(np.asarray(y)).long()
        sample = {'param': x, 'gt':y}
        return sample

class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

# Custom transformation
class AddRandomNoise(object):
	def __init__(self, noise_=0.01):
		self.noise = noise_

	def __call__(self, input_):
		#return input_ + (np.random.rand(input_.size) * 0.01)
		return input_


	def __repr__(self):
		return self.__class__.__name__ + 'CustomRandomNoise'


def get_loader_wjcheon(input_original, output_original, cFold = 1, batch_size=3, num_workers=2):
	# wjcheon

	# K-fold cross validation: K=5
	output_original_length = output_original.__len__()
	rn_output_original = range(0, output_original_length)
	kf5 = KFold(n_splits=5, shuffle=False) # 5 is default

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
	trainingIndex_CV_F = training_indexes.get(trainingDicKey_list[cFold - 1])

	testDicKey_list = list(test_indexes.keys())
	testIndex_CV_F = test_indexes.get(testDicKey_list[cFold - 1])

	input_x = input_original[trainingIndex_CV_F]
	output_gt = output_original[trainingIndex_CV_F]
	input_x_k = input_original[testIndex_CV_F]
	output_gt_k = output_original[testIndex_CV_F]

	print("Data is successfully loaded !!")
	print("Train input:{}, Train gt:{}".format(np.shape(input_x), np.shape(output_gt)))
	print("Test input:{}, Test gt:{}".format(np.shape(input_x_k), np.shape(output_gt_k)))



	albumentations_transform = albumentations.Compose([
		# albumentations.Resize(256, 256),
		# albumentations.RandomCrop(224, 224),
		albumentations.HorizontalFlip(),  # Same with transforms.RandomHorizontalFlip()
		albumentations.Rotate(),
		albumentations.pytorch.transforms.ToTensorV2()
	])

	albumentations_transform_testSet = albumentations.Compose([
		# albumentations.Resize(256, 256),
		# albumentations.RandomCrop(224, 224),
		albumentations.pytorch.transforms.ToTensorV2()
	])

	CustomTransform = transforms.Compose([
		AddRandomNoise(),
		#transforms.ToTensor()
	])

	train_dataset_transform = BladderToxicityDataset(tensors=(input_x, output_gt), transform=CustomTransform)
	train_loader = torch.utils.data.DataLoader(train_dataset_transform, batch_size=batch_size, shuffle=True,
											   num_workers=num_workers)
	# for i, sample_temp in enumerate(train_loader):
	# 	#print(sample_temp)
	# 	print(sample_temp['param'].size())
	# 	print(sample_temp['gt'].size())

	test_dataset_transform = BladderToxicityDataset(tensors=(input_x_k, output_gt_k),
												   transform=CustomTransform)
	test_loader = torch.utils.data.DataLoader(test_dataset_transform, batch_size=1, shuffle=False,
											  num_workers=num_workers)

	# train_dataset_transform = AlbumentationsDataset(tensors=(input_x, output_gt), transform=albumentations_transform)
	# train_loader = torch.utils.data.DataLoader(train_dataset_transform, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	# test_dataset_transform = AlbumentationsDataset(tensors=(input_x_k, output_gt_k), transform=albumentations_transform_testSet)
	# test_loader = torch.utils.data.DataLoader(test_dataset_transform, batch_size=1, shuffle=False, num_workers=num_workers)

	return train_loader, test_loader