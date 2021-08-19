import os
import time
import torch
import logging
import random
from network import FCN
from torch import optim
import torch.nn.functional as F
import numpy as np

timestr = time.strftime("%Y%m%d-%H%M%S")
filenameLog = os.path.join('./Logs','TrainingLog-'+timestr+'.txt')

logging.basicConfig(filename=filenameLog, filemode='a',level=logging.DEBUG, format='%(asctime)s %(msecs)d- %(process)d-%(levelname)s - %(message)s')


class Solver(object):
    def __init__(self, config, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Models
        self.FCN = None
        self.input_ch = config.input_ch
        self.output_ch =config.output_ch
        self.optimizer = None


        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.augmentation_prob = config.augmentation_prob

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step
        self.validation_period = config.validation_period

        self.mode = config.mode
        self.cuda_id = config.cuda_idx
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(f'cuda:{self.cuda_id}' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            print("GPU: {}".format(torch.cuda.get_device_name(self.cuda_id)), "/  Index: {}".format(self.cuda_id))

        self.build_model()

        # Path
        self.model_path = config.model_path  # model save path
        self.val_datapath = os.path.join(config.val_datapath,
                                        'model_BladderToxicity_CV{0}_AugProb_{1:.3f}-'.format(self.num_epochs,
                                                                                  self.augmentation_prob) \
                                        + timestr)
        if not os.path.exists(self.val_datapath):
            os.makedirs(self.val_datapath)
            print(self.val_datapath)


    def build_model(self):

        self.FCN = FCN(input_ch=self.input_ch, output_ch=self.output_ch )
        self.optimizer = optim.Adam(list(self.FCN.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        self.FCN.to(self.device)
        self.print_network(self.FCN, 'FCN')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def multi_acc(self, y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)

        acc = torch.round(acc * 100)

        return acc

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#
        print('Starting the training process !!!')
        # Check and Load U-Net for Train
        fcn_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (
        "FCN", self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
        self.lossFunction = torch.nn.CrossEntropyLoss()


        accuracy_stats = {
            'train': [],
            "val": []
        }
        loss_stats = {
            'train': [],
            "val": []
        }
        if os.path.isfile(fcn_path):
            # 	# Load the pretrained Encoder
            # 	#self.unet.load_state_dict(torch.load(unet_path))
            # 	#print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
            print('hi')
        else:
            # Train for Encoder
            lr = self.lr
            totalEpoch = 0
            self.FCN.train(True)


            for epoch in range(self.num_epochs):

                totalEpoch = totalEpoch + 1
                train_epoch_loss = 0
                train_epoch_acc = 0
                predSet =[]
                gtSet = []
                for i, sample_temp in enumerate(self.train_loader):
                    input_temp = sample_temp['param']
                    gt_temp = sample_temp['gt']
                    input_temp = input_temp.to(self.device)
                    gt_temp = gt_temp.to(self.device)

                    self.optimizer.zero_grad()
                    output = self.FCN(input_temp)  # SR : Segmentation Result


                    train_loss = self.lossFunction(output, gt_temp)
                    train_acc = self.multi_acc(output, gt_temp)
                    train_epoch_loss += train_loss.item()
                    train_epoch_acc += train_acc.item()

                    train_loss.backward()
                    self.optimizer.step()

                    # predSet.extend(output.detach().cpu().numpy())
                    # gtSet.extend(gt_temp.detach().cpu().numpy())


                    #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    #trainCorrect += pred.eq(gt_temp.view_as(pred)).sum().item()
                trainingAvgLoss = train_epoch_loss/ len(self.train_loader)
                trainingAvgAcc = train_epoch_acc / len(self.train_loader)
                print("@Train, Epoch: {0}, avgLoss: {1:.3f}\n".format(epoch, trainingAvgLoss))
                print("@Train, Epoch: {0}, avgACC: {1:.3f}\n".format(epoch, trainingAvgAcc))
                # print("@Predicted: {}".format(predSet))
                # print("@GT: {}".format(gtSet))
                # logging.info("Epoch: {0} --- Predicted {1}".format(epoch, predSet))
                # logging.info("Epoch: {0} --- gt        {1}".format(epoch, gtSet))

                # print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                #     trainingAvgLoss, trainCorrect, len(self.train_loader.dataset),
                #     100. * trainCorrect / len(self.train_loader.dataset)))

                # Decay learning rate
                if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print('Decay learning rate to lr: {}.'.format(lr))
                # ================================ Online-Validation =================================#
                # ====================================================================================#
                # if epoch ==0.0:
                #     continue
                # if np.mod(epoch, self.val_step) == 0.0:
                #     self.FCN.eval()
                #     test_loss = 0
                #     correct = 0
                #     gtSet =[]
                #     predSet = []
                #     with torch.no_grad():
                #         for i, sample_temp in enumerate(self.test_loader):
                #             input_temp = sample_temp['param']
                #             input_temp = input_temp.to(self.device)
                #
                #             gt_temp = sample_temp['gt']
                #             gtSet.append(gt_temp.cpu().numpy())
                #             gt_temp = gt_temp.to(self.device)
                #
                #             self.optimizer.zero_grad()
                #             output = self.FCN(input_temp.float())  # SR : Segmentation Result
                #
                #             test_loss += F.nll_loss(output, gt_temp.long(), reduction='sum').item()  # sum up batch loss
                #             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                #             predSet.append(pred.cpu().numpy())
                #             correct += pred.eq(gt_temp.view_as(pred)).sum().item()
                #
                #     test_loss /= len(self.test_loader.dataset)
                #
                #     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                #         test_loss, correct, len(self.test_loader.dataset),
                #         100. * correct / len(self.test_loader.dataset)))
                #     #print("Pred: {}".format(predSet))
                #     #print("GT: {}".format(gtSet))


    def test(self):
        self.FCN.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for i, sample_temp in enumerate(self.test_loader):
                input_temp = sample_temp['param']
                input_temp = input_temp.to(self.device)

                gt_temp = sample_temp['gt']
                gt_temp = gt_temp.to(self.device)

                self.optimizer.zero_grad()
                output = self.FCN(input_temp.float())  # SR : Segmentation Result

                test_loss += F.nll_loss(output, gt_temp, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(gt_temp.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))