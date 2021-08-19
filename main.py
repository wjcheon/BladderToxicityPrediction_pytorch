# conda activate toxicity
import pandas as pd
from data_loader import get_loader_wjcheon
from solver import Solver
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import argparse
import numpy as np

def main(config):
    # Config modification


    # Data preparation
    Data = pd.read_excel(config.data_path, engine='openpyxl')
    inputParameters = Data.iloc[:, 1:-1]
    gtParameters = Data.iloc[:, -1]

    #inputParameters = np.float(np.asanyarray(inputParameters))
    inputParameters = np.asanyarray(inputParameters)
    gtParameters = np.asanyarray(gtParameters, dtype=np.int)


    scaler = StandardScaler()
    print(scaler.fit(inputParameters))
    inputParametersZscoreNorm = scaler.transform(inputParameters)
    gtParameters = gtParameters.reshape(-1)

    gtParametersF = np.array(gtParameters)
    inputParametersF = np.array(inputParametersZscoreNorm)

    train_loader, test_loader = get_loader_wjcheon(input_original=inputParametersF,
                                                   output_original= gtParametersF,
                                                   cFold=2,
                                                   batch_size=config.batch_size,
                                                   num_workers= config.num_workers)

    solver = Solver(config, train_loader, test_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters

    # training hyper-parameters
    parser.add_argument('--input_ch', type=int, default=18)
    parser.add_argument('--output_ch', type=int, default=5)    # change 21.08.17 // classification: 5, regression: 1
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--validation_period', type=int, default=2) # not used
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)  # used
    parser.add_argument('--val_datapath', type=str, default='ResultValidation')

    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--currentCVNum', type=int, default=1)  # start from 1

    # misc
    # parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')  # path for saving model


    # DB
    parser.add_argument('--data_path', type=str,
                        default=r"C:\Users\admin\Dropbox\Research\개인연구\23_Brachy_BladderTocixity\GU toxicity_DB_python-balance.xlsx")
    # CUDA
    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)

