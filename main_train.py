from cmath import log
import os
import torch
import torch.nn as nn
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from glob import glob
from pathlib import Path
from dataset import SoccerDataset
from trace_regressor import *
from trace_discriminator import TraceDiscriminator
from poss_classifier import PossClassifier, PossTransformerClassifier
from utils_train import train

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='settransformer', help='Option for tracking models(setlstm, settransformer, possclassifier)')
parser.add_argument('--target_type', type=str, default='gk', help='Option for tracking models(ball, gk)')
parser.add_argument('--phy_loss', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--cuda', type=str, default='1')

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
device = "cuda" if torch.cuda.is_available() else "cpu"

data_files = ['fm_traces']

if args.model_type == 'setlstm':
    trace_regressor = TraceSetLSTM(target_type=args.target_type).to(device)
elif args.model_type == 'settransformer':
    trace_regressor = TraceSetTransformer(target_type=args.target_type).to(device)
elif args.model_type == 'possclaasifier':
    trace_regressor = PossTransformerClassifier(ball_trace_given=False, mode='team', hidden_dim=args.hidden_dim).to(device)

def split_datasets(file_list, args):
    """ load datasets and split train, valid sets """
    root = Path(args.data_path)
    files = [glob(str(root/file/'*.csv')) for file in file_list]
    files = sorted(list(itertools.chain(*files)))

    return files[:-1], files[-1:]
    
def log_config(args):
    """ log configurations """
    print('load configurations...')
    for k, v in args._get_kwargs():
        print(f'{k} : {v}')

def load_datasets(t_file, v_file, args):
    """ load dataset, dataLoader """
    train_dataset = SoccerDataset(t_file, mode='true_gk' if args.target_type == 'ball' else 'no_gk', poss_encode=True)
    val_dataset = SoccerDataset(v_file, mode='true_gk' if args.target_type == 'ball' else 'no_gk', poss_encode=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)    

    return train_loader, val_loader

if __name__ == '__main__':
    # logging configurations
    log_config(args)

    print(data_files)

    train_files, val_files = split_datasets(data_files, args)
    
    train_files = glob('./data/fm_traces/*.csv') + glob('./data/metrica_traces/*.csv')
    train_files = [file for file in train_files if 'real_test.csv' not in file]

    val_files = glob('./data/metrica_traces/real_test.csv')

    print(train_files)
    print(val_files)

    train_loader, val_loader = load_datasets(train_files, val_files, args)
    
    # start training
    train(trace_regressor, train_loader, val_loader, args, phys_loss=args.phy_loss, device=device)

    print('Training finished...!')
    
    log_config(args)

    print(data_files)