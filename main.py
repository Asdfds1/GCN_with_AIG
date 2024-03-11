import torch
import transformers
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertModel
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import RandomSampler, SequentialSampler
from scipy.special import softmax
import json
import fnmatch
import os
import Model
import re
from sklearn.preprocessing import MinMaxScaler

from GCN_model import GCN
from AIG_Graph import Aig_graph
from Create_dataset_for_AIG import AIGDataset

path_data_6 = '../VKR_Project/dataset/6'
path_tests_data = '../VKR_Project/tests_data'
path_data_8 = '../VKR_Project/dataset/8'
datasets_aig_path = '../VKR_Project/datasets_aig'

if __name__ == '__main__':
    # graph = Aig_graph()
    # with open('../VKR_Project/dataset/8/8/CCGRCG10/CCGRCG10_BALANCED.aig', 'r') as file:
    #     graph.parse_aig(file.read())
    # graph.padding(10)

    path = datasets_aig_path + '/dataset_12.pickle'
    dataset = AIGDataset(to_create_path=path_data_8)
    #
    # dataset = AIGDataset(created_dataset_path=path)
    train_loader, val_loader = dataset.get_data_loaders()
    num_node_features = dataset.get_num_node_features()

    model = GCN(num_node_features=num_node_features, num_classes=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.fit(train_loader, val_loader, optimizer, criterion, 2)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
