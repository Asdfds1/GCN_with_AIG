import torch
import pickle
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import dense_to_sparse
from AIG_Graph import Aig_graph
import re
import os
import json
import fnmatch
import numpy as np
from joblib import dump, load


class AIGDataset():
    def __init__(self, dataset_number=None, to_create_path=None, split_ratio=(0.7, 0.2), random_seed=42):
        super().__init__()
        self.scaler = MinMaxScaler()
        self.data_list = []
        self.test_list = []
        if dataset_number and os.path.isfile(f'../VKR_Project/datasets_aig/dataset_{dataset_number}.pickle'):
            self.load_dataset(dataset_number)
        elif to_create_path:
            self.create_dataset(to_create_path)
            self.save_dataset()
        else:
            raise ValueError("Either dataset_path or graphs must be provided.")

        # Разделяем на train и val используя split_ratio
        torch.manual_seed(random_seed)  # Для воспроизводимости
        data_size = len(self.data_list)
        train_size = int(split_ratio[0] * data_size)
        indices = torch.randperm(data_size).tolist()
        self.train_data_list = [self.data_list[i] for i in indices[:train_size]]
        self.val_data_list = [self.data_list[i] for i in indices[train_size:]]

    def create_dataset(self, path):
        list_id = self.sorted_alphanumeric(os.listdir(path))
        list_graph = []
        list_labels = []
        for id in list_id:
            list_name = self.sorted_alphanumeric(os.listdir(path + f'/{id}'))
            for name in list_name:
                if len(os.listdir(path + f'/{id}/{name}')) == 9:
                    aig_files = fnmatch.filter(os.listdir(path + f'/{id}/{name}/'), '*.aig')
                    with open(path + f'/{id}/{name}/{name}.json') as json_file:
                        data = json.load(json_file)
                        print(name)
                        list_labels.append(float(data["abcStats"]["area"]))
                        list_labels.append(float(data["abcStatsBalance"]["area"]))
                        list_labels.append(float(data["abcStatsResyn2"]["area"]))
                    for file_name in aig_files:
                        file_path = os.path.join(path + f'/{id}/{name}/', file_name)
                        with open(file_path, 'r') as file:
                            list_graph.append(file.read())
        label_array = np.array(list_labels).reshape(-1, 1)
        print(label_array)
        self.scaler.fit(label_array)
        scaled_label_array = self.scaler.transform(label_array)
        max_size = 0
        list_graphs = []
        for aig_str in list_graph:
            graph = Aig_graph()
            graph.parse_aig(aig_str)
            max_size = max(max_size, graph.matrix_size)
            list_graphs.append(graph)

        for index, graph in enumerate(list_graphs):
            graph.padding(max_size)
            node_features = [graph.node_vectors.get_vector(i) for i in range(len(graph.node_vectors.key_to_index))]

            edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
            edge_atr = torch.tensor(graph.edge_atr, dtype=torch.long)
            x = torch.tensor(node_features, dtype=torch.float)

            # Получение метки для текущего графа
            label = scaled_label_array[index][0]
            print(label)
            y = torch.tensor([label], dtype=torch.float)

            # Создание объекта Data
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_atr, y=y)
            self.data_list.append(data)
        print(self.data_list)

    def load_dataset(self, number):
        datasets_path = '../VKR_Project/datasets_aig/'
        dataset_path = datasets_path + f'dataset_{number}.pickle'
        scaler_path = datasets_path + f'scaler_{number}.joblib'
        self.scaler = load(scaler_path)
        with open(dataset_path, 'rb') as f:
            self.data_list = pickle.load(f)

    def save_dataset(self):
        # Определение директории для сохранения датасетов
        dataset_dir = '../VKR_Project/datasets_aig'
        # Создание директории, если она не существует
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # Поиск последнего индекса файла в указанной директории
        existing_files = [f for f in os.listdir(dataset_dir) if f.startswith('dataset_') and f.endswith('.pickle')]
        indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
        last_index = max(indices) if indices else 0

        # Имя следующего файла
        next_index = last_index + 1
        file_name = f'dataset_{next_index}.pickle'
        scaler_name = f'scaler_{next_index}.joblib'
        file_path = os.path.join(dataset_dir, file_name)
        scaler_path = os.path.join(dataset_dir, scaler_name)

        # Сохранение датасета
        with open(file_path, 'wb') as f:
            pickle.dump(self.data_list, f)
        dump(self.scaler, scaler_path)

        print(f'Датасет сохранен в файл: {file_path}')

    def get_data_loaders(self, batch_size=2, shuffle=False):
        train_loader = DataLoader(self.train_data_list, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(self.val_data_list, batch_size=batch_size,
                                shuffle=False)  # Обычно валидационный набор не перемешивают
        return train_loader, val_loader

    def get_test_loaders(self, path):
        list_id = self.sorted_alphanumeric(os.listdir(path))
        list_graph = []
        list_labels = []
        for id in list_id:
            list_name = self.sorted_alphanumeric(os.listdir(path + f'/{id}'))
            for name in list_name:
                if len(os.listdir(path + f'/{id}/{name}')) == 9:
                    aig_files = fnmatch.filter(os.listdir(path + f'/{id}/{name}/'), '*.aig')
                    with open(path + f'/{id}/{name}/{name}.json') as json_file:
                        data = json.load(json_file)
                        print(name)
                        list_labels.append(float(data["abcStats"]["area"]))
                        list_labels.append(float(data["abcStatsBalance"]["area"]))
                        list_labels.append(float(data["abcStatsResyn2"]["area"]))
                    for file_name in aig_files:
                        file_path = os.path.join(path + f'/{id}/{name}/', file_name)
                        with open(file_path, 'r') as file:
                            list_graph.append(file.read())
        max_size = self.data_list[0].x.shape[0]
        list_graphs = []
        for aig_str in list_graph:
            graph = Aig_graph()
            graph.parse_aig(aig_str)
            max_size = max(max_size, graph.matrix_size)
            list_graphs.append(graph)

        for index, graph in enumerate(list_graphs):
            graph.padding(max_size)
            node_features = [graph.node_vectors.get_vector(i) for i in range(len(graph.node_vectors.key_to_index))]

            edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
            edge_atr = torch.tensor(graph.edge_atr, dtype=torch.long)
            x = torch.tensor(node_features, dtype=torch.float)

            # Получение метки для текущего графа

            # Создание объекта Data
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_atr)
            self.test_list.append(data)
        return self.test_loaders(), list_labels

    def test_loaders(self):
        test_loader = DataLoader(self.test_list)
        return test_loader

    def get_num_node_features(self):
        # Предполагаем, что data_list не пуст и каждый Data объект имеет атрибут x
        if len(self.data_list) == 0:
            raise ValueError("data_list is empty. Ensure dataset is loaded or created correctly.")
        print(self.data_list)

        # Предполагаем, что атрибут x это тензор, где второе измерение это признаки
        print(self.data_list[0].x.shape[1])
        return self.data_list[0].x.shape[1]

    @staticmethod
    def sorted_alphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)


