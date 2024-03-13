import networkx as nx
import re
import numpy as  np
import os
from node2vec import Node2Vec
from gensim.models import KeyedVectors


class Aig_graph:

    node_vectors: KeyedVectors

    def __init__(self):
        self.nodes = {}  # словарь для хранения узлов и их индексов
        self.edges = []  # список для хранения рёбер
        self.edge_index = [[], []]
        self.edge_atr = []
        self.node_types = {}  # словарь для хранения типов узлов
        self.type_map = {'input': 1, 'output': 2, 'and': 3, 'const': 4}
        self.net_graph = nx.Graph()
        self.adjacency_matrix = None
        self.matrix_size = None

    def parse_aig(self, aig_str):
        # Определение регулярных выражений для парсинга AIG
        name = re.compile(r'\bmodule\s+(\w+)')
        input_regex = re.compile(r'input\s+(.*?);')
        wire_regex = re.compile(r'wire\s+(.*?);', re.DOTALL)
        const_regex = re.compile(r"1'b[01]")
        output_regax = re.compile(r'output\s+(.*?);', re.DOTALL)

        assign_regex = re.compile(r'assign\s+(.*?)\s+=\s+(.*?);')


        # Разбор разных частей AIG
        inputs = input_regex.findall(aig_str)
        wires = wire_regex.findall(aig_str)
        consts = const_regex.findall(aig_str)
        outputs = output_regax.findall(aig_str)

        assignments = assign_regex.findall(aig_str)

        wires = [wire.replace('\n', '') for wire in wires]
        outputs = [output.replace('\n', '') for output in outputs]

        # Добавление узлов
        all_nodes = inputs[0]
        if wires and wires[0]:  # Проверяем что wires не пуст и wires[0] не пустая строка
            all_nodes += ',' + wires[0]
        if consts and consts[0]:
            help_list = list(set(consts))
            for const in help_list:
                all_nodes += ',' + const
        all_nodes += ',' + outputs[0]
        all_nodes = "".join(all_nodes.split())
        for index, variable in enumerate(all_nodes.split(',')):
            self.nodes[variable.strip()] = index
        all_nodes_list = all_nodes.split(',')

        # Заполнение словаря узлов типами
        for node in all_nodes_list:
            node = node.strip()

            if node in inputs[0]:
                self.node_types[self.nodes[node]] = self.type_map['input']
            elif node in outputs[0]:
                self.node_types[self.nodes[node]] = self.type_map['output']
            elif wires and wires[0]:
                if node in wires[0]:
                    self.node_types[self.nodes[node]] = self.type_map['and']
            elif node == "1'b0" or node == "1'b1":
                self.node_types[self.nodes[node]] = self.type_map['const']

        # Создание списка ребер

        for output_var, expression in assignments:
            # Получение переменных в текущем выражении
            variable_names = re.findall(r"(~?)(x[0-9]+|new_n[0-9_]+|1'b[0-1])", expression)
            for invert_flag, var_name in variable_names:
                inverted = 1 if invert_flag == '~' else 0
                var_index = self.nodes[var_name.strip()]
                out_index = self.nodes[output_var.strip()]
                self.edge_index[0].append(var_index)
                self.edge_index[1].append(out_index)
                self.edge_atr.append(inverted)
                self.edges.append((var_index, out_index, inverted))
        self.create_adjacency_table()
        self.create_networkx_graph()
        self.create_node_vectors()

    def create_adjacency_table(self):
        num_nodes = len(self.nodes)
        # Инициализация матрицы смежности с нулями
        self.adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]
        self.matrix_size = num_nodes

        # Заполнение матрицы смежности
        for src, dst, inverted in self.edges:
            # print(f'{self.matrix_size}/ {src}/ {dst}')
            self.adjacency_matrix[src][dst] = 1 if not inverted else -1

    def create_networkx_graph(self):
        if self.adjacency_matrix is None:
            raise ValueError("Матрицы смежности отсутствует")

        for node, node_type in self.node_types.items():
            self.net_graph.add_node(node, type=node_type)

        for i in range(self.matrix_size):
            for j in range(i + 1, self.matrix_size):
                if self.adjacency_matrix[i][j] == 1:
                    self.net_graph.add_edge(i, j)
                elif self.adjacency_matrix[i][j] == -1:
                    self.net_graph.add_edge(i, j, type='not')

    def create_node_vectors(self):
        node2vec = Node2Vec(self.net_graph, dimensions=32, walk_length=10, num_walks=10, workers=2)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        self.node_vectors = model.wv

    def padding(self, max_size):
        if self.matrix_size < max_size:
            # Расширяем существующие строки до max_size
            for i in range(self.matrix_size):
                self.adjacency_matrix[i].extend([0] * (max_size - self.matrix_size))

                # Добавляем нулевые строки до достижения max_size
            for i in range(self.matrix_size, max_size):
                self.adjacency_matrix.append([0] * max_size)
        # Получаем существующие ключи и векторы
        existing_keys = self.node_vectors.index_to_key
        existing_vectors = self.node_vectors.vectors

        # Определяем сколько векторов нам нужно добавить
        num_vectors_to_add = max_size - len(existing_keys)
        if num_vectors_to_add > 0:
            # Создаем нулевые векторы
            zero_vectors = np.zeros((num_vectors_to_add, 32))

            # Добавляем нулевые векторы к существующим
            new_vectors = np.vstack((existing_vectors, zero_vectors))

            # Создаем новые ключи для нулевых векторов
            new_keys = [i for i in range(len(existing_keys), len(existing_keys) + num_vectors_to_add)]

            # Объединяем ключи
            all_keys = existing_keys + new_keys

            # Определяем новые векторы в модели
            self.node_vectors.vectors = new_vectors
            self.node_vectors.key_to_index = {key: i for i, key in enumerate(all_keys)}
            self.node_vectors.index_to_key = all_keys






