import torch
import pandas as pd

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

    path = datasets_aig_path + '/dataset_38.pickle'
    dataset = AIGDataset(to_create_path=path_data_6)

    # dataset = AIGDataset(dataset_number=41)
    train_loader, val_loader = dataset.get_data_loaders(batch_size=10)
    test_loader, labels = dataset.get_test_loaders(path_data_8)
    num_node_features = dataset.get_num_node_features()

    model = GCN(num_node_features=num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    model.fit(train_loader, val_loader, optimizer, criterion, 20)
    preds = model.predict(test_loader)
    scaler = dataset.scaler
    predictions = scaler.inverse_transform(preds)

    true_label_df = pd.DataFrame(labels, columns=['true'])
    predictions_df = pd.DataFrame(predictions, columns=['pred'])
    comparison_df = predictions_df.join(true_label_df)
    comparison_df.to_csv('pred_2.csv', index=False)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
