from torch_geometric_temporal import StaticGraphTemporalSignal

# from TemporalSignal import TemporalSignal
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
StaticGraphTemporalSignal

class MyDataLoader_large(object):
    def __init__(self):
        super(MyDataLoader_large, self).__init__()
        self._get_node_featuresssss()

    def _get_node_featuresssss(self):
        df = pd.read_csv('./data/nocircle8.csv', header=None)
        df.columns = ['timestamp', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',
                                   'y0', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7']
        grouped = df.groupby('timestamp')
        node_features_list = []

        for timestamp, group in tqdm(grouped):
            node_features = group.loc[group.timestamp == timestamp, ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']].values
            # print(node_features)
            node_features = torch.LongTensor(node_features).unsqueeze(1)
            # print(node_features)
            node_features_list.append(node_features)

        #拼接node_features得到X(6,1,39993)
        m = 1
        X = node_features_list[0]
        while m < 39990:  #2104
            X = torch.cat([X, node_features_list[m]], axis=0)
            m = m + 1
        X = X.permute(2, 1, 0)
        self.X = X

    def _get_edges_and_weights(self):
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
                                   [1, 2, 3, 4, 5, 6, 7]],
                                  dtype=torch.long)
        edge = edge_index.numpy()
        values = np.ones(5)
        self.edges = edge
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 8, num_timesteps_out: int = 1):
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            # for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_in + num_timesteps_out) + 1)
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            # features.append(np.concatenate(((self.X[:, :, i : i + num_timesteps_in]).numpy(), (self.X[:, :, i + num_timesteps_in + num_timesteps_out : i + num_timesteps_in + num_timesteps_out + num_timesteps_in]).numpy()), axis=-1))
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 8, num_timesteps_out: int = 1
    ) -> StaticGraphTemporalSignal:

        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)

        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        # print("features:", self.features[0].shape)
        # print("edges:", self.edges.shape)
        # print("edge_weights:", self.edge_weights.shape)
        # print("targets:", len(self.targets))

        return dataset