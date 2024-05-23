import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any
from sklearn.metrics import accuracy_score


class WebsitePhishingBaseModel:
    def __init__(
        self, model: Any = None, model_name: str = None, train_ratio: float = 0.8
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.train_ratio = train_ratio
        self.pd_data = None
        self.data = None
    
    def from_arff(self, file_path: str) -> None:
        # Check the file format
        if not file_path.endswith(".arff"):
            raise ValueError("Invalid format of the file, expected ``.arff``")
        
        # Read the entire file
        with open(file_path, 'r') as f:    
            arff_data = f.read()
            
        # Remove comment lines
        arff_data = '\n'.join([line for line in arff_data.split('\n') if not line.strip().startswith('%')])
        
        # Find the attribute definition section
        attr_start = arff_data.index('@attribute')
        attr_end = arff_data.index('@data')
        
        # Extract the attribute definitions
        attributes = []
        for line in arff_data[attr_start:attr_end].split('\n'):
            if line.strip().startswith('@attribute'):
                attr_parts = line.strip().split(' ', 2)
                attr_name = attr_parts[1]
                attr_type = attr_parts[2].strip("'")
                attributes.append((attr_name, attr_type))
        
        # Extract the data section
        data_start = arff_data.index('@data') + 6
        data = [line.strip().split(',') for line in arff_data[data_start:].split('\n') if line.strip()]
        
        # Create a pandas DataFrame
        self.pd_data = pd.DataFrame(data, columns=[attr[0] for attr in attributes], dtype=float)
        self.data = np.array(self.pd_data, dtype=np.int64)
        
    def show_data_distribution(self, save_path: str = "plot/data_distribution.png") -> None:
        self.pd_data.hist(bins = 50, figsize = (15, 15))
        plt.savefig(save_path)
        
    def show_data_heatmap(self, save_path: str = "plot/data_heatmap.png") -> None:
        plt.figure(figsize=(15,13))
        sns.heatmap(self.pd_data.corr())
        plt.savefig(save_path)
    
    def data_process(self):
        data_size = self.data.shape[0]
        train_size = int(data_size * self.train_ratio)
        shuffle_indices = np.random.permutation(data_size)
        data = self.data[shuffle_indices]
        self.train_data = self.data[:train_size]
        self.test_data = self.data[train_size:]
        self.train_x = self.train_data[:, :-1]
        self.train_y = self.train_data[:, -1]
        self.test_x = self.test_data[:, :-1]
        self.test_y = self.test_data[:, -1]
        
    def train(self):
        self.data_process()
        self.model.fit(self.train_x, self.train_y)
    
    def evaluate(self):
        train_pred = self.model.predict(self.train_x)
        test_pred = self.model.predict(self.test_x)
        train_acc = accuracy_score(self.train_y, train_pred)
        test_acc = accuracy_score(self.test_y, test_pred)
        return train_acc, test_acc
        