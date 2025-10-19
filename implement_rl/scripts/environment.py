import numpy as np
import pandas as pd

class NetworkEnvironment:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.states = self.data.drop('label', axis=1).values
        self.labels = self.data['label'].values
        self.n_states = len(self.states)
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return 0

    def step(self, action):
        true_label = self.labels[self.current_index]
        reward = 1 if action == true_label else -1
        self.current_index = (self.current_index + 1) % self.n_states
        done = self.current_index == 0
        return self.current_index, reward, done
