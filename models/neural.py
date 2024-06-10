import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class NNAgent:
    def __init__(self, X_train, X_test, y_train, y_test, input_size, lr=0.001):
        self.train_set = self.get_data_loader(X_train, y_train)
        # self.X_train = X_train
        # self.X_test = X_test
        # self.y_train = y_train
        # self.y_test = y_test
        # self.input_size = input_size
        # self.model = NeuralNetwork(input_size)
        # self.criterion = nn.BCELoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, epochs=100):
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            y_pred = self.model(self.X_train)
            loss = self.criterion(y_pred, self.y_train)
            loss.backward()
            self.optimizer.step()

    def predict(self):
        self.model.eval()
        self.y_pred = self.model(self.X_test)

    def evaluate(self):
        return self.y_pred

    def get_model(self):
        return self.model

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def get_data_loader(self, X, y, batch_size=32):
        tensor_X = torch.tensor(X.toarray(), dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)