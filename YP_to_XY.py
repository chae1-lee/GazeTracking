# yaw, pitch -> X, Y

import torch
from torch import nn

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


def loadData(path, is_train=True, train_ratio=0.6, test_ratio=0.2):
    contents = pd.read_table(path, sep=' ').sample(frac=1, random_state=42, ignore_index=True)  # shuffle

    cut_of_train = int(len(contents) * train_ratio)
    cut_of_test = int(len(contents) * (train_ratio + test_ratio))

    if is_train:
        train = np.array(contents.iloc[:cut_of_train])
        val = np.array(contents.iloc[cut_of_train:cut_of_test])

        train_X = train[:, 1:-2]
        train_Y = train[:, -2:]
        val_X = val[:, 1:-2]
        val_Y = val[:, -2:]

        return train_X.astype("float32"), train_Y.astype("float32"), val_X.astype("float32"), val_Y.astype("float32")

    else:
        test = np.array(contents.iloc[cut_of_test:])

        test_X = test[:, 1:-2]
        test_Y = test[:, -2:]

        return test_X.astype("float32"), test_Y.astype("float32")


if __name__ == "__main__":
    is_train = True
    if is_train:
        train_X, train_Y, val_X, val_Y = loadData(path="../feature.txt")
    else:
        test_X, test_Y = loadData(path="../feature.txt", is_train=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LinearRegression()
    model.fit(train_X, train_Y)

    print("가중치(계수, 기울기 파라미터 W) :", model.coef_)
    print("편향(절편 파라미터 b) :", model.intercept_)

    print("훈련세트 점수: {:.2f}".format(model.score(train_X, train_Y)))
    print("검증세트 점수: {:.2f}".format(model.score(val_X, val_Y)))

    '''
    net = MLP().to(device)
    print(net)

    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    batch_size = 100
    for epoch in range(1000):
        avg_cost = 0.0

        for batch in range(batch_size, len(train_X), batch_size):
            X = torch.Tensor(train_X[batch-batch_size:batch])
            Y = torch.Tensor(train_Y[batch-batch_size:batch])

            optimizer.zero_grad()
            y_pred = net(X)
            cost = criterion(y_pred, Y)
            cost.backward()

            avg_cost += cost / batch_size

        if (epoch+1) % 10 == 0:
            print("Epoch: {:02d}".format(epoch+1), '\t', "Avg. cost: {:.4f}".format(avg_cost))
    '''


# reference: https://tutorials.pytorch.kr/beginner/basics/buildmodel_tutorial.html
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(27, 13),
            nn.ReLU(),
            nn.Linear(13, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU(),
            nn.Linear(3, 2)
        )

    def forward(self, x):
        return self.mlp(x)