# yaw, pitch -> X, Y

import torch
from torch import nn

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as MAE

import matplotlib.pyplot as plt


def loadData(path, is_train=True, train_ratio=0.6, test_ratio=0.2):
    contents = pd.read_table(path, sep=' ').sample(frac=1, random_state=42, ignore_index=True)  # shuffle
    """
        string:
            contents.iloc[:, 0]     # file path
        int:
            contents.iloc[:, 1:13]  # original feature D05 ~ D15
            contents.iloc[:, -5]    # original feature D28 (which eye)
            contents.iloc[:, -2:]   # original feature D02 ~ D03 (target x, y)
        float:
            contents.iloc[:, 13:-5] # original feature D16 ~ D27
            contents.iloc[:, -4:-2] # yaw, pitch
    """
    contents.iloc[:, 1:13] = contents.iloc[:, 1:13].values.astype("int32")
    contents.iloc[:, -5] = contents.iloc[:, -5].values.astype("int32")
    contents.iloc[:, -2:] = contents.iloc[:, -2:].values.astype("int32")
    contents.iloc[:, 13:-5] = contents.iloc[:, 13:-5].values.astype("float32")
    contents.iloc[:, -4:-2] = contents.iloc[:, -4:-2].values.astype("float32")

    cut_of_train = int(len(contents) * train_ratio)
    cut_of_test = int(len(contents) * (train_ratio + test_ratio))

    if is_train:
        train = np.array(contents.iloc[:cut_of_train])
        val = np.array(contents.iloc[cut_of_train:cut_of_test])

        # 전체
        # train_X = train[:, 1:-2]
        # val_X = val[:, 1:-2]

        # 3D target location(x, y, z), yaw, pitch
        # train_X = np.concatenate((train[:, -8:-5], train[:, -4:-2]), axis=1)
        # val_X = np.concatenate((val[:, -8:-5], val[:, -4:-2]), axis=1)

        # 3D target location(x, y), pitch
        train_X = np.concatenate((train[:, -8:-6], train[:, -3].reshape(len(train[:, -3]), 1)), axis=1)
        val_X = np.concatenate((val[:, -8:-6], val[:, -3].reshape(len(val[:, -3]), 1)), axis=1)

        train_Y = train[:, -2:]
        val_Y = val[:, -2:]

        return train_X, train_Y, val_X, val_Y

    else:
        test = np.array(contents.iloc[cut_of_test:])

        # 전체
        # test_X = test[:, 1:-2]

        # 3D target location(x, y, z), yaw, pitch
        # test_X = np.concatenate((test[:, -8:-5], test[:, -4:-2]), axis=1)

        # 3D target location(x, y), pitch
        test_X = np.concatenate((test[:, -8:-6], test[:, -3].reshape(len(test[:, -3]), 1)), axis=1)

        test_Y = test[:, -2:]
        return test_X, test_Y


def drawShadedPlot(x, y, step_size=100):
    x_axis = []
    y_axis = []

    for step in range(step_size, len(x), step_size):
        x_axis.append(x[step - step_size:step].mean())
        y_axis.append((y[step - step_size:step].mean(), y[step - step_size:step].std()))

    return np.array(x_axis), np.array(y_axis)


def showPlot(plotData):

    coor_X = np.array(plotData.sort_values(by=["True_X"]).iloc[:, :2])
    coor_Y = np.array(plotData.sort_values(by=["True_Y"]).iloc[:, 2:])

    step_size = 100

    coor_X_x, coor_X_y = drawShadedPlot(coor_X[:, 0], coor_X[:, 1], step_size=step_size)
    coor_Y_x, coor_Y_y = drawShadedPlot(coor_Y[:, 0], coor_Y[:, 1], step_size=step_size)

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(coor_X_x, coor_X_y[:, 0], label='X')
    plt.fill_between(coor_X_x, coor_X_y[:, 0] - coor_X_y[:, 1], coor_X_y[:, 0] + coor_X_y[:, 1], alpha=0.2)
    plt.legend()
    plt.title("Step size: " + str(step_size))

    plt.subplot(2, 1, 2)
    plt.plot(coor_Y_x, coor_Y_y[:, 0], label='Y')
    plt.fill_between(coor_Y_x, coor_Y_y[:, 0] - coor_Y_y[:, 1], coor_Y_y[:, 0] + coor_Y_y[:, 1], alpha=0.2)
    plt.legend()

    plt.tight_layout
    plt.show()


if __name__ == "__main__":
    is_train = True
    if is_train:
        train_X, train_Y, val_X, val_Y = loadData(path="../feature.txt")
    else:
        test_X, test_Y = loadData(path="../feature.txt", is_train=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # reference: https://inuplace.tistory.com/500
    model = LinearRegression()
    model.fit(train_X, train_Y)

    # print("가중치(계수, 기울기 파라미터 W) :", model.coef_)
    # print("편향(절편 파라미터 b) :", model.intercept_)

    test_X, test_Y = loadData(path="../feature.txt", is_train=False)

    print("Train R2: {:.4f}".format(model.score(train_X, train_Y)))
    print("Train MAE: {:.4f}".format(MAE(train_Y, model.predict(train_X))))
    print()
    print("Val. R2: {:.4f}".format(model.score(val_X, val_Y)))
    print("Val. MAE: {:.4f}".format(MAE(val_Y, model.predict(val_X))))
    print()
    print("Test R2: {:.4f}".format(model.score(test_X, test_Y)))
    print("Test MAE: {:.4f}".format(MAE(test_Y, model.predict(test_X))))

    y_pred = model.predict(test_X)
    y_true = test_Y
    plotData = pd.DataFrame(np.array((y_true[:, 0], y_pred[:, 0], y_true[:, 1], y_pred[:, 1])).T, columns=["True_X", "Pred_X", "True_Y", "Pred_Y"])
    showPlot(plotData)

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