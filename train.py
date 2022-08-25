import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy import signal
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
import _pickle as cPickle
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
# from TenSim.utils.data_reader import TomatoDataset
# from MyiGrow.simulator import Net
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import datetime
import torch.nn.functional as F
import torch.nn as nn
import warnings
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

# import gym
from sklearn.metrics import r2_score


class TomatoDataset(object):

    def __init__(self, train_file, tmp_folder):
        self.train_file = train_file
        self.tmp_folder = tmp_folder

    def read_data(self, train_file):
        with open(train_file, 'r') as f:
            train_file_list = f.readlines()

        data = []
        for file_path in train_file_list:
            X = loadmat(file_path.replace('\n', 'X.mat'))['X']
            Y = loadmat(file_path.replace('\n', 'monitor.mat'))['monitor']

            # idx = (~np.isnan(Y.sum(0))).nonzero()[0]
            # Y = Y[:, idx]
            data.append((X, Y))
        return data

    def illu_irri_process(self, X):

        for i in range(164):  # days of each eposide
            # illumination progress
            illu_time = X[i * 24][16]
            illu_end = X[i * 24][17]
            illu_start = illu_end - illu_time
            # irrigation progress
            irri_start = X[i * 24][35]
            irri_end = X[i * 24][36]
            for j in range(24):

                if illu_start - 1 < j < illu_end:
                    if j + 1 - illu_start < 1:
                        X[i * 24 + j][16] = j + 1 - illu_start
                    elif j + 1 > illu_end:
                        X[i * 24 + j][16] = illu_end - j
                    else:
                        X[i * 24 + j][16] = 1
                else:
                    X[i * 24 + j][16] = 0

                if irri_start - 1 < j < irri_end:
                    if j + 1 - irri_start < 1:
                        X[i * 24 + j][35] = j + 1 - irri_start
                    elif j + 1 > irri_end:
                        X[i * 24 + j][35] = irri_end - j
                    else:
                        X[i * 24 + j][35] = 1
                else:
                    X[i * 24 + j][35] = 0

    def data_process(self, data):

        simulator_model_path = os.path.join(self.tmp_folder, 'model')
        if not os.path.exists(simulator_model_path):
            os.makedirs(simulator_model_path)

        if len(data) == 0:
            return []

        HOURS_IN_DAY = 24
        bad_days = [15, 75]

        train_X_list = []
        train_Y_list = []

        for d in data:
            X, Y = d

            bad_index = []
            n = X.shape[0]
            for bd in bad_days:
                bad_index += list(range(bd * HOURS_IN_DAY,
                                        bd * HOURS_IN_DAY + HOURS_IN_DAY))
            good_index = sorted(list(set(range(n)).difference(bad_index)))
            X = X[good_index, :]
            Y = np.vstack((Y[0], Y))
            Y = Y[good_index, :]

            # x = Y[:,3:9]
            # np.save('weather.npy', x)

            self.illu_irri_process(X)

            Y[:, 37] = np.cumsum(Y[:, 37])
            # 0.0014 * 1e5 /7 *8

            smooth_lai = signal.savgol_filter(
                Y[:, 29], window_length=999, polyorder=2)
            smooth_lai[smooth_lai < 0] = 0
            Y[:, 29] = smooth_lai

            smooth_plantload = signal.savgol_filter(
                Y[:, 35], window_length=999, polyorder=2)
            smooth_plantload[smooth_plantload < 0] = 0
            Y[:, 35] = smooth_plantload

            outside_weather = Y[:, [3, 4, 5, 6, 7, 8]
                              ]  # Igolb, Tout, RHout, Co2out, Windsp, Tsky

            if not os.path.exists(os.path.join(simulator_model_path, 'weather.npy')):
                np.save(os.path.join(simulator_model_path,
                                     'weather.npy'), outside_weather)
            # comp1.temp, comp1.co2, comp1.illumination, comp1.irrigation
            control = X[:, [19, 23, 16, 35]]
            inside_weather = Y[:, [0, 1, 2]]  # AirT, AirRH, Airppm
            crop_state = Y[:, [29, 35, 37]]  # LAI, PlantLoad, NetGrowth
            fw = Y[:, 31].reshape(len(Y), -1)  # CumFruitsCount

            '''PARsensor linear regression'''
            par_x = np.hstack((outside_weather[:, [0]], control[:, [2]]))
            if not os.path.exists(os.path.join(simulator_model_path, 'PARsensor_regression_paramsters.pkl')):
                par_y = Y[:, 9]
                linreg = LinearRegression()
                linreg.fit(par_x, par_y)
                pickle.dump(linreg,
                            open(os.path.join(simulator_model_path, 'PARsensor_regression_paramsters.pkl'), 'wb'))
            else:
                linreg = pickle.load(
                    open(os.path.join(simulator_model_path, 'PARsensor_regression_paramsters.pkl'), 'rb'))
            PARsensor = linreg.predict(par_x)
            PARsensor = np.where(PARsensor > 50.0, PARsensor, 0)
            PARsensor = PARsensor.reshape(len(Y), -1)

            train_X = np.hstack(
                (outside_weather, control, inside_weather, PARsensor, crop_state, fw))[:-1]
            train_Y = np.hstack((inside_weather, crop_state, fw))[1:]

            train_X_list.append(train_X)
            train_Y_list.append(train_Y)

        train_X_all = np.array(train_X_list)
        train_Y_all = np.array(train_Y_list)

        return train_X_all, train_Y_all


    def PAR_x_y(self, X, Y):

        train_X = np.concatenate(X[:, :, [0, 8]], axis=0)
        train_Y = np.concatenate(X[:, :, 13])
        return train_X, train_Y


    def greenhouse_x_y(self, X, Y):

        train_X = np.concatenate(X[:, :, :13], axis=0)
        train_Y = np.concatenate(Y[:, :, :3], axis=0)

        return train_X, train_Y


    def crop_front_x_y(self, X, Y):

        train_X = np.concatenate(X[:, :, 10:17], axis=0)
        train_Y = np.concatenate(Y[:, :, 3:6], axis=0)

        return train_X, train_Y


    def crop_back_x_y(self, X, Y):

        DAY_IN_LIFE_CYCLE = 160
        day_index = [23 + i * 24 for i in range(DAY_IN_LIFE_CYCLE)]
        day_index_plus = [23 + (i + 1) * 24 for i in range(DAY_IN_LIFE_CYCLE)]
        train_X = np.concatenate(X[:, day_index, -4:], axis=0)
        train_Y = np.concatenate(X[:, day_index_plus, -1], axis=0)
        train_Y = train_Y.reshape(len(train_Y), -1)

        return train_X, train_Y


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=120):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        y = self.fc3(x) + x
        y = F.relu(y)
        y = self.fc4(y)
        y = F.relu(y)
        y = self.fc4(y)
        y = F.relu(y)
        z = self.fc4(y) + y
        z = F.relu(z)
        z = self.fc4(z)
        z = F.relu(z)
        z = self.fc4(z)
        z = F.relu(z)
        m = self.fc4(z) + z
        m = F.relu(m)
        m = self.fc4(m)
        m = F.relu(m)
        m = self.fc4(m)
        m = F.relu(m)
        m = self.fc4(m)
        m = F.relu(m)
        n = self.fc4(m) + m
        n = F.relu(n)
        n = self.fc5(n)
        return n


warnings.filterwarnings("ignore")
os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'
# gym.logger.set_level(40)
torch.set_num_threads(1)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class DealDataset(Dataset):
    """


    """

    def __init__(self, train_x, train_y):
        self.x_data, self.y_data = self.get_data(train_x, train_y)
        self.len = len(self.x_data)

    def get_data(self, train_x, train_y):
        '''

        x_scaler = pickle.load(open(x_scaler_path, 'rb'))
        y_scaler = pickle.load(open(y_scaler_path, 'rb'))

        data_x_normal = x_scaler.transform(train_x)
        data_y_normal = y_scaler.transform(train_y)
        '''

        data_x_normal = self.normalization(train_x)
        data_y_normal = self.normalization(train_y)

        return data_x_normal, data_y_normal

    def normalization(self, data):
        scaler = MinMaxScaler()
        data_normal = []
        a = data.shape[0]
        b = data.shape[1]
        c = data.reshape((a * b, data.shape[2]))
        d = scaler.fit_transform(c)
        data_normal.append(d)
        data_normal = np.array(data_normal).reshape((a, b, data.shape[2]))
        return data_normal

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


data_path = r'/openbayes/input/input0/test.txt'
tmp_folder = r'/output'
wur_tomato_reader = TomatoDataset(data_path, tmp_folder)
data = wur_tomato_reader.read_data(data_path)
full_train_x, full_train_y = wur_tomato_reader.data_process(data)

day = []
for i in range(164):
    for k in range(24):
        day.append(i)
day.remove(163)
day = np.array(day * 50).reshape(50, 3935, 1)
train_x = np.concatenate((day, full_train_x[:, :, [0, 1, 2, 3, 5, 10, 11, 12, 13]]), axis=2)
train_y = np.concatenate((full_train_x[:, :, 17].reshape(50, 3935, 1),
                          full_train_y[:, :, 3:6]), axis=2)
# train_y = full_train_x[:, :, 17].reshape(50, 3935, 1)
dataset = DealDataset(train_x, train_y)
length = [len(dataset) * 0.8]
length.append(len(dataset) - length[0])
train_db, test_db = torch.utils.data.random_split(dataset, [int(length[0]), int(length[1])])
train_data = DataLoader(dataset=train_db, batch_size=4, shuffle=True)
test_data = DataLoader(dataset=test_db, batch_size=2, shuffle=False)

net = Net(input_dim=10, output_dim=4, hidden_dim=1000)

lr = 0.001
Epoch = 500
save_path = './model/'
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss_func = torch.nn.MSELoss(reduce=True, reduction='sum')  # sum loss
# mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                            milestones=list(
#                                                                range(50, Epoch, 50)),
#                                                            gamma=0.8)
mult_step_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.8)

train_loss = []
valid_loss = []
min_valid_loss = 10000
# min_train_loss = 100
net.to(device)
for i, epoch in enumerate(range(Epoch)):
    total_train_loss = []
    net.train()
    for j, data in enumerate(train_data):
        x, y = data
        # x, y = Variable(x).float(), Variable(y).float()
        x, y = torch.FloatTensor(x.float()).to(
            device), torch.FloatTensor(y.float()).to(device)
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        total_train_loss.append(loss)
        # total_train_loss[j] = loss
    train_loss.append(np.mean(total_train_loss))
    # train_loss.append(torch.mean(total_train_loss).detach().cpu())

    ################################# eval ###############################
    total_valid_loss = []

    # total_valid_loss = torch.zeros(len(val_loader))
    net.eval()
    for step, (b_x, b_y) in enumerate(test_data):
        b_x = torch.FloatTensor(b_x.float()).to(device)
        b_y = torch.FloatTensor(b_y.float()).to(device)
        pred = net(b_x)
        loss = loss_func(pred, b_y)
        loss = loss.item()
        total_valid_loss.append(loss)
        # total_valid_loss[step] = loss.item()
    valid_loss.append(np.mean(total_valid_loss))
    # valid_loss.append(torch.mean(total_valid_loss).detach().cpu())

    lr = optimizer.param_groups[0]['lr']

    if (valid_loss[-1] <= min_valid_loss):
        print('epoch:{}, save!'.format(epoch))
        min_valid_loss = valid_loss[-1]
        if min_valid_loss < 0.0006:
            torch.save(net.state_dict(), save_path + str(min_valid_loss) + '.pkl',
                       _use_new_zipfile_serialization=False)

    log_string = ('iter: [{:d}/{:d}], train_loss: {:0.10f}, valid_loss: {:0.10f}, '
                  'best_valid_loss: {:0.10f}, lr: {:0.10f}').format((i + 1), Epoch,
                                                                    train_loss[-1],
                                                                    valid_loss[-1],
                                                                    min_valid_loss,
                                                                    lr)
    mult_step_scheduler.step()

    print(str(datetime.datetime.now()) + ': ')
    print(log_string)