import pickle
import pandas as pd
import numpy as np
import argparse
import os
import warnings
import torch
import torch.nn as nn
# import gym
from sklearn.metrics import r2_score
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import style
from train import TomatoDataset, Net


# from MyiGrow.simulator import Net2
# from aTest import Net

# from TenSim.utils.data_reader import TomatoDataset
# from MyiGrow.simulator import PredictModel
# from utils.common import mkdir

warnings.filterwarnings("ignore")
os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8'
# gym.logger.set_level(40)
torch.set_num_threads(1)


# def env(base_tmp_folder):
#     model_path = base_tmp_folder
#     scaler_dir = base_tmp_folder
#
#     ten_env = PredictModel(model1_dir=model_path + '137.19588928222657.pkl',
#                            scaler_x=scaler_dir + 'scaler_x.pkl',
#                            scaler_y=scaler_dir + 'scaler_y.pkl'
#                            )
#     return ten_env


def r2score(real, pre):
    mse = np.sum((np.array(real) - np.array(pre)) ** 2) / len(real)
    var = np.var(real)
    score = 1 - mse / var
    return score


def Table1():
    print("=============Table1===============")
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

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    train_x = scaler_x.fit_transform(train_x.reshape((50 * 3935, 10)))
    train_y = scaler_y.fit_transform(train_y.reshape((50 * 3935, 4)))
    train_x = train_x.reshape((50, 3935, 10))
    train_y = train_y.reshape((50, 3935, 4))

    columns = ['FW', 'LAI', 'PlantLoad', 'NetGrouwth']
    # save_dir = './table1'
    # mkdir(save_dir)

    # save_path = save_dir+'R2_of_per_cache.csv'
    # if os.path.exists(save_path):
    #     os.remove(save_path)

    # PAR_R2 = []
    # AirT_R2 = []
    # AirRH_R2 = []
    # Airppm_R2 = []
    LAI_R2 = []
    PlantLoad_R2 = []
    NetGrowth_R2 = []
    FW_R2 = []
    score = []
    pre = []
    net = Net(input_dim=10, output_dim=4, hidden_dim=1000)
    net.load_state_dict(torch.load(r'/openbayes/home/model/0.0006051297066733241.pkl'))
    net.eval()
    loss_func = torch.nn.MSELoss(reduce=True, reduction='sum')
    total_loss = []
    with torch.no_grad():
        for idx in range(1):
            # PAR_list, real_PAR_list = [], []
            # AirT_list, real_AirT_list = [], []
            # AirRH_list, real_AirRH_list = [], []
            # Airppm_list, real_Airppm_list = [], []
            LAI, LAI_list, real_LAI_list = [], [], []
            PlantLoad, PlantLoad_list, real_PlantLoad_list = [], [], []
            NetGrowth, NetGrowth_list, real_NetGrowth_list = [], [], []
            FW, FW_list, real_FW_list = [], [], []
            input = train_x[2]
            y = train_y[2]
            done = False
            # print(train_x.shape)
            # simulator.reset()
            day_fw = 0

            action = torch.tensor(input, dtype=torch.float)
            obs = net(action)
            target = torch.tensor(y, dtype=torch.float)
            loss = loss_func(obs, target)
            loss = loss.item()
            total_loss.append(loss)

            obs = obs.detach().numpy()
            obs = scaler_y.inverse_transform(obs.reshape(3935, 4))
            y = scaler_y.inverse_transform(y)
            LAI_list.append(obs[:3840, 1])
            PlantLoad_list.append(obs[:3840, 2])
            NetGrowth_list.append(obs[:3840, 3])
            FW_list.append(obs[:3840, 0])

            real_LAI_list.append(y[:3840, 1])
            real_PlantLoad_list.append(y[:3840, 2])
            real_NetGrowth_list.append(y[:3840, 3])
            real_FW_list.append(y[:3840, 0])

            # calculate R^2
            # r2_PAR = r2_score(real_PAR_list, PAR_list)
            # r2_AirT = r2_score(real_AirT_list, AirT_list)
            # r2_AirRH = r2_score(real_AirRH_list, AirRH_list)
            # r2_Airppm = r2_score(real_Airppm_list, Airppm_list)

            r2_LAI = r2score(np.array(real_LAI_list).reshape(1, -1).tolist(),
                             np.array(LAI_list).reshape(1, -1).tolist())
            r2_PlantLoad = r2score(np.array(real_PlantLoad_list).reshape(1, -1).tolist(),
                                   np.array(PlantLoad_list).reshape(1, -1).tolist())
            r2_NetGrowth = r2score(np.array(real_NetGrowth_list).reshape(1, -1).tolist(),
                                   np.array(NetGrowth_list).reshape(1, -1).tolist())
            r2_FW = r2score(np.array(real_FW_list).reshape(1, -1).tolist(),
                            np.array(FW_list).reshape(1, -1).tolist())

            pre = np.concatenate((np.array(FW_list).reshape(-1, 1), np.array(LAI_list).reshape(-1, 1),
                                  np.array(PlantLoad_list).reshape(-1, 1),
                                  np.array(NetGrowth_list).reshape(-1, 1),), axis=1)

            x = [i for i in range(3840)]

            plt.plot(x, np.array(FW_list).reshape(-1, 1), color='r', label='pre')
            plt.plot(x, np.array(real_FW_list).reshape(-1, 1), color='b', label='true')
            plt.xlabel('hours')
            plt.title('FW')
            plt.show()

            plt.plot(x, np.array(LAI_list).reshape(-1, 1), color='r', label='pre')
            plt.plot(x, y[:3840, 1], color='b', label='true')
            plt.xlabel('hours')
            plt.title('LAI')
            plt.show()

            plt.plot(x, np.array(PlantLoad_list).reshape(-1, 1), color='r', label='pre')
            plt.plot(x, y[:3840, 2], color='b', label='true')
            plt.xlabel('hours')
            plt.title('PlantLoad')
            plt.show()

            plt.plot(x, np.array(NetGrowth_list).reshape(-1, 1), color='r', label='pre')
            plt.plot(x, y[:3840, 3], color='b', label='true')
            plt.xlabel('hours')
            plt.title('NetGrowth')
            plt.show()

            # -------------
            goodness = [r2_FW, r2_LAI, r2_PlantLoad, r2_NetGrowth]
            mean_r2 = np.mean(goodness)
            goodness.append(mean_r2)
            print("%d cache score: %.2f" % (idx, mean_r2))

            # save
            # df = pd.DataFrame([goodness], columns=columns+['score'])
            # if os.path.exists(save_path):
            #     ori_df = pd.read_csv(save_path)
            #     df = ori_df.append(df)
            # df.to_csv(save_path, float_format='%.3f', index=False)

            # net1
            # PAR_R2.append(r2_PAR)
            # AirT_R2.append(r2_AirT)
            # AirRH_R2.append(r2_AirRH)
            # Airppm_R2.append(r2_Airppm)

            # # net2------------
            LAI_R2.append(r2_LAI)
            PlantLoad_R2.append(r2_PlantLoad)
            NetGrowth_R2.append(r2_NetGrowth)

            # net3
            FW_R2.append(r2_FW)

            score.append(mean_r2)
            # --------
        #
    # save_path = r'D:\py-code\Greenhouse\iGrow-main\MyiGrow\MyiGrow\table1'
    #
    # Table_pre = pd.DataFrame(pre, columns=columns)
    # Table_pre.to_csv(save_path + 'pre.csv', float_format='%.3f', index=False)
    #
    # Table_y = pd.DataFrame(y, columns=columns)
    # Table_y.to_csv(save_path + 'y.csv', float_format='%.3f', index=False)
    # mean
    # mean_PAR = np.mean(PAR_R2)
    # mean_AirT = np.mean(AirT_R2)
    # mean_AirRH = np.mean(AirRH_R2)
    # mean_Airppm = np.mean(Airppm_R2)
    # ------------
    mean_LAI = np.mean(LAI_R2)
    mean_PlantLoad = np.mean(PlantLoad_R2)
    mean_NetGrowth = np.mean(NetGrowth_R2)
    mean_FW = np.mean(FW_R2)
    mean_score = np.mean(score)

    goodness_of_simulator = [mean_FW, mean_LAI, mean_PlantLoad, mean_NetGrowth,
                             mean_score]

    # save
    Table1_df = pd.DataFrame([goodness_of_simulator],
                             columns=columns + ['score'])
    # Table1_df.to_csv(save_dir+'R2_of_simulator.csv',
    #                  float_format='%.3f', index=False)
    print("mean R2:")
    print(Table1_df.mean(axis=0))
    print(np.mean(total_loss))


if __name__ == '__main__':
    Table1()
    