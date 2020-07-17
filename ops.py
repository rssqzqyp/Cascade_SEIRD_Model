# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


def plot_SEIRD(data, I, R, D, xlen=10, city='武汉', pred_date_len=0):
    def format_datetime(x):
        xd = x.date()
        fxd = f'{xd.month}.{xd.day}'
        return fxd

    # len_data = len(list(set(data.index)))
    # print(len_data)
    plt.figure(figsize=(xlen, 6))
    T_name = 'time'
    time_val = data[T_name].values
    max_time_val = data[T_name].values.max()
    pred_time = []
    for i in range(1, pred_date_len + 1):
        pred_time.append(max_time_val + np.timedelta64(i, 'D'))
    # print(pred_time)
    if pred_time == []:
        merge_time = time_val
    else:
        merge_time = np.concatenate((time_val, pred_time), axis=0)
    merge_time = [format_datetime(pd.to_datetime(d)) for d in merge_time]
    plt.plot(merge_time, I, color='red', label='I-感染人数', marker='.')
    plt.plot(merge_time, R, color='blue', label='R-治愈人数', marker='.')
    # plt.plot(merge_time, S, color = 'darkgreen',label = 'S-易感人群',marker = '.')
    # plt.plot(merge_time, E, color = 'darkorange',label = 'E-疑似人数',marker = '.')
    plt.plot(merge_time, D, color='black', label='D-死亡人数', marker='.')

    I_max = np.argmax(I)
    # for a,b in zip(merge_time, I):
    a = merge_time[I_max]
    b = I[I_max]
    plt.annotate(f'{a},{b}',
                 xy=(a, b),
                 xytext=(-5, 5),
                 textcoords='offset points',
                 color='red')
    # for a,b in zip(merge_time, S):
    #     plt.annotate('%s'%(b),xy=(a,b),xytext=(-5,20), textcoords='offset points',color='darkgreen')

    city_title = '疫情状况-' + city
    plt.title(city_title)
    plt.legend()
    plt.xlabel('日期')
    plt.ylabel('人数')
    plt.show()


def plot_param(model, city_name, data, xlen=10, pred_date_len=5):
    T_name = 'time'

    time_val = data[T_name].values

    max_time_val = data[T_name].values.max()
    pred_time = []
    for i in range(1, pred_date_len + 1):
        pred_time.append(max_time_val + np.timedelta64(i, 'D'))
    if pred_time == []:
        merge_time = time_val
    else:
        merge_time = np.concatenate((time_val, pred_time), axis=0)

    def format_datetime(x):
        xd = x.date()
        fxd = ''
        if xd.month < 10 and xd.day < 10:
            fxd = f'0{xd.month}-0{xd.day}'
        elif xd.month < 10 and xd.day >= 10:
            fxd = f'0{xd.month}-{xd.day}'
        else:
            fxd = f'{xd.month}-{xd.day}'
        return fxd

    dates_list = [format_datetime(pd.to_datetime(d)) for d in merge_time]
    plt.figure(figsize=(xlen, 10))
    beta = []
    gamma_2 = []
    theta = []
    alpha = []
    # omega = []
    for i in range(len(model.SEIR_cells)):
        beta.append(model.SEIR_cells[i].beta.detach().numpy()[0])
        gamma_2.append(model.SEIR_cells[i].gamma_2.detach().numpy()[0])
        theta.append(model.SEIR_cells[i].theta.detach().numpy()[0])
        alpha.append(model.SEIR_cells[i].alpha.detach().numpy()[0])
        # omega.append((model.SEIR_cells[i].theta.detach().numpy()[0])*0.)

    for i in range(len(model.SEIR_pred_cells)):
        beta.append(model.SEIR_pred_cells[i].beta.detach().numpy()[0])
        gamma_2.append(model.SEIR_pred_cells[i].gamma_2.detach().numpy()[0])
        theta.append(model.SEIR_pred_cells[i].theta.detach().numpy()[0])
        alpha.append(model.SEIR_pred_cells[i].alpha.detach().numpy()[0])
    # print('beta:',beta)
    # print('gamma_2:',gamma_2)
    # print('theta:',theta)
    # print('alpha:',alpha)
    # print('omega:',omega)
    plot_title = ['beta-感染率', 'gamma_2-治愈率', 'theta-死亡率', 'alpha-(疑似->感染)率']
    plot_list_sqrt = [beta, gamma_2, theta, alpha]
    plot_list = [np.square(p) for p in plot_list_sqrt]
    colors = ['blue', 'darkgreen', 'darkorange', 'red']
    for i in range(len(colors)):
        plt.plot(dates_list[:len(beta)],
                 plot_list[i],
                 color=colors[i],
                 label=plot_title[i],
                 marker='x')
    for a, b in zip(range(len(beta)), beta):
        plt.annotate('%.4f' % (b),
                     xy=(a, b),
                     xytext=(-2, 2),
                     textcoords='offset points',
                     color=colors[0])
    plt.plot([len(model.SEIR_cells), len(model.SEIR_cells)], [0, 1])
    plt.legend()
    title = 'Param changing process-' + city_name
    plt.title(title)
    plt.show()


def plot_daily_acc(data,
                   accumulated_confirmed,
                   accumulated_pred_confirmed,
                   xlen=10,
                   city=u'武汉',
                   pred_date_len=0):
    T_name = 'time'
    plt.figure(figsize=(xlen, 6))
    time_val = data[T_name].values

    max_time_val = data[T_name].values.max()
    pred_time = []
    for i in range(1, pred_date_len + 1):
        pred_time.append(max_time_val + np.timedelta64(i, 'D'))
    if pred_time == []:
        merge_time = time_val
    else:
        merge_time = np.concatenate((time_val, pred_time), axis=0)
    plt.plot(time_val,
             accumulated_confirmed,
             color='red',
             label='累计确诊人数',
             marker='x')
    plt.plot(merge_time,
             accumulated_pred_confirmed,
             color='blue',
             label='预测的累计确诊人数',
             marker='x')
    for a, b in zip(merge_time, accumulated_pred_confirmed):
        plt.annotate('%s' % (b),
                     xy=(a, b),
                     xytext=(-5, 5),
                     textcoords='offset points',
                     color='blue')
    for a, b in zip(time_val, accumulated_confirmed):
        plt.annotate('%s' % (b),
                     xy=(a, b),
                     xytext=(-5, 20),
                     textcoords='offset points',
                     color='red')
    city_title = u'疫情状况-' + city
    plt.title(city_title)
    plt.legend()
    plt.xlabel(u'日期')
    plt.ylabel(u'人数')
    # plt.savefig(city+u'累计预测')
    plt.show()
    return


def plot_daily_rec(data,
                   accumulated_rec,
                   accumulated_pred_rec,
                   xlen=10,
                   city=u'武汉',
                   pred_date_len=0):
    T_name = 'time'
    plt.figure(figsize=(xlen, 6))
    time_val = data[T_name].values

    max_time_val = data[T_name].values.max()
    pred_time = []
    for i in range(1, pred_date_len + 1):
        pred_time.append(max_time_val + np.timedelta64(i, 'D'))
    if pred_time == []:
        merge_time = time_val
    else:
        merge_time = np.concatenate((time_val, pred_time), axis=0)
    plt.plot(time_val,
             accumulated_rec,
             color='red',
             label='累计确诊人数',
             marker='x')
    plt.plot(merge_time,
             accumulated_pred_rec,
             color='blue',
             label='预测的累计确诊人数',
             marker='x')
    for a, b in zip(merge_time, accumulated_pred_rec):
        plt.annotate('%s' % (b),
                     xy=(a, b),
                     xytext=(-5, 5),
                     textcoords='offset points',
                     color='blue')
    for a, b in zip(time_val, accumulated_rec):
        plt.annotate('%s' % (b),
                     xy=(a, b),
                     xytext=(-5, 20),
                     textcoords='offset points',
                     color='red')
    city_title = u'疫情状况-' + city
    plt.title(city_title)
    plt.legend()
    plt.xlabel(u'日期')
    plt.ylabel(u'人数')
    # plt.savefig(city+u'累计预测')
    plt.show()
    return


def plot_daily_new_rec(data,
                       new_rec,
                       pred_new_rec,
                       xlen=10,
                       city=u'武汉',
                       pred_date_len=0):
    plt.figure(figsize=(xlen, 6))
    T_name = 'time'
    time_val = data[T_name].values
    time_val = time_val[1:]
    max_time_val = data[T_name].values.max()
    pred_time = []
    for i in range(1, pred_date_len + 1):
        pred_time.append(max_time_val + np.timedelta64(i, 'D'))
    if pred_time == []:
        merge_time = time_val
    else:
        merge_time = np.concatenate((time_val, pred_time), axis=0)
    plt.plot(time_val, new_rec, color='red', label='新增确诊人数', marker='x')
    plt.plot(merge_time,
             pred_new_rec,
             color='blue',
             label='预测新增确诊人数',
             marker='x')
    for a, b in zip(merge_time, pred_new_rec):
        plt.annotate('%s' % (b),
                     xy=(a, b),
                     xytext=(-5, 5),
                     textcoords='offset points',
                     color='blue')
    for a, b in zip(time_val, new_rec):
        plt.annotate('%s' % (b),
                     xy=(a, b),
                     xytext=(-5, 20),
                     textcoords='offset points',
                     color='red')
    city_title = u'疫情状况-' + city
    plt.title(city_title)
    plt.legend()
    plt.xlabel('日期')
    plt.ylabel('人数')
    # plt.savefig(city+u'新增预测')
    plt.show()
    return


def plot_daily_new(data,
                   new_confirm,
                   pred_new_confirm,
                   xlen=10,
                   city=u'武汉',
                   pred_date_len=0):
    plt.figure(figsize=(xlen, 6))
    T_name = 'time'
    time_val = data[T_name].values
    time_val = time_val[1:]
    max_time_val = data[T_name].values.max()
    pred_time = []
    for i in range(1, pred_date_len + 1):
        pred_time.append(max_time_val + np.timedelta64(i, 'D'))
    if pred_time == []:
        merge_time = time_val
    else:
        merge_time = np.concatenate((time_val, pred_time), axis=0)
    plt.plot(time_val, new_confirm, color='red', label='新增确诊人数', marker='x')
    plt.plot(merge_time,
             pred_new_confirm,
             color='blue',
             label='预测新增确诊人数',
             marker='x')
    for a, b in zip(merge_time, pred_new_confirm):
        plt.annotate('%s' % (b),
                     xy=(a, b),
                     xytext=(-5, 5),
                     textcoords='offset points',
                     color='blue')
    for a, b in zip(time_val, new_confirm):
        plt.annotate('%s' % (b),
                     xy=(a, b),
                     xytext=(-5, 20),
                     textcoords='offset points',
                     color='red')
    city_title = u'疫情状况-' + city
    plt.title(city_title)
    plt.legend()
    plt.xlabel('日期')
    plt.ylabel('人数')
    # plt.savefig(city+u'新增预测')
    plt.show()
    return


def cal_acc_confirm(I, R, D):
    return np.ceil(I + R + D).astype(int)


def cal_new_confirm(I, R, D):
    acc_confirm = cal_acc_confirm(I, R, D)
    new_confirm = np.zeros((len(acc_confirm) - 1))
    for i in range(len(acc_confirm) - 1):
        new_confirm[i] = int(acc_confirm[i + 1] - acc_confirm[i])
    return new_confirm


def cal_new_R(R):
    new_R = np.zeros((len(R) - 1))
    for i in range(len(R) - 1):
        new_R[i] = np.ceil(R[i + 1] - R[i])
    return new_R


def get_data_acc_confirm(data, c='confirmed'):
    return np.array(data[c])


def save_param(model, model_city_date_path):
    save_path = os.path.join(model_city_date_path, 'params/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    beta = []
    theta = []
    gamma_2 = []
    alpha = []
    for i in range(len(model.SEIR_cells)):
        beta.append(model.SEIR_cells[i].beta.detach().numpy()[0])
        gamma_2.append(model.SEIR_cells[i].gamma_2.detach().numpy()[0])
        theta.append(model.SEIR_cells[i].theta.detach().numpy()[0])
        alpha.append(model.SEIR_cells[i].alpha.detach().numpy()[0])
    param = {'beta': beta, 'theta': theta, 'gamma_2': gamma_2, 'alpha': alpha}
    with open(save_path + 'param.pkl', 'wb') as f:
        pickle.dump(param, f, pickle.HIGHEST_PROTOCOL)


def load_param(model_city_date_path):
    save_path = os.path.join(model_city_date_path, 'params/')
    with open(save_path + 'param.pkl', 'rb') as f:
        return pickle.load(f)


def lr_decay(global_step, learning_rate=0.01, decay_rate=0.8, decay_steps=300):
    decayed_learning_rate = learning_rate * np.power(
        decay_rate, (global_step / decay_steps))
    return decayed_learning_rate


def read_data(path):
    data = pd.read_csv(path)
    data['I'] = data['confirmed'] - data['dead'] - data['cured']
    data['I/cured'] = data['I'] / data['cured']
    data['I/dead'] = data['I'] / data['dead']

    if 'nation' in path:
        data['E'] = data['suspected'] + data['close_contact'] + data[
            'under_medical_observation']
    data['time'] = pd.to_datetime(data['time'])
    return data

def make_dir(city, date):
    save_root_path = 'models/'
    model_city_path = os.path.join(save_root_path, city)

    model_city_date_path = os.path.join(model_city_path, date)

    if not os.path.exists(model_city_date_path):
        print(model_city_date_path)
        os.makedirs(model_city_date_path)
    return model_city_date_path