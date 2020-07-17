# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import os
from model import SEIR_model
from ops import cal_acc_confirm, get_data_acc_confirm, cal_new_confirm, cal_new_R, plot_daily_acc, \
    plot_daily_rec, plot_daily_new, plot_daily_new_rec, make_dir, load_param, plot_SEIRD, lr_decay, save_param
torch.set_default_tensor_type(torch.DoubleTensor)


def train(data,
          model_city_date_path,
          lr_init=0.01,
          N=1e7,
          I_init=1e-6,
          R_init=1e-6 / 2.,
          D_init=1e-6 / 6.,
          cured_ratio=20.,
          dead_ratio=20.,
          param={},
          features=['I', 'cured', 'dead'],
          max_epoches=6000,
          decay_steps=300):
    model_pt = os.path.join(model_city_date_path, 'model.pt')
    data_feat = data[features]
    Input = np.array(data_feat, dtype=np.float)
    print(Input.shape)
    date_len = len(Input)
    print(date_len)
    model = SEIR_model(date_len,
                       pred_date_len=10,
                       N=N,
                       I_init=I_init,
                       R_init=R_init,
                       D_init=D_init,
                       param=param)

    lr = lr_init
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    loss_list = []
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))
    loss_min = 1e8
    for epoch_step in range(max_epoches):
        print("Training step: ", epoch_step)
        Input = torch.tensor(Input)
        model_inp = Input[:-1]
        S, I, E, R, D, beta, gamma_2 = model(model_inp.float())
        loss_fn = torch.nn.MSELoss()
        pred_I = I
        pred_recovered = R
        pred_dead = D
        pred_confirmed = I + R + D

        I_gt_tensor = Input[:, 0]
        recovered_gt_tensor = Input[:, 1]
        dead_gt_tensor = Input[:, 2]
        confirmed_gt_tensor = I_gt_tensor + recovered_gt_tensor + dead_gt_tensor
        # cured_ratio = 2*cured_ratio
        # dead_ratio = 2*dead_ratio
        loss = (loss_fn(pred_confirmed, confirmed_gt_tensor) +
                cured_ratio * loss_fn(pred_recovered, recovered_gt_tensor) +
                dead_ratio * loss_fn(pred_dead, dead_gt_tensor) +
                loss_fn(pred_I, I_gt_tensor)) / (4 + cured_ratio + dead_ratio)
        print("Loss: {}".format(loss))
        loss_list.append(loss)
        if loss < loss_min:
            loss_min = loss
            torch.save(model, model_pt)
        learning_rate = lr_decay(epoch_step, lr, decay_steps=decay_steps)
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate,
                               betas=(beta1, 0.999))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Loss_min:", loss_min)
    save_param(model, model_city_date_path)
    return S, I, E, R, D, loss_list


def load_model_predict(model_city_date_path,
                       data,
                       param_pred=True,
                       exp_decay=True,
                       city_name='深圳',
                       c='confirmed',
                       features=['I', 'cured', 'dead'],
                       return_new_C_R=False,
                       pred_date_len=5):
    I_name, recover_name, dead_name = features
    model_pt = os.path.join(model_city_date_path, 'model.pt')
    model = torch.load(model_pt)
    S = model.S_tensor_cur
    E = model.E_tensor_cur
    I = model.I_tensor_cur
    R = model.R_tensor_cur
    D = model.D_tensor_cur

    I_pred_old_float = (I.detach().numpy())
    R_pred_old_float = (R.detach().numpy())
    D_pred_old_float = (D.detach().numpy())
    # S_pred_old_float = (S.detach().numpy())
    # E_pred_old_float = (E.detach().numpy())

    # I_pred_old = (I.detach().numpy()).astype(np.int)
    # R_pred_old = (R.detach().numpy()).astype(np.int)
    # D_pred_old = (D.detach().numpy()).astype(np.int)
    # S_pred_old = (S.detach().numpy()).astype(np.int)
    # E_pred_old = (E.detach().numpy()).astype(np.int)

    confirm_pred = cal_acc_confirm(I_pred_old_float, R_pred_old_float,
                                   D_pred_old_float)
    confirm_origin = get_data_acc_confirm(data, c=c)
    # print(confirm_origin)
    new_confirm = cal_new_confirm(np.array(data[I_name]),
                                  np.array(data[recover_name]),
                                  np.array(data[dead_name]))
    new_confirm_pred = cal_new_confirm(I_pred_old_float, R_pred_old_float,
                                       D_pred_old_float)

    if param_pred:
        beta = []
        theta = []
        gamma_2 = []
        alpha = []
        for i in range(len(model.SEIR_cells)):
            beta.append(model.SEIR_cells[i].beta.detach().numpy()[0])
            gamma_2.append(model.SEIR_cells[i].gamma_2.detach().numpy()[0])
            theta.append(model.SEIR_cells[i].theta.detach().numpy()[0])
            alpha.append(model.SEIR_cells[i].alpha.detach().numpy()[0])
        # if city_name=='深圳':
        #     theta=get_recent_curve(theta)
        # print(len(theta))
        param = model.param_pred(beta,
                                 gamma_2,
                                 theta,
                                 alpha,
                                 exp_decay=exp_decay,
                                 pred_date_len=pred_date_len)

        # print(param)
        S_pred_tensor, I_pred_tensor, E_pred_tensor, R_pred_tensor, D_pred_tensor = model.pred(
            param=param, pred_date_len=pred_date_len)
    else:
        S_pred_tensor, I_pred_tensor, E_pred_tensor, R_pred_tensor, D_pred_tensor = model.pred(
            pred_date_len=pred_date_len)

    I_pred_new_float = (I_pred_tensor.detach().numpy())
    R_pred_new_float = (R_pred_tensor.detach().numpy())
    D_pred_new_float = (D_pred_tensor.detach().numpy())
    # S_pred_new_float = (S_pred_tensor.detach().numpy())
    # E_pred_new_float = (E_pred_tensor.detach().numpy())

    # I_pred_new = (I_pred_tensor.detach().numpy()).astype(np.int)
    # R_pred_new = (R_pred_tensor.detach().numpy()).astype(np.int)
    # D_pred_new = (D_pred_tensor.detach().numpy()).astype(np.int)
    # S_pred_new = (S_pred_tensor.detach().numpy()).astype(np.int)
    # E_pred_new = (E_pred_tensor.detach().numpy()).astype(np.int)

    I_pred_total = np.concatenate((I_pred_old_float, I_pred_new_float), axis=0)
    R_pred_total = np.concatenate((R_pred_old_float, R_pred_new_float), axis=0)
    D_pred_total = np.concatenate((D_pred_old_float, D_pred_new_float), axis=0)
    # S_pred_total = np.concatenate((S_pred_old_float, S_pred_new_float), axis=0)
    # E_pred_total = np.concatenate((E_pred_old_float, E_pred_new_float), axis=0)

    plot_SEIRD(data,
               I=I_pred_total,
               R=R_pred_total,
               D=D_pred_total,
               city=city_name,
               pred_date_len=pred_date_len)

    confirm_pred = cal_acc_confirm(I_pred_total, R_pred_total, D_pred_total)
    confirm_origin = get_data_acc_confirm(data, c=c)
    plot_daily_acc(data,
                   confirm_origin,
                   confirm_pred,
                   city=city_name,
                   pred_date_len=pred_date_len)
    plot_daily_rec(data,
                   R_pred_old,
                   R_pred_total,
                   city=city_name,
                   pred_date_len=pred_date_len)
    # print("!!!!!!!!!!!!!!max_confirm_pred:",max(confirm_pred))
    new_confirm_pred_total = cal_new_confirm(I_pred_total, R_pred_total,
                                             D_pred_total)
    plot_daily_new(data,
                   new_confirm,
                   new_confirm_pred_total,
                   city=city_name,
                   pred_date_len=pred_date_len)
    new_R_old = cal_new_R(R_pred_old_float)
    new_R_pred_total = cal_new_R(R_pred_total)
    plot_daily_new_rec(data,
                       new_R_old,
                       new_R_pred_total,
                       city=city_name,
                       pred_date_len=pred_date_len)
    # print(new_R_pred_total)

    # print("!!!!!!\nN:\n",S_pred_total+E_pred_total+I_pred_total+R_pred_total+D_pred_total)
    if return_new_C_R:
        return model, new_confirm_pred_total, new_R_pred_total
    else:
        return model


def train_with_city_data(data,
                         N,
                         date,
                         cityname='深圳',
                         lr_init=0.01,
                         max_epoches=2000,
                         is_train=True,
                         load_param_save=False,
                         param_path='',
                         decay_steps=300):
    city_pinyin = {
        'US': 'US',
        '深圳': 'shenzhen',
        '湖北': 'hubei',
        '武汉': 'wuhan',
        '全国': 'china',
        '非湖北': 'nohubei'
    }
    pinyin = city_pinyin[cityname]
    model_city_date_path = make_dir(pinyin, date)
    features = ['I', 'cured', 'dead']
    I_init = float(data['I'].iloc[0])
    R_init = float(data['cured'].iloc[0])
    D_init = float(data['dead'].iloc[0])
    N = N
    cured_ratio = float(
        data['I'].mean() /
        data['cured'].mean()) if data['cured'].mean() != 0 else 50.
    dead_ratio = float(
        data['I'].mean() /
        data['dead'].mean()) if data['dead'].mean() != 0 else 50.
    print('cured_ratio:', cured_ratio)
    print('dead_ratio:', dead_ratio)
    param = {}
    if load_param_save:
        if param_path == '':
            param_path = model_city_date_path
        param = load_param(param_path)
    print(param)
    #  train里面会保存模型
    if is_train:
        S, I, E, R, D, loss_list = train(data,
                                         model_city_date_path,
                                         lr_init=lr_init,
                                         N=N,
                                         I_init=I_init,
                                         R_init=R_init,
                                         D_init=D_init,
                                         cured_ratio=cured_ratio,
                                         dead_ratio=dead_ratio,
                                         features=features,
                                         max_epoches=max_epoches,
                                         param=param,
                                         decay_steps=decay_steps)
        plt.plot(range(len(loss_list)),
                 loss_list,
                 color='darkorange',
                 label='loss training',
                 marker='x')
    return model_city_date_path
