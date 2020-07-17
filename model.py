from layer import exp_model, SEIR_cell
import numpy as np
import torch
from pmdarima.arima import auto_arima
import torch.optim as optim
torch.set_default_tensor_type(torch.DoubleTensor)



def train_exp_decay_pred(beta_exp, lr=0.01, max_epoches=1000, pred_date_len=2):
    date_len = len(beta_exp) - 1
    beta_pred_model = exp_model(date_len)
    input_tensor = torch.tensor([beta_exp[0]])
    beta_pred = beta_pred_model(input_tensor)
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # loss_list = []
    optimizer = optim.Adam(beta_pred_model.parameters(),
                           lr=lr,
                           betas=(beta1, 0.999))
    # loss_min = 1e8
    input_tensor = torch.tensor([beta_exp[0]])
    gt_tensor = torch.tensor(beta_exp[1:])
    loss_fn = torch.nn.MSELoss()
    for epoch_step in range(max_epoches):
        beta_pred = beta_pred_model(input_tensor)
        loss = loss_fn(beta_pred[:-1], gt_tensor[:-1]) + 100 * loss_fn(
            beta_pred[-1], gt_tensor[-1])
        # print("Loss: {}".format(loss))
        learning_rate = lr_decay(epoch_step, lr, decay_steps=100)
        optimizer = optim.Adam(beta_pred_model.parameters(),
                               lr=learning_rate,
                               betas=(beta1, 0.999))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    beta_pred_val = beta_pred_model.pred(
        beta_exp[-1], pred_date_len=pred_date_len).detach().numpy()
    return beta_pred_val


class SEIR_model(torch.nn.Module):
    def __init__(self,
                 date_len,
                 pred_date_len=0,
                 N=2870000.,
                 E_ratio_init=3.,
                 I_init=41,
                 R_init=2.,
                 D_init=0.,
                 param={}):
        super(SEIR_model, self).__init__()
        self.SEIR_cells = torch.nn.ModuleList()
        self.SEIR_pred_cells = torch.nn.ModuleList()
        self.N = N
        self.E_ratio = E_ratio_init
        self.I = I_init
        self.E = (self.I * self.E_ratio)
        self.R = R_init
        self.D = D_init
        self.S = (self.N - self.I - self.E - self.R - self.D)
        self.date_len = date_len - 1
        self.pred_date_len = pred_date_len
        if param != {}:
            len_param = len(param['beta'])
            self.beta_save = param['beta']
            self.gamma_2_save = param['gamma_2']
            self.alpha_save = param['alpha']
            self.theta_save = param['theta']
            for i in range(len_param):
                beta = self.beta_save[i]
                gamma_2 = self.gamma_2_save[i]
                alpha = self.alpha_save[i]
                theta = self.theta_save[i]
                self.SEIR_cells.append(
                    SEIR_cell(self.N, beta, gamma_2, theta, alpha))
            if self.date_len > len_param:
                for i in range(len_param, self.date_len):
                    if len_param >= 1:
                        beta = self.beta_save[len_param - 1]
                        gamma_2 = self.gamma_2_save[len_param - 1]
                        alpha = self.alpha_save[len_param - 1]
                        theta = self.theta_save[len_param - 1]
                        self.SEIR_cells.append(
                            SEIR_cell(self.N, beta, gamma_2, theta, alpha))
                    else:
                        self.SEIR_cells.append(SEIR_cell(self.N))
        else:
            for i in range(self.date_len):
                self.SEIR_cells.append(SEIR_cell(self.N))

        self.S_tensor_cur = torch.zeros((self.date_len + 1, ))
        self.I_tensor_cur = torch.zeros((self.date_len + 1, ))
        self.E_tensor_cur = torch.zeros((self.date_len + 1, ))
        self.R_tensor_cur = torch.zeros((self.date_len + 1, ))
        self.D_tensor_cur = torch.zeros((self.date_len + 1, ))

    def forward(self, X):
        inp = self.S, self.I, self.E, self.R, self.D
        # param = beta_init, gamma_2_init
        S_tensor = torch.zeros((self.date_len + 1, ))
        I_tensor = torch.zeros((self.date_len + 1, ))
        E_tensor = torch.zeros((self.date_len + 1, ))
        R_tensor = torch.zeros((self.date_len + 1, ))
        D_tensor = torch.zeros((self.date_len + 1, ))
        S_tensor[0], I_tensor[0], E_tensor[0], R_tensor[0], D_tensor[0] = inp
        for i in range(self.date_len):
            if i == self.date_len - 1:  # we cannot update the last beta with grad
                self.beta = beta_cur
                self.SEIR_cells[i].update_beta(beta_cur)
            S, I, E, R, D, beta_cur, gamma_2_cur, theta_cur, alpha_cur = self.SEIR_cells[
                i](inp)
            S_tensor[i + 1], I_tensor[i + 1], E_tensor[i + 1], R_tensor[
                i + 1], D_tensor[i + 1] = S, I, E, R, D
            self.beta = beta_cur
            self.gamma_2 = gamma_2_cur
            self.theta = theta_cur
            self.alpha = alpha_cur
            self.S_cur = S
            self.I_cur = I
            self.E_cur = E
            self.R_cur = R
            self.D_cur = D
            inp = [S, I, E, R, D]
        self.S_tensor_cur, self.I_tensor_cur, self.E_tensor_cur, self.R_tensor_cur, self.D_tensor_cur = S_tensor, I_tensor, E_tensor, R_tensor, D_tensor
        return S_tensor, I_tensor, E_tensor, R_tensor, D_tensor, beta_cur, gamma_2_cur

    def pred(self, pred_date_len, param={}):
        check_positive_replace = lambda x, y: [
            yi if xi <= 0 else xi for xi, yi in zip(x, y)
        ]
        # check_positive_replace
        N_cur_list = [self.N] * pred_date_len
        beta_list = [self.beta] * pred_date_len
        gamma_2_list = [self.gamma_2] * pred_date_len
        theta_list = [self.theta] * pred_date_len
        alpha_list = [self.alpha] * pred_date_len
        if param == {}:
            N_cur = N_cur_list
            beta = beta_list
            gamma_2 = gamma_2_list
            theta = theta_list
            alpha = alpha_list
        else:
            N_cur = N_cur_list
            for k in param.keys():
                if len(param[k]) == 1:
                    param[k] = param[k] * pred_date_len
            beta = check_positive_replace(param['beta'],
                                          [0.] * len(param['beta']))
            gamma_2 = check_positive_replace(param['gamma_2'], gamma_2_list)
            theta = check_positive_replace(param['theta'], theta_list)
            alpha = check_positive_replace(param['alpha'],
                                           [0.] * len(param['alpha']))
        cur_pred_cells_len = len(self.SEIR_pred_cells)
        # print("cur_pred_cells_len:", cur_pred_cells_len)
        if cur_pred_cells_len != pred_date_len:
            self.SEIR_pred_cells = torch.nn.ModuleList()
            for i in range(pred_date_len):
                self.SEIR_pred_cells.append(
                    SEIR_cell(N_cur[i], beta[i], gamma_2[i], theta[i],
                              alpha[i]))
        S_pred_tensor = torch.zeros((pred_date_len, ))
        I_pred_tensor = torch.zeros((pred_date_len, ))
        E_pred_tensor = torch.zeros((pred_date_len, ))
        R_pred_tensor = torch.zeros((pred_date_len, ))
        D_pred_tensor = torch.zeros((pred_date_len, ))
        # pred:
        inp = self.S_cur, self.I_cur, self.E_cur, self.R_cur, self.D_cur
        for i in range(pred_date_len):
            S, I, E, R, D, beta_, gamma_2_, theta_, alpha_ = self.SEIR_pred_cells[
                i](inp)
            S_pred_tensor[i], I_pred_tensor[i], E_pred_tensor[
                i], R_pred_tensor[i], D_pred_tensor[i] = S, I, E, R, D
            inp = [S, I, E, R, D]
        return S_pred_tensor, I_pred_tensor, E_pred_tensor, R_pred_tensor, D_pred_tensor

    def beta_pred(self, beta_list_square, pred_date_len=1 + 1):
        # check_positive = lambda x:0 if x <=0 else np.sqrt(x)
        stepwise_model = auto_arima(beta_list_square,
                                    trace=False,
                                    information_criterion='aic',
                                    with_intercept=False,
                                    error_action='ignore',
                                    suppress_warnings=True)
        stepwise_model.fit(beta_list_square)
        beta_future_forecast = stepwise_model.predict(n_periods=pred_date_len)
        return beta_future_forecast

    def param_pred(self,
                   beta_list,
                   gamma_2_list,
                   theta_list,
                   alpha_list,
                   exp_decay=True,
                   pred_date_len=1):
        check_positive = lambda x: 0. if x <= 0 else np.sqrt(x)

        param_dict = {}
        if len(beta_list[:-1]) - np.argmax(beta_list[:-1]) <= 3:
            exp_decay = False
            print(
                "the beta list is so late to the max place! So we use arima instead of exp_decay! "
            )

        # for beta pred
        if exp_decay:
            beta_list_square = np.square(beta_list)
            beta_future_forecast = train_exp_decay_pred(
                beta_list_square[:-1], pred_date_len=pred_date_len + 1)
            beta_future_forecast_pos = [
                check_positive(b) for b in beta_future_forecast
            ]
            # print("beta_future_forecast_pos:",beta_future_forecast_pos)
            self.SEIR_cells[-1].update_beta(beta_future_forecast_pos[0])
            param_dict['beta'] = beta_future_forecast_pos[1:]
        else:
            beta_list_square = np.square(beta_list)
            beta_future_forecast = self.beta_pred(beta_list_square[:-1],
                                                  pred_date_len=pred_date_len +
                                                  1)
            beta_future_forecast_pos = [
                check_positive(b) for b in beta_future_forecast
            ]
            self.SEIR_cells[-1].update_beta(beta_future_forecast_pos[0])
            param_dict['beta'] = beta_future_forecast_pos[1:]

        # for gamma_2
        if exp_decay:
            sqrt_datas = [gamma_2_list, theta_list]
            params = ['gamma_2', 'theta']
            alpha_list_square = np.square(alpha_list)
            alpha_future_forecast = train_exp_decay_pred(
                alpha_list_square, pred_date_len=pred_date_len)
            alpha_future_forecast_pos = [
                check_positive(a) for a in alpha_future_forecast
            ]
            # print("alpha_future_forecast_pos:",alpha_future_forecast_pos)
            param_dict['alpha'] = alpha_future_forecast_pos
        else:
            sqrt_datas = [gamma_2_list, theta_list, alpha_list]
            params = ['gamma_2', 'theta', 'alpha']

        datas = [(np.square(d)) for d in sqrt_datas]
        for i in range(len(params)):
            stepwise_model = auto_arima(datas[i],
                                        trace=False,
                                        error_action='ignore',
                                        information_criterion='aic',
                                        with_intercept=False,
                                        suppress_warnings=True)
            stepwise_model.fit(datas[i])
            future_forecast = stepwise_model.predict(n_periods=pred_date_len)
            param_dict[params[i]] = [
                check_positive(p) for p in future_forecast
            ]

        return param_dict