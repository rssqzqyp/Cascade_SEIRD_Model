import torch
from torch.nn.parameter import Parameter
torch.set_default_tensor_type(torch.DoubleTensor)


class fc_module(torch.nn.Module):
    def __init__(self, w1, b1):
        super(fc_module, self).__init__()
        self.w1 = w1
        self.b1 = b1

    def forward(self, X):
        return self.w1 * X + self.b1


class exp_model(torch.nn.Module):
    def __init__(self, date_len):
        super(exp_model, self).__init__()
        self.w1 = Parameter(torch.tensor([0.8]), requires_grad=True)
        # self.w2 = Parameter(torch.tensor([1.],requires_grad=True))
        self.b1 = Parameter(torch.tensor([0.]), requires_grad=False)
        self.b2 = Parameter(torch.tensor([0.]), requires_grad=False)
        self.date_len = date_len
        # self.fc_modules = torch.nn.ModuleList()
        # for i in range(self.date_len):
        #     self.fc_modules.append(fc_module(self.w1,self.b1))
        self.fc = fc_module(self.w1, self.b1)

    def forward(self, X):
        pred_tensor = torch.zeros((self.date_len, ))
        # out = self.fc_modules[0](X)
        out = self.fc(X)
        pred_tensor[0] = out
        for i in range(1, self.date_len):
            # out = self.fc_modules[i](out)
            out = self.fc(out) + self.b2
            pred_tensor[i] = out
        return pred_tensor

    def pred(self, X, pred_date_len=1):
        pred_tensor = torch.zeros((pred_date_len, ))
        p = X
        pred_tensor[0] = p
        # print(self.w1)
        # print(self.b1)
        for i in range(1, pred_date_len):
            p = fc_module(self.w1, self.b1)(p) + self.b2
            # print(p)
            pred_tensor[i] = p
        return pred_tensor


class SEIR_cell(torch.nn.Module):
    def __init__(self,
                 N,
                 beta_init=0.2586,
                 gamma_2_init=0.018,
                 theta_init=0.001,
                 alpha_init=0.2):
        super(SEIR_cell, self).__init__()
        # self.date_len = date_len
        self.beta = Parameter(torch.tensor([beta_init]), requires_grad=True)
        self.N = Parameter(torch.tensor([N], requires_grad=False))
        self.gamma_2 = Parameter(torch.tensor([gamma_2_init]),
                                 requires_grad=True)
        # self.gamma_2 = Parameter(torch.tensor([0.5], requires_grad=True))
        self.alpha = Parameter(torch.tensor([alpha_init]), requires_grad=True)
        self.theta = Parameter(torch.tensor([theta_init]), requires_grad=True)
        # self.theta = Parameter(torch.tensor([0.2], requires_grad=True))
        # self.E_ratio = Parameter(torch.tensor([3.], requires_grad=True))

    def clamp(self, X):
        # return torch.clamp(X, min=0, max=self.N)
        return X

    def act(self, X):
        return torch.pow(X, 2)

    def forward(self, X):
        S, confirm, Exposed, recover, dead = X
        # self.beta = beta_old + self.beta_add
        # self.gamma_2 = gamma_2_old + self.gamma_2_add
        S_rest = S - self.act(self.beta) * confirm * S / self.N  # dS/dt
        E = Exposed + self.act(self.beta) * confirm * S / self.N - self.act(
            self.alpha) * Exposed  # dE/dt

        I = confirm + self.act(self.alpha) * Exposed - self.act(
            self.gamma_2) * confirm - self.act(self.theta) * confirm  # dI/dt
        R = recover + self.act(self.gamma_2) * confirm  # dR/dt
        D = dead + self.act(self.theta) * confirm

        # I = confirm + self.act(self.alpha)*E - self.act(self.gamma_2)*confirm - self.act(self.theta)*confirm # dI/dt
        # R = recover + self.act(self.gamma_2)*I # dR/dt
        # D = dead + self.act(self.theta)*I

        return S_rest, I, E, R, D, self.beta, self.gamma_2, self.theta, self.alpha

    def update_beta(self, b):
        self.beta = Parameter(torch.tensor([b]), requires_grad=True)