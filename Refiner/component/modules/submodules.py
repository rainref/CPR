import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenQuantityRegressor(nn.Module):
    """Module that try to fit the number of tokens are masked by a regressor layer."""

    def __init__(self,args):
        super(TokenQuantityRegressor, self).__init__()
        self.pre_regressor = nn.Linear(args.final_emsize, args.submodel_dim)
        self.regressor = nn.Linear(args.submodel_dim, 1)
        self.dropout = nn.Dropout(args.dropout_emb)

    def forward(
            self,
            input_vec,
            num_subtoken):
        output=input_vec[:,0]
        output = self.pre_regressor(output)
        output = nn.ReLU()(output)
        output = self.dropout(output)
        output = self.regressor(output)[:,0]


        MSE_loss=F.mse_loss(output,num_subtoken, reduction='mean')

        return output,MSE_loss

class AutoWeightedLoss(nn.Module):
    def __init__(self, num_loss, args):
        super(AutoWeightedLoss, self).__init__()
        self.constant_weight=args.constant_weight
        self.num_loss = num_loss
        self.epoch=torch.zeros(1)
        if self.constant_weight is not None:
            x = torch.Tensor(self.constant_weight)
            self.now_weight = self.constant_weight.copy()
        else: x = torch.ones([num_loss])
        if args.use_cuda:
            x = x.cuda()
        self.coefs = nn.Parameter(data=x)
        self.decay_rate=0


    def forward(self, losses):
        loss_sum = 0
        coefs_final=[]
        if self.constant_weight is None:
            for i in range(self.num_loss):
                square = self.coefs[i] ** 2
                loss_sum += losses[i] / (2 * square) + torch.log(1 + square)
                coefs_final.append(1/ (2 * square) + torch.log(1 + square))
        else :
            self.now_weight[0]=self.constant_weight[0]+(self.epoch-1) * self.decay_rate
            self.now_weight[1] = 1-self.now_weight[0]
            if self.now_weight[0]>1:
                self.now_weight[0]=1
                self.now_weight[1] = 0
            for i in range(self.num_loss):
                loss_sum += self.now_weight[i] * losses[i]
                coefs_final.append(self.now_weight[i])

        return loss_sum, coefs_final