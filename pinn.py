import paddle
import numpy as np
import pandas as pd
import plotly.io

import network
import loss
import csv
import os
import matplotlib.pyplot as plt
# calculation precision setting
precision = 'double'
if precision == 'single':
    d = 'float32'
if precision == 'double':
    d = 'float64'
paddle.set_default_dtype(d)

# network setting
net_name = 'FCnNet'
numFC = 32 #神经网络各个层
numLayer = 6  #层数
numIN = 2  #这里有两个自变量了，一个是时间t，一个是距离
numOUT = 3 #输出，SFC，一般没有算上I

# optimizer setting
opt_name = 'Adam' # Adam/SGD/Momentum/LBFGS
lr = 1e-4
WD = 1e-6

# iteration setting
epoch_read = 0
epoch_start = epoch_read + 1
epoch_end = 100000
check_point = 10000
weightEQ = 1
weightSUP = 1
weightIC = 1

## load training data
data_pd = pd.read_csv('data_pde.CSV')
data = np.array(data_pd)
data_supervise = data #ndarray,50*2 t,dCdt

X_supervise = data_supervise[:,0] + 0.0  #t 50
C_supervise = data_supervise[:,1] + 0.0 #c 50

numTrain = 2000
t_start = 0
t_end = 49
X_train = np.linspace(t_start,t_end,numTrain)
Y_train = np.random.random((numTrain,numOUT))


# convert CPU data to GPU data
X_supervise = paddle.to_tensor(X_supervise)
X_supervise = X_supervise.reshape([50,1])
C_supervise = paddle.to_tensor(C_supervise)
C_supervise = C_supervise.reshape([50,1])
X_train = paddle.to_tensor(X_train)
X_train = X_train.reshape([numTrain,1])
Y_train = paddle.to_tensor(Y_train)
X_ic = paddle.zeros(shape=[1,1],dtype='float64')   #构造出一个全0，一个元素的二维矩阵
Y_ic = paddle.ones(shape=[1,3],dtype='float64')   #构造出一个全1，3个元素，二维矩阵


print(X_supervise)
print(C_supervise)
print(X_train)
print(Y_train)
print(X_ic)
print(Y_ic)


################################################################################
class FCnmodel(paddle.nn.Layer):

    # define network structure
    def __init__(self, numFC, numLayer, numIN, numOUT):
        super(FCnmodel, self).__init__()
        self.numFC = numFC
        self.numLayer = numLayer
        self.numIN = numIN
        self.numOUT = numOUT
        # IN layer
        self.IN = paddle.nn.Sequential(
            (paddle.nn.Linear(self.numIN, self.numFC)),
            (paddle.nn.Tanh())
        )
        # FC layer
        for i in range(self.numLayer - 2):
            FC_tmp = paddle.nn.Sequential(
                (paddle.nn.Linear(self.numFC, self.numFC)),
                (paddle.nn.Tanh())
            )
            exec("self.FC%s=FC_tmp" % i)
        # OUT layer
        self.OUT = paddle.nn.Sequential(
            (paddle.nn.Linear(self.numFC, self.numOUT))
        )
        # alpha:k1, beta:k2, p:k3
        self.k1 = paddle.create_parameter(shape=[1], dtype='float64')
        self.k2 = paddle.create_parameter(shape=[1], dtype='float64')
        self.k3 = paddle.create_parameter(shape=[1], dtype='float64')
        self.add_parameter('k1', self.k1)
        self.add_parameter('k2', self.k2)
        self.add_parameter('k3', self.k3)

    # forward calculation
    def forward(self, inputs):
        self.u = inputs
        self.u = self.IN(self.u)
        for i in range(self.numLayer - 2):
            exec("self.u=self.FC%s(self.u)" % i)
        self.u = self.OUT(self.u)
        self.u = paddle.exp(self.u)
        return self.u


## import network model
if net_name == 'FCnNet':
    nn_fun = network.FCnmodel(numFC,numLayer,numIN,numOUT)

params_info = paddle.summary(nn_fun,input=X_supervise)
print(params_info)

# if you want to check the network parameters
for name, param in nn_fun.named_parameters():
    print(f"Layer: {name}")
    print(param)


# Data Loss (L2)
class DataLoss(paddle.nn.Layer):

    def __init__(self, nn_fun):
        super().__init__()
        self.fun = nn_fun

    def forward(self, X, C_true, CHN):
        Y_pred = self.fun(X)
        beta = 1e-4 * paddle.nn.functional.sigmoid(self.fun.k2)
        p = 0.15 * paddle.nn.functional.sigmoid(self.fun.k3)
        if CHN == 1:   #选择非初始条件
            C_pred = Y_pred[:, 2:3]
            output = paddle.sum(paddle.square(C_pred.reshape([-1]) - C_true.reshape([-1])))
        if CHN == 0:    #选择初始条件
            S_pred = Y_pred[:, 0:1]
            F_pred = Y_pred[:, 1:2]
            C_pred = Y_pred[:, 2:3]
            output = paddle.sum(paddle.square(S_pred.reshape([-1]) - 2e5)) + paddle.sum(
                paddle.square(F_pred.reshape([-1]) - 5)) + paddle.sum(paddle.square(C_pred.reshape([-1]) - 1))   #修改一下
            output = output / 3
        return output


# Equation Loss (L2)
class EqLoss(paddle.nn.Layer):

    def __init__(self, nn_fun):
        super().__init__()
        self.fun = nn_fun

    def forward(self, X):
        t = X[:, 0].reshape([X.shape[0], 1])
        X = paddle.concat([t], axis=-1)
        Y_pred = self.fun(X)

        S = Y_pred[:, 0:1]
        F = Y_pred[:, 1:2]
        C = Y_pred[:, 2:3]

        S_t = paddle.grad(S, t, create_graph=True, retain_graph=True)[0]
        F_t = paddle.grad(F, t, create_graph=True, retain_graph=True)[0]
        C_t = paddle.grad(C, t, create_graph=True, retain_graph=True)[0]

        alpha = 1.5 * paddle.nn.functional.sigmoid(self.fun.k1)
        beta = 1e-4 * paddle.nn.functional.sigmoid(self.fun.k2)
        p = 0.15 * paddle.nn.functional.sigmoid(self.fun.k3)

        # eq1
        eq1 = S_t + beta * S * F
        # eq2
        eq2 = F_t + alpha * F - beta * p * S * F
        # eq3
        eq3 = C_t - beta * p * S * F

        # sum loss
        output = (paddle.sum(paddle.square(eq1)) + paddle.sum(paddle.square(eq2)) + paddle.sum(paddle.square(eq3))) / 3
        return output


## import loss model
DataLoss = loss.DataLoss(nn_fun)
EqLoss = loss.EqLoss(nn_fun)

## define optimizer
if opt_name == 'Adam':
    optimizer = paddle.optimizer.Adam(learning_rate=lr,weight_decay=WD,parameters=nn_fun.parameters())
if opt_name == 'SGD':
    optimizer = paddle.optimizer.SGD(learning_rate=lr, parameters=nn_fun.parameters())
if opt_name == 'Momentum':
    optimizer = paddle.optimizer.Momentum(learning_rate=lr, parameters=nn_fun.parameters())
if opt_name == 'LBFGS':
    optimizer = paddle.incubate.optimizer.LBFGS(lr=lr,weight_decay=WD,parameters=nn_fun.parameters())


## load saved parameters if applicable
if epoch_read > 1:
    load_net_params = paddle.load('net_params_' + str(epoch_read))
    load_opt_params = paddle.load('opt_params_' + str(epoch_read))
    nn_fun.set_state_dict(load_net_params)
    optimizer.set_state_dict(load_opt_params)


import paddle.nn.functional as pf
## training network model without LBFGS
if opt_name != 'LBFGS':

    X_train.stop_gradient = False
    for epoch_id in range(epoch_start,epoch_end+1):

        # define record
        lossIC_Rec = lossEQ_Rec = lossSUP_Rec = 0.0

        # DataLoss Calculate & Backward

        # Supervise Loss
        lossSUP = weightSUP * DataLoss(X_supervise,C_supervise,1) / X_supervise.shape[0]   #选择非初始条件
        # add data_loss backward gradient
        lossSUP.backward()
        # record IC loss just in number
        lossSUP_Rec = lossSUP_Rec + lossSUP

        # IC Loss
        lossIC = weightIC * DataLoss(X_ic,Y_ic[:,0],0)     #选择初始条件
        # add data_loss backward gradient
        lossIC.backward()
        # record IC loss just in number
        lossIC_Rec = lossIC_Rec + lossIC


        # EqLoss Calculate & Backward
        lossEQ = weightEQ * EqLoss(X_train) / X_train.shape[0]
        # add eq_loss backward gradient
        lossEQ.backward()
        # record Equation losses just in number
        lossEQ_Rec = lossEQ_Rec + lossEQ

        # get total loss just in number
        lossTotal = lossSUP_Rec + lossEQ_Rec + lossIC_Rec

        # Update Network Parameter
        optimizer.step()
        # Clear Backward Gradient
        optimizer.clear_grad()
        # Print and Save
        if epoch_id % 100 == 0:
            alpha = 1.5*paddle.nn.functional.sigmoid(nn_fun.k1)
            beta = 1e-4*paddle.nn.functional.sigmoid(nn_fun.k2)
            p = 0.15*paddle.nn.functional.sigmoid(nn_fun.k3)
            print('epoch:',epoch_id,'loss_total:',lossTotal.numpy(),'loss_sup:',lossSUP_Rec.numpy(),'loss_ic:',lossIC_Rec.numpy(),'loss_eq:',lossEQ_Rec.numpy(),'alpha:',alpha.numpy(),'beta:',beta.numpy(),'p:',p.numpy())
            loss_log = [epoch_id, optimizer.get_lr(), lossTotal.numpy()[0], lossSUP_Rec.numpy()[0],lossIC_Rec.numpy()[0],lossEQ_Rec.numpy()[0],alpha.numpy()[0],beta.numpy()[0],p.numpy()[0]]
            with open('loss_log.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerow(loss_log)
        if (epoch_id) % check_point == 0:
            paddle.save(nn_fun.state_dict(),'net_params_' + str(epoch_id))     #存的地址
            paddle.save(optimizer.state_dict(),' opt_params_' + str(epoch_id))




# load trained network
epoch_read = 100000
load_net_params = paddle.load( 'net_params_' + str(epoch_read))
nn_fun.set_state_dict(load_net_params)

# Print alpha, beta, p
alpha = 1.5*paddle.nn.functional.sigmoid(nn_fun.k1)
beta = 1e-4*paddle.nn.functional.sigmoid(nn_fun.k2)
p = 0.15*paddle.nn.functional.sigmoid(nn_fun.k3)
print('alpha:',alpha.numpy(),'beta:',beta.numpy(),'p:',p.numpy())

# Calculate S F C
Y_pred = nn_fun(X_supervise)
S_pred = Y_pred[:,0:1]
F_pred = Y_pred[:,1:2]
C_pred = Y_pred[:,2:3]

# Plot S F C results
plt.plot(X_supervise.numpy(), (C_supervise/1e4).numpy(), label='C_true_norm')
plt.plot(X_supervise.numpy(), (S_pred/2e5).numpy(), label='S_pred_norm')
plt.plot(X_supervise.numpy(), (F_pred/3e2).numpy(), label='F_pred_norm')
plt.plot(X_supervise.numpy(), (C_pred/1e4).numpy(), label='C_pred_norm')
plt.xlabel('Time (h)')
plt.ylabel('Value')
plt.legend()
plt.savefig('test00.png')
plt.show()