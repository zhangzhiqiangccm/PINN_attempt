# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt
from scipy import integrate, optimize

S0 = 505151
F0 = 1
I0 = 1
def sfi_model(y, x, beta, alpha,p):
    S = -y[0]* y[1] * beta
    F = y[0]* y[1] * beta*p  - y[1]* alpha
    I = (1-p) * beta * y[0]* y[1] + alpha * y[1]
    return S, F, I
beta = 2.0372e-5
alpha = 1.0756
p = 0.1141

def fit_odeint(x, beta, alpha , p):
    return integrate.odeint(sfi_model, (S0, F0, I0), x, args=(beta, alpha,p))[:,0],integrate.odeint(sfi_model, (S0, F0, I0), x, args=(beta, alpha,p))[:,1],integrate.odeint(sfi_model, (S0, F0, I0), x, args=(beta, alpha,p))[:,2]


x_data= np.arange(0,60)

s,f , i = fit_odeint(x_data,beta,alpha,p)
print(s)
print(f)
print(i)