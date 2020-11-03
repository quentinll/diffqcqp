import build.pybindings as pyb 
import torch


Q = torch.rand((12,12))
Q = torch.matmul(Q.transpose(0,1), Q)
q = torch.rand((12,1))
warm_start = torch.zeros((12,1))
print(pyb.solveQP(Q, q, warm_start))