import numpy as np
import scipy
from time import time
import timeit
from tqdm import tqdm
from torch import optim
from qcqp import QPFn2, QCQPFn2
import torch
torch.set_default_dtype(torch.double)
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.autograd.functional import jacobian
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
from qpth.qp import QPFunction
import matplotlib.pyplot as plt
import osqp
plt.style.use('bmh')


class QCQP_cvxpy(nn.Module):
    def __init__(self,eps=1e-14, max_iter = 100):
        '''
        '''
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
    
    def forward(self,P,q,l_n,mu):
        N = q.size()[1]
        l_t = cp.Variable(N)
        A = cp.Parameter((N, N),nonneg= True)
        b = cp.Parameter(N)
        c = cp.Parameter(N//2, nonneg=True)
        #Gs = []
        #for i in range(N//2):
            
        constraints = [cp.SOC(c[i], l_t[2*i:2*(i+1)]) for i in range(N//2)]
        #constraints = []
        #objective = cp.Minimize(0.5 * cp.quad_form(l_t,A) + b.T@l_t )
        objective = cp.Minimize(0.5 * cp.sum_squares(A@l_t) + b.T@l_t )
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        cvxpylayer = CvxpyLayer(problem, parameters=[A, b, c], variables=[l_t])
        # solve the problem
        k=1e-11 #regularization of P to get Cholesky's decomposition 
        k=0.
        P_sqrt = scipy.linalg.sqrtm(P.detach().numpy().copy()[0,:,:])
        P_sqrt = torch.tensor(P_sqrt).unsqueeze(0)
        #L = torch.transpose(torch.cholesky(P+k*torch.eye(P.size()[1])),1,2)
        solution, = cvxpylayer(P_sqrt,q.squeeze(2),(mu*l_n).squeeze(2),solver_args={'eps': self.eps,'max_iters':self.max_iter})
        solution, = cvxpylayer(P,q.squeeze(2),(mu*l_n).squeeze(2),solver_args={'eps': self.eps,'max_iters':self.max_iter})
        #print(solution)
        return solution




cvxpy_time = {'forward': [], 'backward':[]}
qcqp_time = {'forward': [], 'backward':[]}
n_testqcqp= 0
for i in tqdm(range(n_testqcqp)):    
    #P = torch.rand((1,8,8),dtype = torch.double)
    P = torch.rand(8)*20 -10
    P = torch.diag(torch.exp(P)).unsqueeze(0)
    #P = torch.diag(P).unsqueeze(0)
    #P = torch.matmul(P, torch.transpose(P,1,2))
    P = torch.nn.parameter.Parameter(P, requires_grad= True)
    q = torch.rand((1,8,1),dtype = torch.double)*2-1
    q = torch.nn.parameter.Parameter(q, requires_grad= True)
    l_n = torch.rand((1,4,1),dtype = torch.double)
    l_n = torch.nn.parameter.Parameter(l_n, requires_grad= True)
    mu = torch.rand((1,4,1),dtype = torch.double)
    mu = torch.nn.parameter.Parameter(mu, requires_grad= True)
    lr = 0.1
    optimizer2 = optim.Adam([P,q,l_n,mu], lr=lr)
    loss = nn.MSELoss()
    relu = torch.nn.ReLU()
    threshold = nn.Threshold(threshold=1e-5, value =1e-5)
    target = torch.ones(q.size())
    qcqp = QCQPFn2().apply
    #warm_start = torch.zeros(q.size())
    warm_start = torch.rand(q.size())
    t0 = time()
    l1= qcqp(P,q,l_n,mu,warm_start,1e-10,1000000)
    t1= time()
    qcqp_time['forward']+= [timeit.timeit(lambda:qcqp(P,q,l_n,mu,warm_start,1e-10,1000000),number = 10)/10.]
    L1 = loss(l1, target)
    optimizer2.zero_grad()
    qcqp_time['backward']+= [timeit.timeit(lambda:L1.backward(retain_graph=True),number = 10)/10.]
    t2 = time()
    L1.backward()
    t3 = time()
    qcqp_time['forward']+= [t1-t0]
    qcqp_time['backward']+= [t3-t2]
    qcqp2 = QCQP_cvxpy(eps=1e-10,max_iter = 1000000)
    t4 = time()
    l1 = qcqp2(P,q,l_n,mu)
    t5= time()
    L1 = loss(l1.unsqueeze(2), target)
    optimizer2.zero_grad()
    t6 = time()
    L1.backward()
    t7 = time()
    cvxpy_time['forward']+= [t5-t4]
    cvxpy_time['backward']+= [t7-t6]


optnet_time = {'forward': [], 'backward':[]}
qp_time = {'forward': [], 'backward':[]}
osqp_time = {'forward': []}
n_testqp= 1
for i in tqdm(range(n_testqp)):    
    #P = torch.rand((1,8,8),dtype = torch.double)
    P = torch.rand(8)*20 -10
    #P = P.unsqueeze(0)
    #P = torch.rand(8)*10
    #P = torch.diag(P).unsqueeze(0)
    P = torch.diag(torch.exp(P)).unsqueeze(0)
    P = torch.matmul(P, torch.transpose(P,1,2))
    P = torch.matmul(P, torch.transpose(P,1,2))
    P = torch.nn.parameter.Parameter(P, requires_grad= True)
    q = torch.rand((1,8,1),dtype = torch.double)*2-1
    q = torch.nn.parameter.Parameter(q, requires_grad= True)
    #P = torch.tensor([[[0.4979, 0.3295],[0.3295, 0.2432]]], requires_grad= True)
    #q = torch.tensor([[[-0.3661],[-0.9514]]], requires_grad = True)
    lr = 0.1
    optimizer2 = optim.Adam([P,q], lr=lr)
    loss = nn.MSELoss()
    target = torch.ones(q.size())
    qp = QPFn2.apply
    warm_start = torch.zeros(q.size())
    t0 = time()
    #qp_time['forward']+= [timeit.timeit(lambda:qp(P,q,warm_start,1e-10,1000000),number = 10)/10.]
    l1= qp(P,q,warm_start,1e-10,1000000)
    l1[0,1].backward()
    #print(jacobian(lambda x,y: qp(x,y, warm_start,1e-10,1000000), (P,q))) #get jacobian of the solution wrt parameters of QP
    t1= time()
    L1 = loss(l1, target)
    optimizer2.zero_grad()
    #qp_time['backward']+= [timeit.timeit(lambda:L1.backward(retain_graph=True),number = 10)/10.]
    t2 = time()
    L1.backward()
    qp_time['forward']+= [t1-t0]
    qp_time['backward']+= [t3-t2]
    e = Variable(torch.Tensor())
    u = torch.zeros(q.size(), requires_grad = False, dtype = torch.double).squeeze(2)
    B = -torch.eye(q.size()[1], requires_grad = False, dtype = torch.double).unsqueeze(0)
    t4 = time()
    l1 = QPFunction(eps = 1e-10, verbose=-1, maxIter=1000000)(P, q.squeeze(2), B, u, e, e)
    optnet_time['forward']+= [timeit.timeit(lambda:QPFunction(eps = 1e-10, verbose=-1, maxIter=1000000)(P, q.squeeze(2), B, u, e, e),number = 10)/10.]
    t5= time()
    L1 = loss(l1.unsqueeze(2), target)
    optimizer2.zero_grad()
    t6 = time()
    optnet_time['backward']+= [timeit.timeit(lambda:L1.backward(retain_graph=True),number = 10)/10.]
    L1.backward()
    t7 = time()
    m = osqp.OSQP()
    m.setup(P=scipy.sparse.csc_matrix(P[0,:,:].detach().numpy()), q=q[0,:,0].detach().numpy(), A=scipy.sparse.csc_matrix(np.eye(q.size()[1])), l=np.zeros(q.size()[1]), u=np.inf*np.ones(q.size()[1]), verbose = False, eps_abs = 1e-10,eps_rel = 1e-20,max_iter = 1000000)
    osqp_time['forward']+= [timeit.timeit(lambda:m.solve(), number=1)/1.]
    optnet_time['forward']+= [t5-t4]
    optnet_time['backward']+= [t7-t6]

optnet_time['mean forward'] = sum(optnet_time['forward'])/n_testqp
optnet_time['mean backward'] = sum(optnet_time['backward'])/n_testqp
qp_time['mean forward'] = sum(qp_time['forward'])/n_testqp
qp_time['mean backward'] = sum(qp_time['backward'])/n_testqp
osqp_time['mean forward'] = sum(osqp_time['forward'])/n_testqp

optnet_time['error forward'] = np.std(optnet_time['forward'])
optnet_time['error backward'] = np.std(optnet_time['backward'])
print(optnet_time['error forward'],np.max(optnet_time['forward']), np.min(optnet_time['forward']))
qp_time['error forward'] = np.std(qp_time['forward'])
print(qp_time['error forward'], np.max(qp_time['forward']), np.min(qp_time['forward']))
qp_time['error backward'] = np.std(qp_time['backward'])
osqp_time['error forward'] = np.std(osqp_time['forward'])
print("osqp", osqp_time)

cvxpy_time['mean forward'] = sum(cvxpy_time['forward'])/n_testqcqp
cvxpy_time['mean backward'] = sum(cvxpy_time['backward'])/n_testqcqp
qcqp_time['mean forward'] = sum(qcqp_time['forward'])/n_testqcqp
qcqp_time['mean backward'] = sum(qcqp_time['backward'])/n_testqcqp

cvxpy_time['error forward'] = np.std(cvxpy_time['forward'])
cvxpy_time['error backward'] = np.std(cvxpy_time['backward'])
print(cvxpy_time['error forward'],np.max(cvxpy_time['forward']), np.min(cvxpy_time['forward']))
qcqp_time['error forward'] = np.std(qcqp_time['forward'])
print(qcqp_time['error forward'], np.max(qcqp_time['forward']), np.min(qcqp_time['forward']))
qcqp_time['error backward'] = np.std(qcqp_time['backward'])


barWidth = 0.35
y1 = [optnet_time['mean forward'],qp_time['mean forward']]
y2 = [optnet_time['mean backward'], qp_time['mean backward'] ]
er1 = [optnet_time['error forward'],qp_time['error forward']]
er2 = [optnet_time['error backward'],qp_time['error backward']]
r1 = range(len(y1))
r2 = [x + barWidth for x in r1]
plt.figure()
plt.bar(r1, y1, width = barWidth, color = ['cornflowerblue' for i in y1], linewidth = 2,log = True, label="forward", yerr = er1)
plt.bar(r2, y2, width = barWidth, color = ['coral' for i in y2], linewidth = 4,log = True,label="backward", yerr = er2)
plt.xticks([r + barWidth / 2 for r in range(len(y1))], ['OptNet', 'Ours'])
plt.ylabel('Runtime (s)')
plt.title('QP solvers')
plt.ylim(bottom = 1e-5, top= 1e-1)
plt.legend()
plt.show()

y1 = [cvxpy_time['mean forward'],qcqp_time['mean forward']]
y2 = [cvxpy_time['mean backward'], qcqp_time['mean backward'] ]
er1 = [cvxpy_time['error forward'],qcqp_time['error forward']]
er2 = [cvxpy_time['error backward'],qcqp_time['error backward']]
r1 = range(len(y1))
r2 = [x + barWidth for x in r1]
plt.figure()
plt.bar(r1, y1, width = barWidth, color = ['cornflowerblue' for i in y1], linewidth = 2,log = True, label="forward", yerr = er1)
plt.bar(r2, y2, width = barWidth, color = ['coral' for i in y1], linewidth = 4,log = True,label="backward", yerr = er2)
plt.xticks([r + barWidth / 2 for r in range(len(y1))], ['cvxpylayers', 'Ours'])
plt.ylabel('Runtime (s)')
plt.title('QCQP solvers')
plt.ylim(bottom = 1e-5, top= 1e-1)
plt.legend()
plt.show()