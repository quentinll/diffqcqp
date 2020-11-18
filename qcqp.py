#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:37:27 2020

@author: quentin
"""


import torch 
from torch.autograd import Function, Variable
import torch.nn as nn
torch.set_default_dtype(torch.double)

import torch.autograd.profiler as profiler

from build.pybindings import solveQP, solveQCQP, solveDerivativesQP, solveDerivativesQCQP

import time
import timeit

class QPFn2(Function):
    
    @staticmethod
    def forward(ctx,P,q,warm_start,eps,max_iter,mu_prox =1e-7):
       # with profiler.profile(record_shapes=True) as prof:
       #     with profiler.record_function("solving_qp"):
        #durations = {"power iter":[], "iters":[], "proxs":[], "l update": [], "batch prox":[]}
        batch_size = q.size()[0]
        l_2 =torch.zeros(q.size())
        #print(P.size(), q.size(), warm_start.size() )
        adaptative_rho =True
        Pi,qi,warm_starti = torch.zeros(1,P.size()[1],P.size()[2]),torch.zeros(1,P.size()[2],1),torch.zeros(1,P.size()[2],1) 
        for i in range(batch_size):
            #print('before',P[i,:,:],q[i,:,:], warm_start[i,:,:])
            #y = torch.tensor(pyb.solveQP(P[i,:,:].detach().numpy(),q[i,:,:].detach().numpy(), warm_start[i,:,:].detach().numpy(), eps, mu_prox, 30))
            #print(y)
            Pi,qi,warm_starti = P[i,:,:].detach().numpy(),q[i,:,:].detach().numpy(), warm_start[i,:,:].detach().numpy()
            #t3 = time.time()
            
            #l_2[i,:,0] = torch.from_numpy(solveQP(P[i,:,:].detach().numpy(),q[i,:,:].detach().numpy(), warm_start[i,:,:].detach().numpy(), eps,mu_prox,max_iter, adaptative_rho))
            l_2[i,:,0] = torch.from_numpy(solveQP(Pi,qi, warm_starti, eps,mu_prox,max_iter, adaptative_rho))
            #t0 = time.time()
            #print("forward comp time qp",timeit.timeit(lambda: l_2[i,:,0] + torch.from_numpy(solveQP(Pi,qi, warm_starti, eps,mu_prox,max_iter, adaptative_rho)), number=10)/10.)
        #durations['iters'] += [t3-t0]
        #print("forward comp time qp", sum(durations["iters"]))
        ctx.save_for_backward(P,q,l_2)
        #print("save for backward ",timeit.timeit(lambda:ctx.save_for_backward(P,q,l_2), number=1))
        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        return l_2
    
    @staticmethod
    def backward(ctx, grad_l):
        '''
        Compute derivatives of the solution of the QCQP with respect to 
        '''
        P,q,l = ctx.saved_tensors
        #durations = {'backward QP':[], 'comp dual':[], 'inverse system':[], 'iterative refinement':[], 'build system':[]}
        batch_size = q.size()[0]
        grad_P, grad_q = None, None
        dl = torch.zeros(l.size())
        #t0 = time.time()
        Pi,qi,li,grad_li = torch.zeros(1,P.size()[1],P.size()[2]),torch.zeros(1,P.size()[2],1),torch.zeros(1,P.size()[2],1),torch.zeros(1,P.size()[2],1)  
        for i in range(batch_size):
            Pi,qi,li, grad_li = P[i,:,:].detach().numpy(),q[i,:,:].detach().numpy(), li[i,:,:].detach().numpy(),grad_li[i,:,:].detach().numpy()
            dl[i,:,0] = torch.from_numpy(solveDerivativesQP(Pi,qi,li,grad_li))
        if ctx.needs_input_grad[0]:
            grad_P = -torch.bmm(dl, torch.transpose(l,1,2))
        if ctx.needs_input_grad[1]:
            grad_q = - dl
        #t1 = time.time()
        #durations['backward QP']+=[t1-t0]
        #print('backward comp time 1 qp', timeit.timeit(lambda: dl[i,:,0] + torch.from_numpy(solveDerivativesQP(Pi,qi,li,grad_li)),number=10)/10.)
        return grad_P, grad_q, None, None, None, None

class QCQPFn2(Function):
    
    @staticmethod
    def forward(ctx,P,q,l_n,mu,warm_start,eps,max_iter,mu_prox =1e-7):
        durations = {"power iter":[], "iters":[], "l update":[],"u update":[],"res update":[], "batch prox":[]}
        batch_size = q.size()[0]
        l_2 =torch.zeros(q.size())
        #print(P.size(), q.size(), warm_start.size() )
        adaptative_rho =True
        for i in range(batch_size):
            #print('before',P[i,:,:],q[i,:,:], warm_start[i,:,:])
            #y = torch.tensor(pyb.solveQP(P[i,:,:].detach().numpy(),q[i,:,:].detach().numpy(), warm_start[i,:,:].detach().numpy(), eps, mu_prox, 30))
            #print(y)
            t0 = time.time()
            l_2[i,:,0] = torch.from_numpy(solveQCQP(P[i,:,:].detach().numpy(),q[i,:,:].detach().numpy(),l_n[i,:,:].detach().numpy(), mu[i,:,:].detach().numpy(), warm_start[i,:,:].detach().numpy(), eps, mu_prox, max_iter,adaptative_rho))
        #print("qcqp",i)
        #t3 = time.time()
        #durations['iters'] += [t3-t0]
        #print("qcqp power", sum(durations["power iter"]))
        #print("forward comp time 1 qcqp", sum(durations["iters"]))
        #print("qcqp batch prox", sum(durations["batch prox"]))
        #print("l update", sum(durations["l update"]))
        #print("u update", sum(durations["u update"]))
        #print("res update", sum(durations["res update"]))
        ctx.save_for_backward(P,q,l_n,mu,l_2)
        return l_2
    
    @staticmethod    
    def backward(ctx, grad_l):
        '''
        Compute derivatives of the solution of the QCQP with respect to 
        '''
        #durations = {'backward QCQP':[], 'comp dual':[], 'inverse system':[], 'iterative refinement':[], 'build system':[]}
        #t0 = time.time()
        P,q,l_n,mu,l = ctx.saved_tensors
        num_contact = mu.size()[1]
        batch_size = q.size()[0]
        grad_P, grad_q, grad_l_n, grad_mu = None, None, None, None
        dl = torch.zeros(l.size())
        dgamma = torch.zeros(l_n.size())
        E1,E2 = torch.zeros((batch_size,num_contact,num_contact)), torch.zeros((batch_size,num_contact,num_contact))
        for i in range(batch_size):
            E1_i,E2_i,dlgamma = solveDerivativesQCQP(P[i,:,:].detach().numpy(),q[i,:,:].detach().numpy(),l_n[i,:,:].detach().numpy(),mu[i,:,:].detach().numpy(),l[i,:,:].detach().numpy(),grad_l[i,:,:].detach().numpy())
            dlgamma = torch.from_numpy(dlgamma)
            dl[i,:,0] = dlgamma[num_contact:]
            dgamma[i,:,0] = dlgamma[:num_contact]
            E1[i,:,:],E2[i,:,:] = torch.tensor(E1_i),torch.tensor(E2_i)
        if ctx.needs_input_grad[0]:
            grad_P = -torch.bmm(dl, torch.transpose(l,1,2))
        if ctx.needs_input_grad[1]:
            grad_q = - dl
        if ctx.needs_input_grad[2]:
            grad_l_n = torch.bmm(E2,dgamma) #avoid transposing by directly returning E2^T and E1^T
        if ctx.needs_input_grad[3]:
            grad_mu = torch.bmm(E1,dgamma)
            
        #t1 = time.time()
        #durations['backward QCQP']+=[t1-t0]
        #print('backward comp time 1 qcqp', sum(durations['backward QCQP']))
        return grad_P, grad_q, grad_l_n, grad_mu, None, None, None, None
