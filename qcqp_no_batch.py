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

from pybindings import solveQP, solveQCQP, solveDerivativesQP, solveDerivativesQCQP

import time
import timeit


class QPFn2(Function):
    @staticmethod
    def forward(ctx, P, q, warm_start, eps, max_iter, mu_prox=1e-7):
        adaptative_rho = True
        l_2 = torch.zeros(q.size())
        Pnp, qnp, warm_startnp = P.detach().numpy(), q.detach().numpy(), warm_start.detach().numpy()
        l_2 = torch.from_numpy(solveQP(Pnp, qnp, warm_startnp, eps, mu_prox, max_iter, adaptative_rho))
        ctx.save_for_backward(P, q, l_2)
        return l_2

    @staticmethod
    def backward(ctx, grad_l):
        '''
        Compute derivatives of the solution of the QCQP with respect to entries
        '''
        P, q, l = ctx.saved_tensors
        grad_P, grad_q = None, None
        dl = torch.zeros(l.size())
        Pnp, qnp, lnp, grad_lnp = P.detach().numpy(), q.detach().numpy(), l.detach().numpy(), grad_l.detach().numpy()
        dl = torch.from_numpy(solveDerivativesQP(Pnp, qnp, lnp, grad_lnp))
        if ctx.needs_input_grad[0]:
            if P.size()[0] == 1:
                grad_P = - (dl * l).unsqueeze(-1)
            else:
                grad_P = - torch.mm(dl.unsqueeze(-1), l.unsqueeze(0))
        if ctx.needs_input_grad[1]:
            grad_q = - dl.unsqueeze(-1)

        return grad_P, grad_q, None, None, None, None


class QCQPFn2(Function):
    @staticmethod
    def forward(ctx, P, q, l_n, mu, warm_start, eps, max_iter, mu_prox=1e-7):
        adaptative_rho = True
        l_2 = torch.zeros(q.size())
        Pnp, qnp, warm_startnp = P.detach().numpy(), q.detach().numpy(), warm_start.detach().numpy()
        munp, l_nnp = mu.detach().numpy(), l_n.detach().numpy()

        l_2 = torch.from_numpy(
                solveQCQP(Pnp, qnp, l_nnp,
                          munp, warm_startnp, eps, mu_prox, max_iter,
                          adaptative_rho))
        ctx.save_for_backward(P, q, l_n, mu, l_2)
        return l_2

    @staticmethod
    def backward(ctx, grad_l):
        '''
        Compute derivatives of the solution of the QCQP with respect to entries
        '''
        # print("Incoming grad QCQP: ", grad_l)
        P, q, l_n, mu, l = ctx.saved_tensors
        # print("P: ", P)
        # print("q: ", q)
        # print("l_n: ", l_n)
        # print("l: ", l)
        num_contact = mu.size()[0]
        dl = torch.zeros(l.size())
        dgamma = torch.zeros(l_n.size())
        E1, E2 = torch.zeros((num_contact,num_contact)), torch.zeros((num_contact,num_contact))
        grad_P, grad_q, grad_l_n, grad_mu = None, None, None, None

        E1, E2, dlgamma = solveDerivativesQCQP(P.detach().numpy(), q.detach().numpy(),
                                                       l_n.detach().numpy(), mu.detach().numpy(),
                                                       l.detach().numpy(), grad_l.detach().numpy())
        dlgamma = torch.from_numpy(dlgamma)
        dgamma = dlgamma[:num_contact]
        dl = dlgamma[num_contact:]
        E1, E2 = torch.from_numpy(E1), torch.from_numpy(E2)
        # print("E1: ", E1)
        # print("E2: ", E2)
        # print("dlgamma: ", dlgamma)
        if ctx.needs_input_grad[0]:
            if P.size()[0] == 1:
                grad_P = - (dl * l).unsqueeze(-1)
            else:
                grad_P = - torch.mm(dl.unsqueeze(-1), l.unsqueeze(-1).transpose(0, 1))
        if ctx.needs_input_grad[1]:
            grad_q = - dl.unsqueeze(-1)
        if ctx.needs_input_grad[2]:
            grad_l_n = torch.mm(E2, dgamma.unsqueeze(-1))  # avoid transposing by directly returning E2^T and E1^T
        if ctx.needs_input_grad[3]:
            grad_mu = torch.mm(E1, dgamma.unsqueeze(-1))

        return grad_P, grad_q, grad_l_n, grad_mu, None, None, None, None
