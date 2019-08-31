import math
import torch
from torch.optim.optimizer import Optimizer
import torch.optim as optim


class DistrSGD(Optimizer):
    def __init__(self, params1, params2, params3, params4, lr=0.05, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        self.sgd1 = optim.SGD(params1, lr, momentum, dampening, weight_decay, nesterov)
        self.sgd2 = optim.SGD(params2, lr, momentum, dampening, weight_decay, nesterov)
        self.sgd3 = optim.SGD(params3, lr, momentum, dampening, weight_decay, nesterov)
        self.sgd4 = optim.SGD(params4, lr, momentum, dampening, weight_decay, nesterov)
        self.param1 = params1
        self.param2 = params2
        self.param3 = params3
        self.param4 = params4
        self.param_groups = [self.sgd1.param_groups[0],
                             self.sgd2.param_groups[0],
                             self.sgd3.param_groups[0],
                             self.sgd4.param_groups[0]]

    def __setstate__(self, state):
        self.sgd1.__setattr__(state)
        self.sgd2.__setattr__(state)
        self.sgd3.__setattr__(state)
        self.sgd4.__setattr__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        '''
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)
        '''
        group1 = self.param_groups[0]
        group2 = self.param_groups[1]
        group3 = self.param_groups[2]
        group4 = self.param_groups[3]
        weight_decay = group1['weight_decay']
        momentum = group1['momentum']
        dampening = group1['dampening']
        nesterov = group1['nesterov']
        for i in range(len(group1['params'])):
            p1 = group1['params'][i]
            p2 = group2['params'][i]
            p3 = group3['params'][i]
            p4 = group4['params'][i]

            if p1.grad is None:
                continue
            d_p1 = p1.grad.data
            d_p2 = p2.grad.data
            d_p3 = p3.grad.data
            d_p4 = p4.grad.data
            if weight_decay != 0:
                d_p1.add_(weight_decay, p1.data)
                d_p2.add_(weight_decay, p2.data)
                d_p3.add_(weight_decay, p3.data)
                d_p4.add_(weight_decay, p4.data)
            if momentum != 0:
                param_state1 = self.sgd1.state[p1]
                param_state2 = self.sgd2.state[p2]
                param_state3 = self.sgd3.state[p3]
                param_state4 = self.sgd4.state[p4]
                if 'momentum_buffer' not in param_state1:  # 1
                    buf = param_state1['momentum_buffer'] = torch.clone(d_p1).detach()
                else:
                    buf = param_state1['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p1)
                if nesterov:
                    d_p1 = d_p1.add(momentum, buf)
                else:
                    d_p1 = buf

                if 'momentum_buffer' not in param_state2:  # 2
                    buf = param_state2['momentum_buffer'] = torch.clone(d_p2).detach()
                else:
                    buf = param_state2['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p2)
                if nesterov:
                    d_p2 = d_p2.add(momentum, buf)
                else:
                    d_p2 = buf

                if 'momentum_buffer' not in param_state3:  # 3
                    buf = param_state3['momentum_buffer'] = torch.clone(d_p3).detach()
                else:
                    buf = param_state3['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p3)
                if nesterov:
                    d_p3 = d_p3.add(momentum, buf)
                else:
                    d_p3 = buf

                if 'momentum_buffer' not in param_state4:  # 4
                    buf = param_state4['momentum_buffer'] = torch.clone(d_p4).detach()
                else:
                    buf = param_state4['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p4)
                if nesterov:
                    d_p4 = d_p4.add(momentum, buf)
                else:
                    d_p4 = buf

            d_p = (d_p1 + d_p2 + d_p3 + d_p4) / 4
            p1.data.add_(-group1['lr'], d_p)

        return loss
