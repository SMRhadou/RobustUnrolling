import numpy as np
import logging
import os
import pickle

import torch
from torch.functional import F
from torch.autograd.functional import jacobian
import torch.nn as nn

from utils import distance_to_optimal

def unconstrained_learning(model, dataset, optimizer, **kwargs):
    device = kwargs["device"]
    batchSize = kwargs["batchSize"]
    nBatches = kwargs["nBatches"]
    
    # Mini-Batch Ttraining
    for ibatch in range(nBatches):
        # Forward
        xbatch, Zbatch_opt = dataset[ibatch*batchSize:(ibatch+1)*batchSize]
        xbatch = xbatch.T.to(device)
        Zbatch_opt = Zbatch_opt.T.to(device).float()
        Zbatch, _ = model(xbatch, **kwargs)
        loss = loss_function(z=Zbatch, z_opt=Zbatch_opt)

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model

def objective_function(z, x, W, alpha=0.5):
    return torch.sum(0.5 * torch.norm(x - W.float() @ z, p=2, dim=0)**2) # + alpha * torch.norm(z, p=2, dim=0))
def objective_function_l1reg(z, x, W, alpha=0.5):
    return torch.sum(0.5 * torch.norm(x - W.float() @ z, p=2, dim=0)**2 + alpha * torch.norm(z, p=1, dim=0))

def GradPenalty(objective_function, x, outsTrain, **kwargs):
    """
    evaluate Supermartingale constraints
    """
    eps = kwargs['eps']
    torch.cuda.empty_cache()
    L = len(outsTrain.keys()) - 1
    gradVector = torch.zeros(L+1, kwargs['SysID'].shape[1], x.shape[1]).to(kwargs['device'])
    for l in outsTrain.keys():
        z = outsTrain[l]
        gradVector[l] = jacobian(objective_function, (z, x, kwargs['SysID']), create_graph=True)[0]
    cons = (torch.norm(gradVector[1:], p=2, dim=1)/torch.norm(gradVector[:-1], p=2, dim=1)) - (1-eps) 
    return cons

def distancePenalty(z_opt, outsTrain, **kwargs):
    """
    Assess whther the distance to the optimal decreases over the layers
    """
    eps = kwargs['eps']
    L = len(outsTrain.keys()) - 1
    distance = torch.zeros(L+1, z_opt.shape[0], z_opt.shape[1]).to(kwargs['device'])
    for l in outsTrain.keys():
        distance[l] = torch.norm(outsTrain[l] - z_opt, p=2, dim=0)
    cons = (torch.norm(distance[1:], p=2, dim=1)/torch.norm(distance[:-1], p=2, dim=1)) - (1-eps)
    return cons

def Lagrang_loss(Zbatch, Zbatch_opt, penalty, W=None, x=None):
    loss = loss_function(z=Zbatch, z_opt=Zbatch_opt) + penalty
    return loss

def constrained_learning(model, dataset, optimizer, objective_function, nu, grad=False, **kwargs):
    device = kwargs["device"]
    lr_dual = kwargs["lr_dual"]
    nBatches = kwargs["nBatches"]
    batchSize = kwargs["batchSize"]

    # Mini-Batch Ttraining
    for ibatch in range(nBatches):
        # Forward Step 
        xbatch, Zbatch_opt = dataset[ibatch*batchSize:(ibatch+1)*batchSize]
        xbatch = xbatch.T.to(device)
        Zbatch_opt = Zbatch_opt.T.to(device).float()
        Zbatch, outsBatch = model(xbatch, **kwargs)
        if grad:
            cons = GradPenalty(objective_function, xbatch, outsBatch, **kwargs)
        else:
            cons = distancePenalty(Zbatch_opt, outsBatch, **kwargs)

        # Lagrangian Function
        penalty = torch.sum(nu * torch.mean(cons, dim=1))
        Lagrang = Lagrang_loss(Zbatch, Zbatch_opt, penalty)

        # Backward Step
        # Primal update
        Lagrang.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.empty_cache()
    with torch.no_grad():
        xTrain, zTrain_opt = dataset[:(ibatch+1)*batchSize]
        xTrain = xTrain.T.to(device)
        zTrain_opt = zTrain_opt.T.to(device).float()
        _, outsTrain = model(xTrain, **kwargs)
        if grad:
            cons = GradPenalty(objective_function, xTrain, outsTrain, **kwargs)
        else:
            cons = distancePenalty(zTrain_opt, outsTrain, **kwargs)
        
    # Dual update
    nu_temp = nu + lr_dual * torch.mean(cons, dim=1)
    nu = nn.ReLU()(nu_temp)
    nu = nu.detach()

    del xTrain, outsTrain, cons
    torch.cuda.empty_cache()

    return model, nu

def loss_function(x=None, W=None, z=None, z_opt=None, alpha=0.5):
    return 0.5*(torch.norm(z-z_opt, dim=0)**2).mean()
    #return F.mse_loss(z, z_opt) #+ torch.mean(alpha * torch.norm(z, p=1, dim=0))


def evaluate(model, dataset, objective_function, eps, saveResults=False, resultPath=None, grad=False, **kwargs):
    device = kwargs["device"]
    if 'epoch' in kwargs:
        epoch = kwargs['epoch']
    else:
        epoch = 'Test'

    with torch.no_grad():
        xValid, zValid_opt = dataset[:]
        xValid = xValid.T.to(device)
        zValid_opt = zValid_opt.T.to(device).float()
        zValid, outsValid = model(xValid, eps=eps, **kwargs)
        validloss = loss_function(z=zValid, z_opt=zValid_opt)
        if grad:
            cons = GradPenalty(objective_function, xValid, outsValid, eps=eps,**kwargs)
        else:
            cons = distancePenalty(zValid_opt, outsValid, eps=eps, **kwargs)
        logging.debug('distance to optimal: {}'.format(list(distance_to_optimal(outsValid, zValid_opt)[0])))
        #logging.debug('distance to my optimal: {}'.format(list(disance_to_optimal(outsValid, zValid))))
        logging.debug('Gradients: {}'.format([torch.norm(jacobian(objective_function, (outsValid[i], xValid, kwargs['SysID']), 
                        create_graph=True)[0], p=2, dim=0).mean().item() for i in range(model.nLayers+1)]))
        logging.debug('constraints {}'.format(list(torch.mean(cons, dim=1).detach().cpu().numpy())))
        
    logging.debug("Epoch {}, Loss {:.4f}, obj. fn. {}, sparse: {} \n".format(epoch, validloss.item(),
                        objective_function(zValid, xValid, kwargs['SysID'])/xValid.shape[1], torch.norm(zValid[:,0], p=0).float().mean().item()))
    logging.debug("optimal obj. fn. {:.3f} with gradient {:.3f}".format(objective_function(zValid_opt, xValid, kwargs['SysID'])/xValid.shape[1],
                                    torch.norm(jacobian(objective_function, (zValid_opt, xValid, kwargs['SysID']), create_graph=True)[0], p=2, dim=0).mean().item()))
    
    # Save results
    if saveResults and resultPath is not None:
        with open(resultPath, 'wb') as ObjFile:
            pickle.dump((kwargs['SysID'], xValid, outsValid, zValid_opt, cons, eps), ObjFile)
        
    return validloss, outsValid


def noisy_evaluate(model, dataset, objective_function, eps, saveResults=False, resultPath=None, grad=False, **kwargs):
    device = kwargs["device"]
    if 'epoch' in kwargs:
        epoch = kwargs['epoch']
    else:
        epoch = 'Test'
    beta = kwargs["beta"]
    model = model.to(device)
    with torch.no_grad():
        xValid, zValid_opt = dataset[:]
        xValid = xValid.T.to(device)
        zValid_opt = zValid_opt.T.to(device).float()
        zValid, outsValid, zbeforeNoise = model.forward_noise(xValid, eps=eps, **kwargs)
        validloss = loss_function(z=zValid, z_opt=zValid_opt)

        if grad:
            cons = GradPenalty(objective_function, xValid, outsValid, eps=eps,**kwargs)
            consBeforeNoise = GradPenalty(objective_function, xValid, zbeforeNoise, eps=eps,**kwargs) - (1-eps)
        else:
            cons = distancePenalty(zValid_opt, outsValid, eps=eps,**kwargs)
            consBeforeNoise = distancePenalty(zValid_opt, zbeforeNoise, eps=eps,**kwargs) - (1-eps)
        descentPercentage = torch.mean((consBeforeNoise < 0).float(), dim=1).detach().cpu().numpy()
        logging.debug('distance to optimal: {}'.format(list(distance_to_optimal(outsValid, zValid_opt)[0])))
        #logging.debug('distance to my optimal: {}'.format(list(disance_to_optimal(outsValid, zValid))))
        logging.debug('Gradients: {}'.format([torch.norm(jacobian(objective_function, (outsValid[i], xValid, kwargs['SysID']), 
                        create_graph=True)[0], p=2, dim=0).mean().item() for i in range(model.nLayers+1)]))
        logging.debug('constraints {}'.format(list(torch.mean(cons, dim=1).detach().cpu().numpy())))
        
    logging.debug("Epoch {}, Loss {:.4f}, obj. fn. {}, sparse: {} \n".format(epoch, validloss.item(),
                        objective_function(zValid, xValid, kwargs['SysID'])/xValid.shape[1], torch.norm(zValid[:,0], p=0).float().mean().item()))
    logging.debug("optimal obj. fn. {:.3f} with gradient {:.3f}".format(objective_function(zValid_opt, xValid, kwargs['SysID'])/xValid.shape[1],
                                    torch.norm(jacobian(objective_function, (zValid_opt, xValid, kwargs['SysID']), create_graph=True)[0], p=2, dim=0).mean().item()))
    
    # Save results
    if saveResults and resultPath is not None:
        with open(resultPath+f"_beta{beta}.pkl", 'wb') as ObjFile:
            pickle.dump((kwargs['SysID'], xValid, outsValid, zValid_opt, cons, eps), ObjFile)

    mean, _, _, var = distance_to_optimal(outsValid, zValid_opt)
    l1_norm = [torch.norm(outsValid[l], p=1, dim=0).float().mean().item() for l in range(model.nLayers+1)]
    l1_var = [torch.sqrt(torch.var(torch.norm(outsValid[l], p=1, dim=0))).item() for l in range(model.nLayers+1)]
        
    return mean, var, l1_norm, l1_var, descentPercentage[:-1:2]