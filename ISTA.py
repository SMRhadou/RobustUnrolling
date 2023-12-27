import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging
import pickle
import argparse

from utils import ISTA_parser

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

def ISTA_W(Loader, W, alpha, Wsize, iters, batchSize, device):
    nBatches = len(Loader)
    M = W.shape[1]
    Z = np.zeros((M,Loader.dataset.data.shape[0]))
    Z = torch.tensor(Z, requires_grad=False)
    # Update Parameters
    for iter in tqdm(range(iters)):
        for ibatch, (xbatch, _) in enumerate(Loader):
            xbatch = xbatch.reshape((-1, xbatch.shape[0])).to(device)
            if ibatch == nBatches - 1:
                Zbatch = Z[:,ibatch*batchSize:].to(device)
            else:    
                Zbatch = Z[:,ibatch*batchSize:(ibatch+1)*batchSize].to(device)
            assert Zbatch.shape[1] == xbatch.shape[1]
            
            W = torch.tensor(W, device=torch.device(device))
            W.requires_grad = True
            L = 1.1 * torch.norm(W, p=2)**2
            theta = torch.tensor(alpha/L, requires_grad=False)

            # ISTA Update
            Zbatch = Zbatch - 1/L * (W.T @ (W @ Zbatch - xbatch))
            Zbatch = torch.sign(Zbatch) * torch.maximum(torch.abs(Zbatch) - theta, torch.zeros_like(Zbatch))    #Soft Thresholding
            # Cost Function
            J = 0.5 * torch.norm(W @ Zbatch - xbatch, dim=0)**2 + alpha * torch.norm(Zbatch, p =1, dim=0)
            J = J.mean()
            #Update W
            J.backward()
            W = (W.detach() - Wsize * W.grad)        
            Z[:,ibatch*batchSize:(ibatch+1)*batchSize] = Zbatch.detach().cpu()
        
        # I = 0.5 * torch.norm(W.detach().cpu() @ Z - Loader.dataset.transforms()(Loader.dataset.data).reshape((-1, len(Loader.dataset))), dim=0) + alpha * torch.norm(Z, p =1, dim=0)
        # I = I.mean()
        logging.debug('Iteration {}: Cost = {:.3f}, obj fn = {:.3f}, sparse = {}'.format(iter,
                    J.item(), 0.5 * torch.norm(W.detach() @ Zbatch - xbatch, dim=0).mean().item(),
                    torch.sum(torch.abs(Zbatch) > 1e-2, dim=0)[0]))
        
        if iter%100 == 0:
            with open('./ISTAdata/optimal_internal.pkl', 'wb') as ObjFile:
                pickle.dump((W, Z), ObjFile)

    return Z, W

def ISTA(X, W, alpha, iters, device):
    M = W.shape[1]
    Z = np.zeros((M,X.shape[0]))
    Z = torch.tensor(Z, requires_grad=False, device=torch.device(device))
    X = X.T.to(device)
    for iter in tqdm(range(iters)):
        W = torch.tensor(W, device=torch.device(device))
        L = 1.1 * torch.norm(W, p=2)**2
        theta = torch.tensor(alpha/L, requires_grad=False)

        # ISTA Update
        Z = Z - 1/L * (W.T @ (W @ Z - X))
        Z = torch.sign(Z) * torch.maximum(torch.abs(Z) - theta, torch.zeros_like(Z))    #Soft Thresholding
        # Cost Function
        J = 0.5 * (torch.norm(W @ Z - X, dim=0)**2 + alpha * torch.norm(Z, p =1, dim=0)).mean()

        logging.debug('Iteration {}: Cost = {:.3f}, obj. fn. = {:.3f}, sparse = {}'.format(iter,
                    J.item(), (0.5 * torch.norm(W @ Z - X, dim=0)**2).mean().item(),
                    torch.sum(torch.abs(Z) > 1e-2, dim=0)[0]))
        
        if iter%100 == 0:
            with open('./ISTAdata/optimal_internal.pkl', 'wb') as ObjFile:
                pickle.dump((W, Z), ObjFile)

    return Z, W

def main(alpha=0.5, Wsize=0.5, batchSize=512):
    FLAGS = argparse.ArgumentParser()
    _, args = ISTA_parser(FLAGS)

    # Logging
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logfile = f"./logs/logfile_{args.Trial}.log"
    logging.basicConfig(filename=logfile, level=logging.DEBUG, force=True)
    logging.debug("New Experiemnts ...")
    # Choose device
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    torch.cuda.empty_cache()

    # Load MNIST
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor()])
    dataset  = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainData = torch.stack([transform(x) for x in dataset.data]).reshape((len(dataset), -1))
    trainData = trainData[torch.randperm(trainData.shape[0])]
    
    # Dictionary W
    if args.pretrained_W:
        # Load W
        with open(args.WpretrainPath, 'rb') as ObjFile:
            W, _ = pickle.load(ObjFile)
        N = W.shape[0]
        M = W.shape[1]
    else:
        N = trainData.shape[1]
        M = int(N*1.2)
        W = np.random.rand(N, M)

    if args.train_W:
        Z, W = ISTA_W(trainData, W, alpha, Wsize, args.iters, batchSize, device)
    else:
        Z, W = ISTA(trainData, W, alpha, args.iters, device)

    if args.train_W:
        savePath = './ISTAdata/oWpretrain.pkl'
    else:
        savePath = f'./ISTAdata/optimal_{args.Trial}.pkl'

    dataset = torch.utils.data.TensorDataset(trainData, Z.T)
    with open(savePath, 'wb') as ObjFile:
        pickle.dump((W, Z, dataset), ObjFile)


if __name__ == '__main__':
    main()  
    print('OK!')
