import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging
import argparse

from ISTA import generate_dataset
from utils import load_data, parser, printing
from training import constrained_learning, unconstrained_learning, evaluate, loss_function, benchmark_learning

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from functools import partial
from torch.autograd.functional import jacobian

def objective_function(y, x, D, alpha=0.5):
    return torch.sum(0.5 * torch.norm(x - D.float() @ y, p=2, dim=0)**2) # + alpha * torch.norm(z, p=2, dim=0))
def objective_function_l1reg(y, x, D, alpha=0.5):
    return torch.sum(0.5 * torch.norm(x - D.float() @ y, p=2, dim=0)**2 + alpha * torch.norm(y, p=1, dim=0))

# Unrolled LISTA
class LISTA(nn.Module):
    def __init__(self, nLayers, N, M, sigma=1e-2):
        super(LISTA, self).__init__()
        self.M = M
        self.N = N
        self.nLayers = nLayers
        self.sigma = sigma
        self.layers = nn.ModuleList()
        for l in range(self.nLayers):
            layer = nn.ParameterDict({
                            "W1": nn.Parameter(torch.randn(M, M)*sigma),
                            "W2": nn.Parameter(torch.randn(M, N)*sigma),
                            "theta": nn.Parameter(torch.randn(M,1)*sigma)})
            self.layers.append(layer)

    def forward(self, X:torch.tensor, **kwargs):
        noisyOuts = kwargs.get('noisyOuts', False)
        W = kwargs['SysID']
        # Initialize Z
        X = 100*X
        Y = torch.zeros((self.M, X.shape[1]), device=X.device)
        Y = torch.randn_like(Y, device=X.device)
        outs = {} 
        outs[0] = Y
        for l in range(self.nLayers):
            W1 = self.layers[l]['W1']
            W2 = self.layers[l]['W2']
            theta = self.layers[l]['theta']
            #Y = Y - (W.T @ (W @ Y - X))
            Y = W1@Y +  W2@X.float()
            Y = torch.sign(Y) * torch.maximum(torch.abs(Y) - theta, torch.zeros_like(Y))
            if noisyOuts and l < self.nLayers-1:
                grad = torch.norm(jacobian(objective_function, (Y, X, W), create_graph=True)[0], p=2, dim=0).detach()
                Y = Y + torch.randn_like(Y) * torch.log(grad) 
            outs[l+1] = Y/100
        return Y/100, outs

    def forward_noise(self, X:torch.tensor, **kwargs):
        beta = kwargs.get('beta', 100)
        W = kwargs['SysID']
        # Initialize Y
        X = X*100
        Y = torch.zeros((self.M, X.shape[1]), device=X.device)
        Y = torch.randn_like(Y, device=X.device) * beta
        true_outs = {}
        outs = {} 
        true_outs[0] = Y
        outs[0] = Y
        for l in range(self.nLayers):
            W1 = self.layers[l]['W1']
            W2 = self.layers[l]['W2']
            theta = self.layers[l]['theta']
            #Y = Y - (W.T @ (W @ Y - X))
            Y = W1.float()@Y +  W2.float()@X
            Y = torch.sign(Y) * torch.maximum(torch.abs(Y) - theta, torch.zeros_like(Y))
            true_outs[2*l+1] = Y/100
            if  l < self.nLayers-1:
                grad = torch.norm(jacobian(objective_function, (Y, X, W), create_graph=True)[0], p=2, dim=0)
                Y = Y + torch.randn_like(Y) * torch.log(grad) * beta 
            outs[l+1] = Y/100
            true_outs[2*l+2] = Y/100
        return Y/100, outs, true_outs


def train(lista, dataset, optimizer, objective_function, W, args, device, split, **kwargs):
    nTrain = int(len(dataset) * split)
    nBatches = nTrain // args.batchSize
    modelPath = kwargs["modelPath"]
    lr_dual = args.lr_dual
    constrained = args.constrained
    noisyOuts = args.noisyOuts
    NU = np.zeros((args.nEpochs+1, lista.nLayers))

    # train and valid dataset
    trainset = torch.utils.data.TensorDataset(dataset[:nTrain][0], dataset[:nTrain][1])
    validset = torch.utils.data.TensorDataset(dataset[nTrain:][0], dataset[nTrain:][1])

    #nu = torch.distributions.Uniform(0.1, 0.3).sample((lista.nLayers,)).double().to(device)
    nu = torch.zeros((lista.nLayers,), device=torch.device(device)).double()
    best = np.inf
    for epoch in tqdm(range(args.nEpochs)):
        # Training
        if constrained:
            lista, nu = constrained_learning(lista, trainset, optimizer, objective_function, nu, SysID=W, batchSize=args.batchSize, 
                                            nBatches=nBatches, noisyOuts=noisyOuts,
                                            lr_dual = lr_dual, eps=args.eps, device=device)
            NU[epoch+1] = nu.detach().cpu().numpy()
            logging.debug('duals {}'.format(list(nu.detach().cpu().numpy())))
        else:
            lista = unconstrained_learning(lista, trainset, optimizer, SysID=W, batchSize=args.batchSize, 
                                            nBatches=nBatches, noisyOuts=noisyOuts, eps=args.eps, device=device)

        # Validation
        validloss, _ = evaluate(lista, validset, objective_function, eps=args.eps, SysID=W.to(device), device=device, epoch=epoch)

        # Save model
        if validloss < best:
            best = validloss
            torch.save({"epoch": epoch,
                        "model_state_dict": lista.state_dict(),
                        "valid_loss": validloss
                        }, modelPath+"LISTA_best.pth")
        logging.debug("best: {}".format(best))
            
        del validloss
        torch.cuda.empty_cache()
    return lista, NU

def lpl_train(lista, dataset, objective_function, W, args, device, split, **kwargs):
    nTrain = int(len(dataset) * split)
    nBatches = nTrain // args.batchSize
    modelPath = kwargs["modelPath"]
    constrained = args.constrained
    noisyOuts = args.noisyOuts

    # train and valid dataset
    trainset = torch.utils.data.TensorDataset(dataset[:nTrain][0], dataset[:nTrain][1])
    validset = torch.utils.data.TensorDataset(dataset[nTrain:][0], dataset[nTrain:][1])

    # Optimizer
    optimizer = [optim.Adam(lista.layers[i].parameters(), lr=args.lr) for i in range(lista.nLayers)]

    best = np.inf
    for l in tqdm(range(lista.nLayers)):
        for epoch in tqdm(range(args.nEpochs), leave=False):
            # Training
            lista = benchmark_learning(lista, trainset, optimizer[l], SysID=W, batchSize=args.batchSize, l=l+1,
                                            nBatches=nBatches, noisyOuts=noisyOuts, eps=args.eps, device=device)

            # Validation
            validloss, _ = evaluate(lista, validset, objective_function, eps=args.eps, SysID=W.to(device), device=device, epoch=(l+1)*epoch)

            # Save model
            if validloss < best:
                best = validloss
                torch.save({"epoch": (l+1)*epoch,
                            "model_state_dict": lista.state_dict(),
                            "valid_loss": validloss
                            }, modelPath+"LISTA_best.pth")
            logging.debug("best: {}".format(best))
                
            del validloss
            torch.cuda.empty_cache()
    return lista

def inc_train(lista, dataset, objective_function, W, args, device, split, **kwargs):
    nTrain = int(len(dataset) * split)
    nBatches = nTrain // args.batchSize
    modelPath = kwargs["modelPath"]
    constrained = args.constrained
    noisyOuts = args.noisyOuts

    # train and valid dataset
    # trainset = torch.utils.data.TensorDataset(dataset[:nTrain//10][0], dataset[:nTrain][1])
    validset = torch.utils.data.TensorDataset(dataset[nTrain:][0], dataset[nTrain:][1])

    # Optimizer
    optimizer = optim.Adam(lista.layers[0].parameters(), lr=args.lr)

    best = np.inf
    for l in tqdm(range(lista.nLayers), leave=False):
        if l>0:
            optimizer.add_param_group({'params': lista.layers[l].parameters(), 'lr': args.lr})
        trainset = torch.utils.data.TensorDataset(dataset[l*nTrain//10:(l+1)*nTrain//10][0], dataset[l*nTrain//10:(l+1)*nTrain//10][1])

        for epoch in tqdm(range(args.nEpochs), leave=False):
            # Training
            lista = benchmark_learning(lista, trainset, optimizer, SysID=W, batchSize=args.batchSize, l=l+1,
                                            nBatches=nBatches, noisyOuts=noisyOuts, eps=args.eps, device=device)

            # Validation
            validloss, _ = evaluate(lista, validset, objective_function, eps=args.eps, SysID=W.to(device), device=device, epoch=(l+1)*epoch)

            # Save model
            if validloss < best:
                best = validloss
                torch.save({"epoch": (l+1)*epoch,
                            "model_state_dict": lista.state_dict(),
                            "valid_loss": validloss
                            }, modelPath+"LISTA_best.pth")
            logging.debug("best: {}".format(best))
                
            del validloss
            torch.cuda.empty_cache()
    return lista

def main():
    FLAGS = argparse.ArgumentParser()
    _, args = parser(FLAGS)

    # Logging
    if not os.path.exists("logs/LISTA"):
        os.makedirs("logs/LISTA")
    logfile = f"./logs/LISTA/logfile_LISTA__L{args.nLayers}_lr_{args.lr}_lrdual_{args.lr_dual}_eps_{args.eps}_constrained_{args.constrained}.log"
    logging.basicConfig(filename=logfile, level=logging.DEBUG, force=True)
    if not os.path.exists(f"models/{args.Trial}_L{args.nLayers}_lr_{args.lr}_lrdual_{args.lr_dual}_eps_{args.eps}_constrained_{args.constrained}"):
        os.makedirs(f"models/{args.Trial}_L{args.nLayers}_lr_{args.lr}_lrdual_{args.lr_dual}_eps_{args.eps}_constrained_{args.constrained}")
    modelPath = f"./models/{args.Trial}_L{args.nLayers}_lr_{args.lr}_lrdual_{args.lr_dual}_eps_{args.eps}_constrained_{args.constrained}/"

    logging.debug("Trial ...")
    printing(vars(args))
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # Generate Data (CIFAR10)
    if args.generate:
        generate_dataset(args)
    # Load Data
    dataset, testset, D, _ = load_data(data_dir=os.getcwd())
    # trainLoader = DataLoader(trainset, batch_size=batchSize, shuffle=False, num_workers=4)
    # testLoader = DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=4)
    # # Visualize
    # img = next(iter(trainLoader))[0][0]
    # plt.imshow(transforms.ToPILImage()(img))

    # Initialize LISTA
    N = D.shape[0]
    M = D.shape[1]
    lista = LISTA(nLayers=args.nLayers, N=N, M=M).to(device)

    # Initialize optimizer 
    optimizer = optim.Adam(lista.parameters(), lr=args.lr)

    # Training
    torch.cuda.empty_cache()
    
    if args.benchmark1:
        # Benchmarks (layer per layer training)
        NU = None
        lista = lpl_train(lista, dataset, objective_function, D, args, device, split=0.8, modelPath=modelPath)
    elif args.benchmark2:
        # Benchmarks (incremental training)
        NU = None
        lista = inc_train(lista, dataset, objective_function, D, args, device, split=0.8, modelPath=modelPath)
    else:
        # Proposed constrained learning
        lista, NU = train(lista, dataset, optimizer, objective_function, D, args, device, split=0.8, modelPath=modelPath)

    # Testing
    if not os.path.exists("Results"):
        os.makedirs("Results")
    resultPath = f"./Results/{args.Trial}_L{args.nLayers}_lr_{args.lr}_lrdual_{args.lr_dual}_eps_{args.eps}_constrained_{args.constrained}.pkl"
    logging.debug("\n Testing ...")
    checkpoint = torch.load(modelPath+"LISTA_best.pth")
    lista.load_state_dict(checkpoint["model_state_dict"])
    lista = lista.to(device)
    lista.eval()
    
    testloss, _ = evaluate(lista, testset, objective_function, eps=args.eps, saveResults=True, resultPath=resultPath,
                             SysID=D.to(device), NU=NU, device=device)

    printing(vars(args))
    del testset
    torch.cuda.empty_cache()

def testLoss(lista, device="cpu"):
    _, testset, W, _ = load_data()
    with torch.no_grad():
        print(testset[0])
        xTest, yTest_opt = testset[:]
        xTest = xTest.T.to(device)
        yTest_opt = yTest_opt.T.to(device).float()
        yTest, outsTest = lista(xTest, W.float())
        loss = loss_function(z=yTest, z_opt=yTest_opt)
        objFun = objective_function(yTest, xTest, W)

    return loss.item(), objFun.item()
    
if __name__ == "__main__":
    main()
    print('OK!')
