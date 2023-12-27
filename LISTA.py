import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging
import argparse

from utils import load_data, parser, printing
from training.training_LISTA import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from functools import partial

def objective_function(z, x, W, alpha=0.5):
    return torch.sum(0.5 * torch.norm(x - W.float() @ z, p=2, dim=0)**2) # + alpha * torch.norm(z, p=2, dim=0))
def objective_function_l1reg(z, x, W, alpha=0.5):
    return torch.sum(0.5 * torch.norm(x - W.float() @ z, p=2, dim=0)**2 + alpha * torch.norm(z, p=1, dim=0))

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
        Z = torch.zeros((self.M, X.shape[1]), device=X.device)
        Z = torch.randn_like(Z, device=X.device)
        outs = {} 
        outs[0] = Z
        for l in range(self.nLayers):
            W1 = self.layers[l]['W1']
            W2 = self.layers[l]['W2']
            theta = self.layers[l]['theta']
            #Z = Z - (W.T @ (W @ Z - X))
            Z = W1@Z +  W2@X.float()
            Z = torch.sign(Z) * torch.maximum(torch.abs(Z) - theta, torch.zeros_like(Z))
            if noisyOuts and l < self.nLayers-1:
                grad = torch.norm(jacobian(objective_function, (Z, X, W), create_graph=True)[0], p=2, dim=0).detach()
                Z = Z + torch.randn_like(Z) * torch.log(grad) 
            outs[l+1] = Z/100
        return Z/100, outs

    def forward_noise(self, X:torch.tensor, **kwargs):
        beta = kwargs.get('beta', 100)
        W = kwargs['SysID']
        # Initialize Z
        X = X*100
        Z = torch.zeros((self.M, X.shape[1]), device=X.device)
        Z = torch.randn_like(Z, device=X.device) * beta
        true_outs = {}
        outs = {} 
        true_outs[0] = Z
        outs[0] = Z
        for l in range(self.nLayers):
            W1 = self.layers[l]['W1']
            W2 = self.layers[l]['W2']
            theta = self.layers[l]['theta']
            #Z = Z - (W.T @ (W @ Z - X))
            Z = W1.float()@Z +  W2.float()@X
            Z = torch.sign(Z) * torch.maximum(torch.abs(Z) - theta, torch.zeros_like(Z))
            true_outs[2*l+1] = Z/100
            if  l < self.nLayers-1:
                grad = torch.norm(jacobian(objective_function, (Z, X, W), create_graph=True)[0], p=2, dim=0)
                Z = Z + torch.randn_like(Z) * torch.log(grad) * beta 
            outs[l+1] = Z/100
            true_outs[2*l+2] = Z/100
        return Z/100, outs, true_outs


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

def main():
    FLAGS = argparse.ArgumentParser()
    _, args = parser(FLAGS)

    # Logging
    if not os.path.exists("logs/LISTA"):
        os.makedirs("logs/LISTA")
    logfile = f"./logs/LISTA/logfile_LISTA_{args.Trial}.log"
    logging.basicConfig(filename=logfile, level=logging.DEBUG, force=True)
    if not os.path.exists(f"models/{args.Trial}"):
        os.makedirs(f"models/{args.Trial}")
    modelPath = f"./models/{args.Trial}/"

    logging.debug("Trial ...")
    printing(vars(args))
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # Load CIFAR10
    dataset, testset, W, _ = load_data(data_dir=os.getcwd())
    # trainLoader = DataLoader(trainset, batch_size=batchSize, shuffle=False, num_workers=4)
    # testLoader = DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=4)
    # # Visualize
    # img = next(iter(trainLoader))[0][0]
    # plt.imshow(transforms.ToPILImage()(img))

    # Initialize LISTA
    N = W.shape[0]
    M = W.shape[1]
    lista = LISTA(nLayers=args.nLayers, N=N, M=M).to(device)

    # Initialize optimizer 
    optimizer = optim.Adam(lista.parameters(), lr=args.lr)

    # Training
    torch.cuda.empty_cache()
    lista, NU = train(lista, dataset, optimizer, objective_function, W, args, device, split=0.8, modelPath=modelPath)

    # Testing
    if not os.path.exists("Results"):
        os.makedirs("Results")
    resultPath = f"./Results/{args.Trial}.pkl"
    logging.debug("\n Testing ...")
    checkpoint = torch.load(modelPath+"LISTA_best.pth")
    lista.load_state_dict(checkpoint["model_state_dict"])
    lista = lista.to(device)
    lista.eval()
    
    testloss, _ = evaluate(lista, testset, objective_function, eps=args.eps, saveResults=True, resultPath=resultPath,
                             SysID=W.to(device), NU=NU, device=device)

    printing(vars(args))
    del testset
    torch.cuda.empty_cache()

def partial_train(config, **kwargs):
    # Load data
    dataset, _, W, _ = load_data("/home/samar/Documents/Github/Unrolling")
    nTrain = int(len(dataset) * 0.8)
    nBatches = nTrain // config["batchSize"]
    constrained = kwargs["constrained"]
    noisyOuts = kwargs["noisyOuts"]
   
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            lista = nn.DataParallel(lista)
    
    # Initialize LISTA
    N = W.shape[0]
    M = W.shape[1]
    lista = LISTA(nLayers=config["nLayers"], N=N, M=M, sigma=config["sigma"]).to(device)
    
    # Initialize optimizer 
    optimizer = optim.Adam(lista.parameters(), lr=config["lr"])

    chkpt = session.get_checkpoint()
    if chkpt:
        lista = chkpt.to_dict()["model_state"]
        optimizer = chkpt.to_dict()["optimizer_state"]

    nu = torch.zeros((lista.nLayers,), device=torch.device(device)).double()
    for epoch in range(100):
        # Training
        if constrained:
            lista, nu = constrained_learning(lista, dataset, optimizer, nu, W, batchSize=config["batchSize"], 
                                            nBatches=nBatches, noisyOuts=noisyOuts,
                                            lr_dual = config["lr_dual"], eps=config["eps"], device=device)
        else:
            lista = unconstrained_learning(lista, dataset, optimizer, W, batchSize=config["batchSize"], 
                                            nBatches=nBatches, noisyOuts=noisyOuts, eps=config["eps"], device=device)

        # Validation
        with torch.no_grad():
            xValid, zValid_opt = dataset[nTrain:]
            xValid = xValid.T.to(device)
            zValid_opt = zValid_opt.T.to(device).float()
            zValid, outsValid = lista(xValid, W, noisyOuts=noisyOuts)
            validloss = loss_function(x=xValid, W=W, z=zValid, z_opt=zValid_opt)
            objFun = objective_function(zValid, xValid, W)

        checkpoint = Checkpoint.from_dict(dict(
            model_state=lista.state_dict(), optimizer_state=optimizer.state_dict()
        ))
        session.report({"loss":validloss.item(), "objFun":objFun.item()}, checkpoint=checkpoint)
            
        del xValid, zValid_opt, zValid, outsValid, validloss
        torch.cuda.empty_cache()

def testLoss(lista, device="cpu"):
    _, testset, W, _ = load_data()
    with torch.no_grad():
        print(testset[0])
        xTest, zTest_opt = testset[:]
        xTest = xTest.T.to(device)
        zTest_opt = zTest_opt.T.to(device).float()
        zTest, outsTest = lista(xTest, W.float())
        loss = loss_function(z=zTest, z_opt=zTest_opt)
        objFun = objective_function(zTest, xTest, W)

    return loss.item(), objFun.item()
    
if __name__ == "__main__":
    main()
    print('OK!')
