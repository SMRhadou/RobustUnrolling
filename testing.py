import os
import torch
import argparse

from training import evaluate, objective_function
from ISTA import ISTA
from LISTA import LISTA
from utils import load_data, parser
from utils import plotting, plot_histograms

if not os.path.exists("figs"):
    os.makedirs("figs")

FLAGS = argparse.ArgumentParser()
_, args = parser(FLAGS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load In-distribution Data
dataset, testset, D, _ = load_data(data_dir=os.getcwd())

# Initialize LISTA
N = D.shape[0]
M = D.shape[1]

# Load constrained model
lista_constrained = LISTA(nLayers=args.nLayers, N=N, M=M).to(device)
checkpoint = torch.load("./models/Noisy_distLISTA1_eps1_constrained_full/LISTA_best.pth")
lista_constrained.load_state_dict(checkpoint["model_state_dict"])
lista_constrained.to(device)
lista_constrained.eval()
testloss, layer_outputs = evaluate(lista_constrained, testset, objective_function, eps=args.eps, SysID=D.to(device), device=device)


# Load unconstrained model
lista_unconstrained = LISTA(nLayers=args.nLayers, N=N, M=M).to(device)
checkpoint = torch.load("./models/Noisy_distLISTA1_eps1_unconstrainedlr6/LISTA_best.pth")
lista_unconstrained.load_state_dict(checkpoint["model_state_dict"])
lista_unconstrained.to(device)
lista_unconstrained.eval()
utestloss, ulayer_outputs = evaluate(lista_unconstrained, testset, objective_function, eps=args.eps, SysID=D.to(device), device=device)


# In-distribution results
plotting(objective_function, testset[:][1].T, ulayer_outputs, layer_outputs, title='original')


# OOD experiments
testlosses, utestlosses = [], []
p_size = [0.05, 0.1, 0.15, 0.2, 0.25]
for p in p_size:
    n = p*torch.randn_like(testset[:][0])
    per_images = torch.clamp(testset[:][0]+n, 0, 1)
    # w = torch.randn_like(D)
    new_optimal, _ = ISTA(per_images, D, alpha=0.5, iters=1000, device=device)
    OODset = torch.utils.data.TensorDataset(per_images, new_optimal.T)
    testloss, layer_outputs = evaluate(lista_constrained, OODset, objective_function, eps=args.eps, SysID=D.to(device), device=device)
    utestloss, ulayer_outputs = evaluate(lista_unconstrained, OODset, objective_function, eps=args.eps, SysID=D.to(device), device=device)
    plot_histograms(new_optimal, title=f'p=ISTA_{p}')
    plot_histograms(new_optimal, value2=ulayer_outputs[args.nLayers], title=f'ISTALISTA_p={p}_')
    plot_histograms(layer_outputs[args.nLayers], value2=ulayer_outputs[args.nLayers], title=f'constraints_p={p}', c1='b')
    testlosses.append(testloss)
    utestlosses.append(utestloss)
# plotting(objective_function, new_optimal, ulayer_outputs, layer_outputs, title='OOD')



print('Ok!')
