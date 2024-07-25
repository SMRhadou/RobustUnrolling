import os
import torch
import argparse
import matplotlib.pyplot as plt

from training import evaluate, objective_function, noisy_evaluate
from ISTA import ISTA
from LISTA import LISTA
from utils import load_data, parser
from utils import plotting, plot_histograms, make_video, plot_noisyOuts, plotting_OOD

plt.rc('font', family='serif', serif='cm10')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

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
lista_unconstrained = LISTA(nLayers=10, N=N, M=M).to(device)
checkpoint = torch.load("./models/Noisy_distLISTA1_eps1_unconstrainedlr6/LISTA_best.pth")
# checkpoint = torch.load("./models/unconstrained_1e5_L10/LISTA_best.pth")
lista_unconstrained.load_state_dict(checkpoint["model_state_dict"])
lista_unconstrained.to(device)
lista_unconstrained.eval()
utestloss, ulayer_outputs = evaluate(lista_unconstrained, testset, objective_function, eps=args.eps, SysID=D.to(device), device=device)

# Load unconstrained model trained with noise
lista_unconstrained_noise = LISTA(nLayers=10, N=N, M=M).to(device)
checkpoint = torch.load("./models/noisy_unconstrained_L10/LISTA_best.pth")
# checkpoint = torch.load("./models/noFirstCon_constrained_again_L10/LISTA_best.pth")
lista_unconstrained_noise.load_state_dict(checkpoint["model_state_dict"])
lista_unconstrained_noise.to(device)
lista_unconstrained_noise.eval()
utestloss_noise, ulayer_outputs_noise = evaluate(lista_unconstrained_noise, testset, objective_function, eps=args.eps, SysID=D.to(device), device=device)

# plot_histograms(testset[:][1].T, title=f'original', xmax=ulayer_outputs[args.nLayers].norm(p=1, dim=0).max().item()+2)
# plot_histograms(testset[:][1].T, value2=ulayer_outputs[args.nLayers], title=f'original2', xmax=ulayer_outputs[args.nLayers].norm(p=1, dim=0).max().item()+2)
# plot_histograms(testset[:][1].T, value2=ulayer_outputs[args.nLayers], value3=layer_outputs[args.nLayers], title=f'original3', xmax=ulayer_outputs[args.nLayers].norm(p=1, dim=0).max().item()+2)
# In-distribution results
# plotting(objective_function, testset[:][1].T, ulayer_outputs, layer_outputs, ulayer_outputs_noise, x=testset[:][0].T, W=D, title='noisy')
# make_video(testset[:][1].T, ulayer_outputs, layer_outputs, 'k', 'r', 'b', duration=8)


# # Robustness to Perturbations
# results_constrained = {}
# results_unconstrained = {}
# results_unconstrained_noise = {}

# betas = [0.01, 0.1, 1, 10, 100, 1000]
# print(args.eps)
# for beta in betas:
#     mean, var, l1_norm, l1_var, descentPercentage = noisy_evaluate(lista_constrained, testset, objective_function, args.eps, saveResults=True, resultPath=f"./Results/{beta}_noisy_constrained",
#                                 SysID=D.to(device), device=device, beta=beta)
#     results_constrained[beta] = (mean, var, l1_norm, l1_var, descentPercentage)

#     mean, var, l1_norm, l1_var, _ = noisy_evaluate(lista_unconstrained, testset, objective_function, args.eps, saveResults=True, resultPath=f"./Results/{beta}_noisy_unconstrained",
#                                 SysID=D.to(device), device=device, beta=beta)
#     results_unconstrained[beta] = (mean, var, l1_norm, l1_var)

#     mean, var, l1_norm, l1_var, _ = noisy_evaluate(lista_unconstrained_noise, testset, objective_function, args.eps, saveResults=True, resultPath=f"./Results/{beta}_noisy_unconstrained",
#                                 SysID=D.to(device), device=device, beta=beta)
#     results_unconstrained_noise[beta] = (mean, var, l1_norm, l1_var)

# plot_noisyOuts(results_constrained, results_unconstrained, results_unconstrained_noise, betas, testset[:][1].T)


# OOD experiments
testlosses, utestlosses, utestlosses_noise = {}, {}, {}
testlosses2, utestlosses2, utestlosses_noise2 = {}, {}, {}
# p_size = [0.05, 0.1, 0.15, 0.2, 0.25]
p_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for p in p_size:
    n = p*torch.randn_like(testset[:][0])
    per_images = torch.clamp(testset[:][0]+n, 0, 1)
    # w = torch.randn_like(D)
    new_optimal, _ = ISTA(per_images, D, alpha=0.5, iters=1000, device=device)
    OODset = torch.utils.data.TensorDataset(per_images, new_optimal.T)
    testloss, layer_outputs = evaluate(lista_constrained, OODset, objective_function, eps=args.eps, SysID=D.to(device), device=device)
    utestloss, ulayer_outputs = evaluate(lista_unconstrained, OODset, objective_function, eps=args.eps, SysID=D.to(device), device=device)
    utestloss, ulayer_outputs_noise = evaluate(lista_unconstrained_noise, OODset, objective_function, eps=args.eps, SysID=D.to(device), device=device)
    # plot_histograms(new_optimal, title=f'ISTA_p={p}', xmax=ulayer_outputs[args.nLayers].norm(p=1, dim=0).max().item()+2)
    # plot_histograms(new_optimal, value2=ulayer_outputs[args.nLayers], title=f'ISTALISTA_p={p}_', xmax=ulayer_outputs[args.nLayers].norm(p=1, dim=0).max().item()+2)
    # plot_histograms(new_optimal, value2=ulayer_outputs[10], value3=layer_outputs[args.nLayers], title=f'noisy_constraints_p={p}', xmax=ulayer_outputs[10].norm(p=1, dim=0).max().item()+2)
    testlosses[p] = layer_outputs[args.nLayers]
    utestlosses[p] = ulayer_outputs[args.nLayers]
    utestlosses_noise[p] = ulayer_outputs_noise[args.nLayers]
# plotting(objective_function, new_optimal, ulayer_outputs, layer_outputs, title='OOD')
plotting_OOD(p_size, testset[:][1].T, utestlosses, testlosses, utestlosses_noise, x=testset[:][0].T, W=D, title='OOD_long')


print('Ok!')
