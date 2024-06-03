import pickle
import matplotlib.pyplot as plt
import argparse
import os
import logging
import seaborn as sns

import numpy as np

import torch

plt.rc('font', family='serif', serif='cm10')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def load_data(data_dir, split=0.8):
    with open(os.path.join(data_dir, "ISTAdata/optimal_shuffled.pkl"), 'rb') as ObjFile:
        W, Z_optimal, dataset = pickle.load(ObjFile)

    nTest = int(split * Z_optimal.shape[1])
    # Training dataset
    # transform = transforms.Compose([transforms.ToPILImage(),
    #                                 transforms.Grayscale(num_output_channels=1),
    #                                 transforms.ToTensor()])
    # dataset  = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    # trainData = torch.stack([transform(x) for x in dataset.data[:nTest]]).reshape((nTest, -1))
    trainData = dataset[:nTest][0]
    trainTargets = Z_optimal.T[:nTest]
    trainset = torch.utils.data.TensorDataset(trainData, trainTargets)
    # Test dataset
    #testData = torch.stack([transform(x) for x in dataset.data[nTest:]]).reshape((len(dataset)-nTest, -1))
    testData = dataset[nTest:][0]
    testTargets = Z_optimal.T[nTest:]
    testset = torch.utils.data.TensorDataset(testData, testTargets)
    del dataset
    torch.cuda.empty_cache()
    return trainset, testset, W, Z_optimal

def parser(FLAGS):
    FLAGS = argparse.ArgumentParser(description='LISTA')
    FLAGS.add_argument('--Trial', type=str, default='authentication', help='Trial')
    FLAGS.add_argument('--nLayers', type=int, default=10, help='nLayers')
    FLAGS.add_argument('--batchSize', type=int, default=512, help='batchSize')
    FLAGS.add_argument('--nEpochs', type=int, default=5000, help='nEpochs')
    FLAGS.add_argument('--lr', type=float, default=1e-5, help='lr')
    FLAGS.add_argument('--lr_dual', type=float, default=1e-3, help='lr_dual')
    FLAGS.add_argument('--eps', type=float, default=0.05, help='epsilon')
    FLAGS.add_argument('--seed', type=int, default=0, help='seed')
    FLAGS.add_argument('--generate', action="store_true")
    FLAGS.add_argument('--constrained', action="store_true")
    FLAGS.add_argument('--noisyOuts', action="store_true")
    FLAGS.add_argument('--supervised', action="store_true")
    # Data generation
    FLAGS.add_argument('--WpretrainPath', type=str, default='./ISTAdata/Wpretrain.pkl', help='Wpretrain file path')
    FLAGS.add_argument('--iters', type=int, default=3000, help='iters')
    FLAGS.add_argument('--train_W', action="store_true")
    FLAGS.add_argument('--pretrained_W', action="store_true")
    return FLAGS, FLAGS.parse_args()

def printing(args):
    logging.debug("="*60)
    for i, item in args.items():
        logging.debug("{}: {}".format(i, item))
    logging.debug("="*60)

def distance_to_optimal(outs, zOpt):
    dist = [torch.norm(outs[l][:zOpt.shape[0]]-zOpt, dim=0).mean() for l in outs.keys()]
    var = [torch.var(torch.norm(outs[l][:zOpt.shape[0]]-zOpt, dim=0)) for l in outs.keys()]
    return torch.stack(dist, dim=0).detach().cpu().numpy(), torch.sqrt(torch.stack(var, dim=0)).detach().cpu().numpy()

def plotting(objective_function, zOpt, outs_unconstrained, outs_constrained, x=None, W=None, cons=None, eps=None, NU=None, title='original'):
    plt.rcParams["font.size"] = "14"
    # plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = False

    nLayers = len(outs_unconstrained.keys()) - 1
    # Figure 1: Distance to optimal
    # sns.set_context('notebook')
    # sns.set_style('darkgrid')
    plt.figure(figsize=(4,3))
    mean, var = distance_to_optimal(outs_constrained, zOpt)
    # plt.subplot(1,2,1)
    plt.plot(mean, 'b', label='constrained LISTA (ours)')
    plt.errorbar(np.arange(11), mean, yerr=var, fmt='b', capsize=3, alpha=0.5)
    mean, var = distance_to_optimal(outs_unconstrained, zOpt)
    plt.plot(mean, 'r', label='LISTA')
    plt.errorbar(np.arange(11), mean, yerr=var, fmt='r', capsize=3, alpha=0.5)
    plt.xlabel("layer $l$")
    plt.ylabel("Distance to optimal $||\mathbf{y}_l - \mathbf{y}^*||_2$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'figs/{title}_distance_to_optimal.pdf')

    # # Figure 2: Gradients
    # x1 = [torch.norm(jacobian(objective_function, (outs_constrained[i], x, W), 
    #                     create_graph=True)[0], p=2, dim=0).mean().item() for i in range(nLayers+1)]
    # x2 = [torch.norm(jacobian(objective_function, (outs_unconstrained[i], x, W), 
    #                     create_graph=True)[0], p=2, dim=0).mean().item() for i in range(nLayers+1)]
    # # plt.figure(2, figsize=(6,4))
    # plt.subplot(1,2,2)
    # plt.plot(x1, 'b', label='constrained LISTA (ours)')
    # plt.plot(x2, 'r', label='LISTA')
    # plt.xlabel("layer $l$")
    # plt.ylabel(r"$\mathbf{E}[\widehat{\nabla} f_{\theta}(\mathbf{y}_l)]$")
    # plt.yscale('log')
    # plt.legend(loc='upper right')
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig('gradients.png')
    # sns.reset_orig()

    # Figure 2: Sparsity measured by l1-norm
    plt.figure(figsize=(16,6))
    j=1
    for i in [0, 1, 3, 5, 7, 9, 10]:
        plt.subplot(2, 4, j)
        plt.hist(torch.norm(outs_constrained[i], p=1, dim=0).detach().cpu().numpy(), label='constrained LISTA', alpha=0.5, color='b')
        plt.hist(torch.norm(outs_unconstrained[i], p=1, dim=0).detach().cpu().numpy(), label='LISTA', alpha=0.5, color='r')
        plt.xlabel(r"$||\mathbf{{y}}_{{ {:1} }}||_1$".format(i))
        plt.ylabel("Frequency across the testset")
        plt.grid()
        plt.legend(loc='best')
        if j == 1:
            plt.xlim((0,1200))
        elif j < 5:
            plt.xlim((0, 1000))
        elif j < 7:
            plt.xlim((0, 800))
        else:
            plt.xlim((0, 50))
        j+=1
    plt.subplot(2, 4, j)
    plt.hist(torch.norm(zOpt, p=1, dim=0).detach().cpu().numpy(), label='Ground truth', alpha=0.5, color='k')
    plt.xlim((0, 50))
    plt.xlabel(r"$||\mathbf{y}^*||_1$ ")
    plt.ylabel("Frequency across the dataset")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figs/{title}_sparsity.pdf')


    # # Figure 3: How often we descend
    # plt.figure(4, figsize=(6,4))
    # plt.plot(np.arange(1,11),np.mean(cons.detach().cpu().numpy()-eps < 0, axis=1)*100, 'k')
    # plt.xlabel("layer l")
    # plt.ylabel("$\%$ of times we descend")
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    # # Figure 4: Dual variables
    # sns.set_context('notebook')
    # sns.set_style('darkgrid')
    # plt.figure(5, figsize=(6,4))
    # for i in range(NU.shape[1]):
    #     plt.plot(NU[:500,i], label=r"$\lambda_{{ {:1} }}$".format(str(i+1)))
    # plt.xlabel("epochs")
    # plt.ylabel("dual variables")
    # plt.legend(loc='upper right')
    # plt.tight_layout()
    # plt.savefig('dual_variables.pdf')
    # sns.reset_orig()

def plot_histograms(value1, value2=None, title=None, c1='k', c2='r'):
    plt.figure()
    plt.hist(torch.norm(value1, p=1, dim=0).detach().cpu().numpy(), label='constrained LISTA', alpha=0.5, color=c1)
    if value2 is not None:
        plt.hist(torch.norm(value2, p=1, dim=0).detach().cpu().numpy(), label='LISTA', alpha=0.5, color=c2)
        # plt.legend(loc='best') 
    plt.xlabel(r"$\|\widehat{\bf x}_L\|_1$")
    plt.ylabel("Frequency across the testset")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'figs/{title}_sparsity.pdf')

def plot_noisyOuts(constrained, unconstrained, betas, zTest_opt):
    plt.rcParams["font.size"] = "14"
    # plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = False

    # sns.set_context('notebook')
    # sns.set_style('darkgrid')
    plt.figure(6, figsize=(12,3))
    plt.subplot(1,3,1)
    plt.plot(betas, [constrained[beta][0][-1] for beta in betas], 'b', label='constrained LISTA')
    plt.errorbar(betas, [constrained[beta][0][-1] for beta in betas], yerr=[constrained[beta][1][-1] for beta in betas], fmt='b', capsize=3, alpha=0.5)
    plt.plot(betas, [unconstrained[beta][0][-1] for beta in betas], 'r', label='LISTA')
    plt.errorbar(betas, [unconstrained[beta][0][-1] for beta in betas], yerr=[unconstrained[beta][1][-1] for beta in betas], fmt='r', capsize=3, alpha=0.5)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r"$\hat\sigma$")
    plt.ylabel("Distance to optimal $||\mathbf{y}_L - \mathbf{y}^*||_2$")
    plt.grid()
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(betas, [constrained[beta][2][-1] for beta in betas], 'b', label='constrained LISTA')
    plt.errorbar(betas, [constrained[beta][2][-1] for beta in betas], yerr=[constrained[beta][3][-1] for beta in betas], fmt='b', capsize=3, alpha=0.5)
    plt.plot(betas, [unconstrained[beta][2][-1] for beta in betas], 'r', label='LISTA')
    plt.errorbar(betas, [unconstrained[beta][2][-1] for beta in betas], yerr=[unconstrained[beta][3][-1] for beta in betas], fmt='r', capsize=3, alpha=0.5)
    plt.plot(betas, [torch.norm(zTest_opt, p=1, dim=0).mean().item() for beta in betas], 'k--', label="Ground truth")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$\hat\sigma$")
    plt.ylabel("$||\mathbf{y}_L||_1$")
    plt.grid()
    plt.legend()

    plt.subplot(1,3,3)
    colors = ['k', 'bo', 'r*']
    i=0
    for beta in betas:
        if beta >=1 and beta<1000:
            plt.plot(np.arange(1,11), constrained[beta][4], colors[i], label=f'$\hat\sigma$ = {beta}')
            i+=1
    plt.xlabel(r"layer $l$")
    plt.ylabel("Descending Frequency")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig('noisy.pdf')
    # sns.reset_orig()


def plot_trajectory(outs_unconstrained, outs_constrained, trajectory, xset, exp, **kwargs):
    plt.rcParams["font.size"] = "11"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(1,3, figsize=(12, 3))

    yset = xset[:,exp].detach().cpu().numpy()
    A = kwargs["SysID"].detach().cpu().numpy()
    delta = 0.025
    x = np.arange(-40, 40.0, delta)
    y = np.arange(-40, 40.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.append(np.expand_dims(X, axis=2), np.expand_dims(Y, axis=2), axis=2)
    f = 0.5 * np.linalg.norm(Z@A.T - yset, ord=2, axis=2)
    CS = ax[0].contour(X, Y, f)
    ax[0].clabel(CS, inline=True, fontsize=10)
    CS = ax[2].contour(X, Y, f)
    ax[2].clabel(CS, inline=True, fontsize=10) 

    x = np.arange(-40, 40.0, delta)
    y = np.arange(-40, 40.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.append(np.expand_dims(X, axis=2), np.expand_dims(Y, axis=2), axis=2)
    f = 0.5 * np.linalg.norm(Z@A.T - yset, ord=2, axis=2)
    CS = ax[1].contour(X, Y, f)
    ax[1].clabel(CS, inline=True, fontsize=10)
    

    # Figure 2
    x = [r[0,exp].item() for r in trajectory[::5]]
    y = [r[1,exp].item() for r in trajectory[::5]]
    ax[0].plot(x, y, 'r-*', markersize=4)
    ax[0].plot(x[0], y[0], 'k*', markersize=4)
    ax[0].plot(x[-1], y[-1], 'g*', markersize=4)
    ax[0].set_xlim([-40, 20])
    ax[0].set_ylim([-40, 40])

    # Figure 2
    x = [outs_unconstrained[l][0,exp].item() for l in outs_unconstrained.keys()]
    y = [outs_unconstrained[l][1,exp].item() for l in outs_unconstrained.keys()]
    ax[1].plot(x, y, 'r-*', markersize=4)
    ax[1].plot(x[0], y[0], 'k*', markersize=4)
    ax[1].plot(x[-1], y[-1], 'g*', markersize=4)
    ax[1].set_xlim([-40, 20])
    ax[1].set_ylim([-40, 40])

    # Figure 3
    x = [outs_constrained[l][0,exp].item() for l in outs_constrained.keys()]
    y = [outs_constrained[l][1,exp].item() for l in outs_constrained.keys()]
    ax[2].plot(x, y, 'r-*', markersize=4)
    ax[2].plot(x[0], y[0], 'k*', markersize=4)
    ax[2].plot(x[-1], y[-1], 'g*', markersize=4)
    ax[2].set_xlim([-40, 20])
    ax[2].set_ylim([-40, 40])
    plt.tight_layout()
    plt.savefig(f'GD_trajectory_{exp}.pdf')
