import torch
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ._utils import max_min_distance
from ._plots import plot_model_vs_mascon_contours, plot_model_rejection, plot_model_vs_mascon_rejection


def load_sample(sample, use_differential=False):
    """Loads the mascon model of the sample

    Args:
        sample (str): Sample to load

    Returns:
        torch tensors: points and masses of the sample
    """

    with open("mascons/" + sample, "rb") as file:
        mascon_points, mascon_masses_u, name = pk.load(file)

    mascon_points = torch.tensor(mascon_points)
    mascon_masses_u = torch.tensor(mascon_masses_u)

    if use_differential:
        try:
            with open("mascons/"+sample[:-3]+"_nu.pk", "rb") as file:
                _, mascon_masses_nu, _ = pk.load(file)
            mascon_masses_nu = torch.tensor(mascon_masses_nu)
            print("Loaded non-uniform model")
        except:
            mascon_masses_nu = None
    else:
        mascon_masses_nu = None

    # If we are on the GPU , make sure these are on the GPU. Some mascons were stored as tensors on the CPU. it is weird.
    if torch.cuda.is_available():
        mascon_points = mascon_points.cuda()
        mascon_masses_u = mascon_masses_u.cuda()
        if mascon_masses_nu is not None:
            mascon_masses_nu = mascon_masses_nu.cuda()

    print("Name: ", name)
    print("Number of mascon_points: ", len(mascon_points))
    print("Total mass: ", sum(mascon_masses_u).item())
    return mascon_points, mascon_masses_u, mascon_masses_nu


def save_results(loss_log, weighted_average_log, validation_results, model, folder):
    """Stores the results of a run

    Args:
        loss_log (list): list of losses recorded
        weighted_average_log (list): list of weighted average losses recorded
        validation_results (pandas.df): results of the validation as dataframe
        model (torch model): Torch model that was trained
        folder (str): results folder of the run
    """
    print(f"Saving run results to {folder} ...", end="")
    np.save(folder+"loss_log.npy", loss_log)
    np.save(folder+"weighted_average_log.npy", loss_log)
    torch.save(model.state_dict(), folder + "last_model.mdl")
    validation_results.to_csv(folder + "validation_results.csv", index=False)
    print("Done.")


def save_plots(model, encoding, mascon_points, lr_log, loss_log, weighted_average_log, vision_loss_log, n_inferences, folder, c, N):
    """Creates plots using the model and stores them

    Args:
        model (torch nn): trained model
        encoding (func): encoding function
        mascon_points (torch tensor): Points of the mascon model
        lr_log (list): list of learning rates
        loss_log (list): list of losses recorded
        weighted_average_log (list): list of weighted average losses recorded
        vision_loss_log (list): list of vision loss values
        n_inferences (list): list of number of model evaluations
        folder (str): results folder of the run
    """
    print("Creating rejection plot...", end="")
    plot_model_rejection(model, encoding, views_2d=True,
                         bw=True, N=N, alpha=0.1, s=50, save_path=folder + "rejection_plot_iter999999.png",save_data=folder + "rejection_data_iter999999.pk", c=c)
    print("Done.")
    print("Creating model_vs_mascon_rejection plot...", end="")
    plot_model_vs_mascon_rejection(
        model, encoding, mascon_points, N=N, save_path=folder + "model_vs_mascon_rejection.png", c=c)
    print("Done.")

    print("Creating model_vs_mascon_contours plot...", end="")
    plot_model_vs_mascon_contours(
        model, encoding, mascon_points, N=N, save_path=folder + "contour_plot_iter999999.png",save_data=folder + "contour_plot_iter999999", c=c)
    print("Done.")

    print("Creating loss plots...", end="")
    plt.figure()
    abscissa = np.cumsum(n_inferences)
    plt.semilogy(abscissa, loss_log)
    plt.semilogy(abscissa, weighted_average_log)
    plt.semilogy(abscissa, vision_loss_log)
    plt.xlabel("Thousands of model evaluations")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Weighted Average Loss",
                "Vision Loss"])
    plt.savefig(folder+"loss_plot.png", dpi=150)
    print("Done.")

    print("Creating LR plot...")
    plt.figure()
    abscissa = np.cumsum(n_inferences)
    plt.semilogy(abscissa, lr_log)
    plt.xlabel("Thousands of model evaluations")
    plt.ylabel("LR")
    plt.savefig(folder+"lr_plot.png", dpi=150)


# 保存为csv文件
def save_csv(point,acc,file):
    point1 = torch.tensor([item.cpu().detach().numpy() for item in point])
    acc1 = torch.tensor([item.cpu().detach().numpy() for item in acc])
    point2=np.array(point1.cpu())
    point2=point2.reshape(-1,3)
    acc2=np.array(acc1.cpu())
    acc2 = acc2.reshape(-1, 3)

    # 加速度与坐标拼接
    point_ade=np.concatenate((point2, acc2), axis=1)
    acc_s=row_norm(acc2)
    point_accv=np.concatenate((point_ade,acc_s),axis=1)

    cols = [
        "x",
        "y",
        "z",
        "ax",
        "ay",
        "az",
        "a",
    ]
    results = pd.DataFrame(point_accv, columns=cols)
    results.to_csv(file, index=False)

# 求取数组的二范数
def row_norm(acc_matrix):
    acc_norms = []
    for row in acc_matrix:
        norm = np.linalg.norm(row)
        acc_norms.append(norm)
    acc_norms=np.array(acc_norms)
    acc_norms=np.reshape(acc_norms,(-1,1))

    return acc_norms

def save_results1(validation_results, model, folder):
    """Stores the results of a run

    Args:
        loss_log (list): list of losses recorded
        weighted_average_log (list): list of weighted average losses recorded
        validation_results (pandas.df): results of the validation as dataframe
        model (torch model): Torch model that was trained
        folder (str): results folder of the run
    """
    print(f"Saving run results to {folder} ...", end="")
    torch.save(model.state_dict(), folder + "last_model.mdl")
    validation_results.to_csv(folder + "validation_results.csv", index=False)
    print("Done.")
def save_plots1(model, encoding, mascon_points, folder,mde, c, N):
    """Creates plots using the model and stores them

    Args:
        model (torch nn): trained model
        encoding (func): encoding function
        mascon_points (torch tensor): Points of the mascon model
        lr_log (list): list of learning rates
        loss_log (list): list of losses recorded
        weighted_average_log (list): list of weighted average losses recorded
        vision_loss_log (list): list of vision loss values
        n_inferences (list): list of number of model evaluations
        folder (str): results folder of the run
    """
    print("Creating rejection plot...", end="")
    plot_model_rejection(model, encoding, views_2d=True,
                         bw=True, N=N, alpha=0.1, s=50, save_path=folder + "rejection_plot_iter999999.png",save_data=folder + "rejection_data_iter999999.pk", c=c)
    print("Done.")
    print("Creating model_vs_mascon_rejection plot...", end="")
    plot_model_vs_mascon_rejection(
        model, encoding, mascon_points, N=N, save_path=folder + "model_vs_mascon_rejection.png", c=c)
    print("Done.")

    print("Creating model_vs_mascon_contours plot...", end="")
    plot_model_vs_mascon_contours(
        model, encoding, mascon_points, N=N, save_path=folder + "contour_plot_iter999999.png",save_data=folder + "contour_plot_iter999999", c=c,add_shape_base_value=mde)
    print("Done.")



