import gravann
import torch
import numpy as np

if __name__ == "__main__":
    gravann.enableCUDA(device=0)
    device = torch.device("cuda:0")
    differential_training = True
    model, encoding, sample, c, use_acc, mascon_points,\
        mascon_masses_u, mascon_masses_nu, params=gravann.load_model_run(folderpath=r"D:\lwt\大地与二次层初始化方案\Erosd\MD/",differential_training=differential_training)
    print("长度：",len(mascon_points))
    model=model.to(device)
    print(model)
    sampling_altitudes = np.asarray([0.001, 0.0025,0.005,0.0075, 0.01,0.025, 0.05,0.075, 0.1, 0.25,0.5, 0.75,1.0,2.5, 5.0, 7.5,10.0,25.0, 50.0])
    validation_results =gravann. validation(
        model, encoding, mascon_points, mascon_masses_u,
        use_acc, "3dmeshes/" + sample,
        sampling_altitudes=sampling_altitudes,
        mascon_masses_nu=mascon_masses_nu,
        N_integration=500000, N=1000)

    # 保存为cvs数据
    run_folder=r"D:\lwt\大地与二次层初始化方案\Erosd\sh/"


    gravann._io.save_results1(
                 validation_results, model, run_folder)
    gravann._io.save_plots1(model, encoding, mascon_points,  run_folder,"3dmeshes/" + sample, c, 2500)


