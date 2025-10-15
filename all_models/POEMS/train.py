import sys
import os
import numpy as np
import pandas as pd
import itertools as it
import torch
import math
import copy
from pathlib import Path


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#cwd = os.getcwd()
cwd = os.path.abspath(os.path.join(Path(__file__).resolve(),".."))
root_dir = os.path.abspath(os.path.join(cwd,"../.."))
sys.path.insert(1, root_dir)
print("Current working directory: ", cwd)
print("Updated path: ", root_dir)
print(sys.path)
import setup_seed
from load_data_mocs import load_data_mocs
from all_models.POEMS.models import POEMS
from scipy.stats import chi2
import torch.optim as optim
import wandb
import argparse
from helper import EarlyStopper, perform_kmeans, perform_knn, get_sigma_params
import util

seed = 21

def train_POEMS(lr_in, wd_in, batch_size_in, nepoch_in, is_wandb, experiment_note, disease,is_test,model_name="POEMS"):
    setup_seed.setup_seed_21()
    patience= 30
    lr = lr_in
    wd = wd_in
    batch_size = batch_size_in
    nepoch = nepoch_in
    
    if(is_test):
        trained_model_name = "elated-night-2"
        out_dir = str(cwd) + "/results/" + trained_model_name + "/"
        model_dir = str(cwd) + "/trained/" + trained_model_name + "/"
    else:
        train_name = "train_POEMS"
        project_name =  experiment_note + "--" + train_name + "--" + model_name
        out_dir = str(cwd) + "/results/" + project_name + "/"
        model_dir = str(cwd) + "/trained/" + project_name + "/"

    omic1_dim, omic2_dim, omic3_dim, X_train_all, X_val_all, X_test_all, y_train, y_val, y_test, n_clusters = load_data_mocs(disease=disease)

    train_loader = torch.utils.data.DataLoader(dataset=torch.tensor(X_train_all, dtype=torch.float), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=torch.tensor(X_val_all, dtype=torch.float), batch_size=batch_size, shuffle=False)

    if model_name == "POEMS":
        sigmas_init = np.std(X_train_all[:,:omic1_dim],axis=0)
        sig_df, sig_scale = get_sigma_params(sigmas_init,disease)
        row_normalize = False
        omic1_info = {
            "input_dim":omic1_dim,
            "hidden_dim1":512,
            "hidden_dim2":256,
            "hidden_dim3":128,
            "latent_dim":32,
            "sigmas_init":sigmas_init,
            "sig_df": sig_df,
            "sig_scale":sig_scale,
            "row_normalize":row_normalize
        }
        sigmas_init = np.std(X_train_all[:,omic1_dim:omic1_dim+omic2_dim],axis=0)
        sig_df, sig_scale = get_sigma_params(sigmas_init,disease)
        omic2_info = {
            "input_dim":omic2_dim,
            "hidden_dim1":512,
            "hidden_dim2":256,
            "hidden_dim3":128,
            "latent_dim":32,
            "sigmas_init":sigmas_init,
            "sig_df": sig_df,
            "sig_scale":sig_scale,
            "row_normalize":row_normalize
        }
        sigmas_init = np.std(X_train_all[:,omic1_dim+omic2_dim:],axis=0)
        sig_df, sig_scale = get_sigma_params(sigmas_init,disease)
        omic3_info = {
            "input_dim":omic3_dim,
            "hidden_dim1":256,
            "hidden_dim2":128,
            "hidden_dim3":64,
            "latent_dim":32,
            "sigmas_init":sigmas_init,
            "sig_df": sig_df,
            "sig_scale":sig_scale,
            "row_normalize":row_normalize
        }
        model = POEMS(batch_size, omic1_info, omic2_info, omic3_info).to(device)
    else:
        raise ValueError(f"No such model exists")
    
    # WANDB
    if(is_wandb): 
        wandb.init(project=project_name)
        wandb.log({"lr":lr, "wd":wd, "patience":patience, "n_epoch":nepoch,"batch_size":batch_size,
                   "lambda0":model.specific_modules["specific1"].lambda0,"lambda1":model.specific_modules["specific1"].lambda1})
        out_dir = out_dir + str(wandb.run.name) + "/"
    
    os.makedirs(out_dir,exist_ok=True)
    
    beta_kl = 1.0
    
    if(not is_test):
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        early_stopper = EarlyStopper(patience=patience, min_delta=0)
            
        for epoch in range(1,nepoch+1):
            train_loss_dict = init_loss_dict("train")
            for batch_idx, data in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                new_loss_values = model.all_loss(data[:,:omic1_dim],data[:,omic1_dim:omic1_dim+omic2_dim],data[:,omic1_dim+omic2_dim:omic1_dim+omic2_dim+omic3_dim])
                loss = update_loss_dict_return_total_loss(train_loss_dict,new_loss_values,beta_kl,"train")
                loss.backward()
                optimizer.step()
                
                for net_key in ["specific1","specific2","specific3"]:
                    pstar = model.specific_modules[net_key].pstar.detach()
                    thetas = model.specific_modules[net_key].thetas.detach()
                    lambda0 = model.specific_modules[net_key].lambda0
                    lambda1 = model.specific_modules[net_key].lambda1
                    # W = model.specific_modules[net_key].W.detach()
                    W = model.specific_modules[net_key].get_generator_mask().detach()
                    a = model.specific_modules[net_key].a
                    b = model.specific_modules[net_key].b
                    input_dim = model.specific_modules[net_key].config["input_dim"]
                    for k in range(pstar.shape[1]):
                        pstar[:, k] = thetas[k] * torch.exp(-lambda1 * W[:, k].abs()) /\
                                    (thetas[k] * torch.exp(-lambda1 * W[:, k].abs()) + (1-thetas[k]) * torch.exp(-lambda0 * W[:, k].abs()))
                        thetas[k] = (pstar[:, k].sum() + a - 1) / (a + b + input_dim - 2)

            #Mean of the batch losses
            train_loss_dict = {key: value / len(train_loader) for key, value in train_loss_dict.items()}

            model.eval()
            val_loss_dict = init_loss_dict("val")
            with torch.no_grad():
                for batch_idx, data in enumerate(val_loader):
                    new_loss_values = model.all_loss(data[:,:omic1_dim],data[:,omic1_dim:omic1_dim+omic2_dim],data[:,omic1_dim+omic2_dim:omic1_dim+omic2_dim+omic3_dim])
                    loss = update_loss_dict_return_total_loss(val_loss_dict,new_loss_values,beta_kl,"val")

                #Mean of the batch losses
                val_loss_dict = {key: value / len(val_loader) for key, value in val_loss_dict.items()}
                other_metrics_dict = {}
                
                other_metrics_dict = {
                    "min_log_sigma1":torch.min(model.specific_modules["specific1"].log_sigmas.detach()),
                    "min_log_sigma2":torch.min(model.specific_modules["specific2"].log_sigmas.detach()),
                    "min_log_sigma3":torch.min(model.specific_modules["specific3"].log_sigmas.detach()),
                    "w_min_abs_val1":model.specific_modules["specific1"].W.detach().abs().min(),
                    "w_max_abs_val1":model.specific_modules["specific1"].W.detach().abs().max(),
                    "w_min_abs_val2":model.specific_modules["specific2"].W.detach().abs().min(),
                    "w_max_abs_val2":model.specific_modules["specific2"].W.detach().abs().max(),
                    "w_min_abs_val3":model.specific_modules["specific3"].W.detach().abs().min(),
                    "w_max_abs_val3":model.specific_modules["specific3"].W.detach().abs().max(),
                    "beta_kl":beta_kl,
                }
                loss_to_early_stop = val_loss_dict["val_total_loss_all"]    
                is_early_stop = early_stopper.early_stop(loss_to_early_stop.item())
                if(epoch % 10 == 0 or is_early_stop):
                    final_embedding,alphas = model.get_final_embedding(X_val_all)
                    final_embedding = final_embedding.detach()
                    alphas = alphas.detach()
                    kmeans_acc_mean,kmeans_acc_std,kmeans_nmi_mean,kmeans_nmi_std,silhouette_mean,silhouette_std = perform_kmeans(final_embedding, y_val,n_clusters)
                    knn_acc_mean,knn_acc_std = perform_knn(final_embedding,y_val,n_clusters)
                    other_metrics_dict["val_kmeans_nmi_mean"] = kmeans_nmi_mean
                    other_metrics_dict["val_knn_acc_mean"] = knn_acc_mean
                    other_metrics_dict["val_alpha1_mean"] = alphas[:, 0].mean().item()
                    other_metrics_dict["val_alpha2_mean"] = alphas[:, 1].mean().item()
                    other_metrics_dict["val_alpha3_mean"] = alphas[:, 2].mean().item()
                    other_metrics_dict["val_silhouette_mean"] = silhouette_mean
                    other_metrics_dict["val_silhouette_std"] = silhouette_std
                    print("Epoch", epoch,":", "Train KL loss:",train_loss_dict["train_kl_loss_all"].item(),"Train loss:", train_loss_dict["train_total_loss_all"].item(), "Val loss:", val_loss_dict["val_total_loss_all"].item())
                    if(is_wandb):
                        wandb.log(train_loss_dict|val_loss_dict|other_metrics_dict) #merge dictionaries
                    if is_early_stop:
                        print("Early stopping")
                        break 
                
                if(epoch % 1000 == 0):
                    val_out_dir = out_dir + "val"+ str(epoch)+ "/"
                    os.makedirs(val_out_dir,exist_ok=True)
                    final_embedding,alphas = model.get_final_embedding(X_val_all)
                    final_embedding = final_embedding.detach()
                    alphas = alphas.detach()
                    np.savetxt(val_out_dir + "_final_em_val.csv", final_embedding.numpy(), delimiter=",")
                    kmeans_acc_mean,kmeans_acc_std,kmeans_nmi_mean,kmeans_nmi_std,silhouette_mean,silhouette_std = perform_kmeans(final_embedding, y_val,n_clusters)
                    knn_acc_mean,knn_acc_std = perform_knn(final_embedding,y_val,n_clusters)
                    print("Kmeans accuracy mean-std: ",kmeans_acc_mean, kmeans_acc_std,"  KNN accuracy mean-std: ", knn_acc_mean,knn_acc_std)
                    util.visualize_Ws(model,val_out_dir)
                    util.plot_tsne(final_embedding,y_val,val_out_dir)
                    util.plot_umap(final_embedding,y_val,val_out_dir)

            model.train()
    
    
    if(is_test):
        model.load_state_dict(torch.load(str(cwd) + "/trained/" + trained_model_name + "/"+"model.pth"))
    else:
        os.makedirs(model_dir,exist_ok=True)
        torch.save(model.state_dict(), model_dir+"model.pth")
    # EVALUATION
    model.eval()
    # test_model = copy.deepcopy(model).to(device)
    # test_model.load_state_dict(torch.load(model_dir+"model"))
    # test_model.eval()
    with torch.no_grad():
        final_embedding, alphas = model.get_final_embedding(X_test_all)
        final_embedding = final_embedding.detach()
        alphas = alphas.detach()
        np.savetxt(out_dir +  "final_em_test.csv", final_embedding.numpy(), delimiter=",")
        kmeans_acc_mean,kmeans_acc_std,kmeans_nmi_mean,kmeans_nmi_std,silhouette_mean,silhouette_std = perform_kmeans(final_embedding, y_test,n_clusters)
        knn_acc_mean,knn_acc_std = perform_knn(final_embedding,y_test,n_clusters)
        util.plot_tsne(final_embedding,y_test,out_dir)
        util.plot_umap(final_embedding,y_test,out_dir)
        if disease == "brca":
            gene_names_1 = pd.read_csv(os.path.join(sys.path[1],'data',disease,'1_featname.csv'),header=None).iloc[:, 0].tolist()
            gene_names_2 = pd.read_csv(os.path.join(sys.path[1],'data',disease,'2_featname.csv'),header=None).iloc[:, 0].tolist()
            gene_names_3 = pd.read_csv(os.path.join(sys.path[1],'data',disease,'3_featname.csv'),header=None).iloc[:, 0].tolist()
            util.visualize_Ws(model,out_dir)
            subtypes = ["Normal-like", "Basal-like", "HER2-enriched", "Luminal A", "Luminal B"]
            omics = ["mRNA","DNAMeth","miRNA"]
            X1_test = X_test_all[:,:omic1_dim]
            X2_test = X_test_all[:,omic1_dim:omic1_dim+omic2_dim]
            X3_test = X_test_all[:,omic1_dim+omic2_dim:]
            util.visualize_final_embedding(final_embedding,y_test,dir=out_dir)
            # util.plot_per_latent_dist_by_subtype(final_embedding,y_test,subtype_names=subtypes,dir=out_dir)
            # util.plot_per_subtype_latent_dist(final_embedding,y_test,subtype_names=subtypes,dir=out_dir)
            util.plot_subtype_correlations(X1_test,X2_test,X3_test,final_embedding,y_test,subtype_names=subtypes,dir=out_dir)
            util.plot_feature_importance(model,disease,omic_names=omics,dir=out_dir)
            util.plot_gating_alphas_stacked(alphas,dir=out_dir)
        
        
    if(is_wandb): 
        wandb.log({"kmeans_acc_mean":kmeans_acc_mean,"kmeans_acc_std":kmeans_acc_std,
                   "kmeans_nmi_mean":kmeans_nmi_mean,"kmeans_nmi_std":kmeans_nmi_std,
                   "knn_acc_mean":knn_acc_mean,"knn_acc_std":knn_acc_std,
                   "test_alpha1_mean":alphas[:, 0].mean().item(),"test_alpha2_mean":alphas[:, 1].mean().item(),"test_alpha3_mean":alphas[:, 2].mean().item()})
        wandb.finish()   
        

def update_loss_dict_return_total_loss(loss_dict, new_loss_values, beta_kl, mode):
    rec_loss_all = new_loss_values["rec_loss"][0] + new_loss_values["rec_loss"][1] + new_loss_values["rec_loss"][2]
    kl_loss_all = new_loss_values["kl_loss"][0]
    mask_loss_all = new_loss_values["mask_loss"][0] + new_loss_values["mask_loss"][1] + new_loss_values["mask_loss"][2]
    
    sigma_loss_all = new_loss_values["sigma_loss"][0] + new_loss_values["sigma_loss"][1] + new_loss_values["sigma_loss"][2]
    loss_dict[mode+"_sigma_loss_all"] += sigma_loss_all.detach()
    loss_dict[mode+"_sigma_loss1"] += new_loss_values["sigma_loss"][0].detach()    
    loss_dict[mode+"_sigma_loss2"] += new_loss_values["sigma_loss"][1].detach()
    loss_dict[mode+"_sigma_loss3"] += new_loss_values["sigma_loss"][2].detach()
    
    loss_dict[mode+"_mse_loss1"] += new_loss_values["mse_loss"][0].detach()    
    loss_dict[mode+"_mse_loss2"] += new_loss_values["mse_loss"][1].detach()
    loss_dict[mode+"_mse_loss3"] += new_loss_values["mse_loss"][2].detach()
    
    loss_dict[mode+"_rec_loss_all"] += rec_loss_all.detach()
    loss_dict[mode+"_kl_loss_all"] += kl_loss_all.detach()
    loss_dict[mode+"_mask_loss_all"] += mask_loss_all.detach()
    loss_dict[mode+"_rec_loss1"] += new_loss_values["rec_loss"][0].detach()
    loss_dict[mode+"_mask_loss1"] += new_loss_values["mask_loss"][0].detach()
    loss_dict[mode+"_rec_loss2"] += new_loss_values["rec_loss"][1].detach()
    loss_dict[mode+"_mask_loss2"] += new_loss_values["mask_loss"][1].detach()
    loss_dict[mode+"_rec_loss3"] += new_loss_values["rec_loss"][2].detach()
    loss_dict[mode+"_mask_loss3"] += new_loss_values["mask_loss"][2].detach()
    total_loss_all = rec_loss_all + beta_kl * kl_loss_all + mask_loss_all + sigma_loss_all 
    loss_dict[mode+"_total_loss_all"] += total_loss_all.detach()
    return total_loss_all

def init_loss_dict(mode):
    loss_dict = {}
    loss_dict[mode+"_rec_loss_all"] = 0.0
    loss_dict[mode+"_kl_loss_all"] = 0.0
    loss_dict[mode+"_mask_loss_all"] = 0.0
    
    loss_dict[mode+"_sigma_loss_all"] = 0.0
    loss_dict[mode+"_sigma_loss1"] = 0.0
    loss_dict[mode+"_sigma_loss2"] = 0.0
    loss_dict[mode+"_sigma_loss3"] = 0.0
    
    loss_dict[mode+"_mse_loss1"] = 0.0
    loss_dict[mode+"_mse_loss2"] = 0.0
    loss_dict[mode+"_mse_loss3"] = 0.0
    
    loss_dict[mode+"_rec_loss1"] = 0.0
    loss_dict[mode+"_mask_loss1"] = 0.0
    
    loss_dict[mode+"_rec_loss2"] = 0.0
    loss_dict[mode+"_mask_loss2"] = 0.0
    
    loss_dict[mode+"_rec_loss3"] = 0.0
    loss_dict[mode+"_mask_loss3"] = 0.0
    
    loss_dict[mode+"_total_loss_all"] = 0.0
    return loss_dict
