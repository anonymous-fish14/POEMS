import torch
import torch.nn as nn
from torch.nn import functional as F
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OmicSparseVAE(nn.Module):
    """
    Module for a modality specific branch.
    This module uses an SparseVAE structure.
    """
    def __init__(self, config, key):
        """
        Args:
            config (dict): Configuration dictionary for the omic modality.
            key (str): Either "specific1", "specific2", or "specific3" to determine the architecture.
        """
        super(OmicSparseVAE, self).__init__()
        self.config = config
        self.key = key
        
        self.log_sigmas = nn.Parameter(torch.log(torch.tensor(self.config["sigmas_init"], dtype=torch.float)))
        min_a = torch.min(self.log_sigmas)
        max_a = torch.max(self.log_sigmas)
        self.min_log_sigma_val = torch.min(self.log_sigmas)
        self.register_buffer("min_log_sigma", torch.tensor(self.min_log_sigma_val, dtype=torch.float))
        
        self.sigma_prior_df = self.config["sig_df"]
        self.sigma_prior_scale = self.config["sig_scale"]
  
        # Spike-slab parameters (non-trainable lambdas, a, b stored as attributes)
        self.lambda0 = 5
        self.lambda1 = 0.5
        self.a = 1
        self.b = config["input_dim"]/10
        self.row_normalize = config["row_normalize"]

        # Manually updated, non-trainable parameters for spike-slab prior
        pstar_init = 0.5 * torch.ones(config["input_dim"], config["latent_dim"], dtype=torch.float, device=device)
        self.pstar = nn.Parameter(pstar_init, requires_grad=False)
        self.thetas = nn.Parameter(torch.rand(config["latent_dim"]), requires_grad=False)
        self.W = nn.Parameter(torch.randn(config["input_dim"], config["latent_dim"]), requires_grad=True)
            
        dim_before_latent = 100
        
        # Build encoder and decoder 
        self.encoder = nn.Sequential(nn.Linear(config["input_dim"], dim_before_latent),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(dim_before_latent, dim_before_latent),
                                    nn.ReLU(),
                                    nn.Dropout(0.2))
        
        # Latent space layers
        self.zmean = nn.Linear(dim_before_latent, config["latent_dim"])
        self.zlogvar = nn.Linear(dim_before_latent, config["latent_dim"])
        
        self.generator = nn.Sequential(nn.Linear(config["latent_dim"], dim_before_latent, bias=False),
									nn.ReLU(),
                                    nn.Dropout(0.2))
        # Column means parameters
        column_means = nn.ModuleList([nn.Linear(dim_before_latent, 1) for i in range(config["input_dim"])]) 
        column_means_weight_init = torch.randn(config["input_dim"],dim_before_latent,1)
        column_means_bias_init = torch.zeros(config["input_dim"],1)
        
        with torch.no_grad():
            for j, layer in enumerate(column_means):
                column_means_weight_init.data[j].copy_(layer.weight.data.T)
                column_means_bias_init.data[j].copy_(layer.bias.data)
        self.column_means_weight = nn.Parameter(column_means_weight_init)
        self.column_means_bias = nn.Parameter(column_means_bias_init)
        
    def get_generator_mask(self):
        if self.row_normalize:
            # re-scale so rows sum to 1
            W = self.W.abs() + 1e-6
            W = F.normalize(W, p=1, dim=-1)
        else:
            W = self.W
        return W
	
    def encode(self, x):
        q_z = self.encoder(x)
        z_mean = self.zmean(q_z)
        z_logvar = self.zlogvar(q_z)
        #min_z_logvar = round(torch.min(z_logvar.detach()).item(),2) # only for debugging
        return z_mean, z_logvar
	
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mean + (eps * std)
        return sample 
    
    def decode(self, z):
        mask = self.get_generator_mask()
        input_dim = self.config["input_dim"]
        batch_size = z.shape[0]
        latent_dim = z.shape[1]
        masked_inputs = torch.mul(z.unsqueeze(0), mask.unsqueeze(1)) # Expand and apply mask -> (input_dim, batch_size, latent_dim)
        decoder_input = masked_inputs.view(input_dim * batch_size, latent_dim) # Flatten for decoder -> (input_dim * batch_size, latent_dim)
        hidden = self.generator(decoder_input)
        hidden_out = hidden.view(input_dim, batch_size, -1) # Reshape back -> (input_dim, batch_size, hidden_dim1)
        x_mean = torch.matmul(hidden_out, self.column_means_weight).squeeze(-1) + self.column_means_bias # Compute column-wise means -> (input_dim, batch_size)
        x_mean = x_mean.T  # Reshape -> (batch_size, input_dim)
        #x_mean = F.sigmoid(x_mean)
        return x_mean
    
class POEMS(torch.nn.Module):
    def __init__(self, batch_size, omic1_info, omic2_info, omic3_info, **kwargs):
        super(POEMS, self).__init__()
        self.batch_size = batch_size
        self.temperature = 0.4

        self.specific_modules = nn.ModuleDict({
            "specific1": OmicSparseVAE(omic1_info, key="specific1"),
            "specific2": OmicSparseVAE(omic2_info, key="specific2"),
            "specific3": OmicSparseVAE(omic3_info, key="specific3")
        })

        self.gating_network = nn.Sequential(
            nn.Linear(omic1_info["latent_dim"] + omic2_info["latent_dim"] + omic3_info["latent_dim"], 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)  # Ensures weights sum to 1
        )
        
        self.norm1 = nn.LayerNorm(omic1_info["latent_dim"])
        self.norm2 = nn.LayerNorm(omic2_info["latent_dim"])
        self.norm3 = nn.LayerNorm(omic3_info["latent_dim"])

    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mean + (eps * std)
        return sample 
    
    def encode(self, x1, x2, x3):
        z1_mean, z1_log_var = self.specific_modules["specific1"].encode(x1)
        z2_mean, z2_log_var = self.specific_modules["specific2"].encode(x2)
        z3_mean, z3_log_var = self.specific_modules["specific3"].encode(x3)
        
        #Normalization
        z1_norm = self.norm1(z1_mean)
        z2_norm = self.norm2(z2_mean)
        z3_norm = self.norm3(z3_mean)
        alphas = self.gating_network(torch.cat([z1_norm,z2_norm,z3_norm],dim=1))
        
        #Precisions
        z1_prec = torch.exp(-z1_log_var)
        z2_prec = torch.exp(-z2_log_var)
        z3_prec = torch.exp(-z3_log_var)
        a1 = alphas[:,0].unsqueeze(1)
        a2 = alphas[:,1].unsqueeze(1)
        a3 = alphas[:,2].unsqueeze(1)
        prec_comb = a1*z1_prec + a2*z2_prec + a3*z3_prec
        #Mean
        mean_comb = (a1*z1_prec*z1_mean + a2*z2_prec*z2_mean + a3*z3_prec*z3_mean)/prec_comb
        #Log variance
        log_var_comb = -torch.log(prec_comb)
        z = self.reparameterize(mean_comb,log_var_comb)
        return mean_comb, log_var_comb, z, alphas
        
    def forward(self, x1, x2, x3):
        zmean, zlogvar, z, alphas = self.encode(x1,x2,x3)
        #Omic1
        x1_rec = self.specific_modules["specific1"].decode(z)
        #Omic2
        x2_rec = self.specific_modules["specific2"].decode(z)
        #Omic3
        x3_rec = self.specific_modules["specific3"].decode(z)
        
        return x1_rec, x2_rec, x3_rec, zmean, zlogvar, z, alphas
    
    def reconstruction_loss(self, x_pred, x, net_key):
        # NLL
        sigmas = torch.exp(self.specific_modules[net_key].log_sigmas)
        loss = nn.MSELoss()
        reconstruction_loss = 0.5 * loss(x_pred / sigmas, (x / sigmas))
        return reconstruction_loss
    
    def mse_loss(self, x_pred, x):
        loss = nn.MSELoss()
        reconstruction_loss = loss(x_pred ,x)
        return reconstruction_loss

    def kl_loss(self, z_mu, z_log_var):
        # KL formula of standard prior
        kld = -0.5 * torch.mean(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
        return kld
    
    def sigma_loss(self, net_key):
        sig_loss = (self.batch_size + self.specific_modules[net_key].sigma_prior_df + 2) * self.specific_modules[net_key].log_sigmas.sum() \
			+ 0.5 * self.specific_modules[net_key].sigma_prior_df * self.specific_modules[net_key].sigma_prior_scale * torch.sum(1/torch.exp(2 * self.specific_modules[net_key].log_sigmas))

        sig_loss = sig_loss / self.batch_size # Original calculation
        return sig_loss

    def mask_loss(self, net_key):
        W = self.specific_modules[net_key].get_generator_mask()
        pstar = self.specific_modules[net_key].pstar
        lambda0 = self.specific_modules[net_key].lambda0
        lambda1 = self.specific_modules[net_key].lambda1
        loss = (lambda1 * pstar + lambda0 * (1 - pstar)) * W.abs()
        loss = loss.sum() 
        #loss = loss / self.specific_modules[net_key].config["input_dim"] # Added
        loss = loss / self.batch_size # Original calculation
        return loss
    
    def all_loss(self, x1, x2, x3):
        # Specific
        loss_dict = {
            "rec_loss": [],
            "kl_loss": [],
            "mask_loss": [],
            "sigma_loss":[],
            "mse_loss":[]
        }
        x1_rec, x2_rec, x3_rec, zmean, zlogvar, z, alphas = self.forward(x1,x2,x3)
        # omic 1
        net_key = "specific1"
        loss_dict["rec_loss"].append(self.reconstruction_loss(x1_rec, x1,net_key))
        loss_dict["mask_loss"].append(self.mask_loss(net_key))
        loss_dict["sigma_loss"].append(self.sigma_loss(net_key))
        loss_dict["mse_loss"].append(self.mse_loss(x1_rec, x1))
        # omic 2
        net_key = "specific2"
        loss_dict["rec_loss"].append(self.reconstruction_loss(x2_rec, x2,net_key))
        loss_dict["mask_loss"].append(self.mask_loss(net_key))
        loss_dict["sigma_loss"].append(self.sigma_loss(net_key))
        loss_dict["mse_loss"].append(self.mse_loss(x2_rec, x2))
        # omic 3
        net_key = "specific3"
        loss_dict["rec_loss"].append(self.reconstruction_loss(x3_rec, x3,net_key))
        loss_dict["mask_loss"].append(self.mask_loss(net_key))
        loss_dict["sigma_loss"].append(self.sigma_loss(net_key))
        loss_dict["mse_loss"].append(self.mse_loss(x3_rec, x3))
        #common
        loss_dict["kl_loss"].append(self.kl_loss(zmean,zlogvar))
        return loss_dict

    def get_final_embedding(self, X_test_all):
        with torch.no_grad():
            omic1_dim = self.specific_modules["specific1"].config["input_dim"]
            omic2_dim = self.specific_modules["specific2"].config["input_dim"]
            omic3_dim = self.specific_modules["specific3"].config["input_dim"]
            x1 = torch.tensor(X_test_all[:, :omic1_dim], dtype=torch.float, device=device)
            x2 = torch.tensor(X_test_all[:, omic1_dim : omic1_dim + omic2_dim], dtype=torch.float,device=device)
            x3 = torch.tensor(X_test_all[:, omic1_dim + omic2_dim : omic1_dim + omic2_dim + omic3_dim],dtype=torch.float,device=device)
            zmean, zlogvar, z, weights = self.encode(x1,x2,x3)
            return z, weights
        
