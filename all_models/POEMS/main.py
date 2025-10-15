import wandb
import argparse
from train import train_POEMS

def main():
    # wandb.login(key="[INSERT_KEY_HERE]")
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--lr',type=float,required=True)
    # parser.add_argument('--wd',type=float,required=True)
    # parser.add_argument('--batch_size',type=int,required=True)
    # parser.add_argument('--experiment_note',type=str,required=True)
    # parser.add_argument('--disease',type=str,required=True)
    # args = parser.parse_args()
    # train_MocsSparseVAE(lr_in=args.lr, wd_in=args.wd, batch_size_in=args.batch_size, nepoch_in=1000, is_wandb=True, anneal_kl_mon=False, experiment_note=args.experiment_note, disease=args.disease)
    train_POEMS(lr_in=0.0009, wd_in=0.0001, batch_size_in=512, nepoch_in=5000, is_wandb=False, experiment_note="POEMS", disease="brca",is_test=True)
    #evaluate_MocsSparseVAE(batch_size_in=64,model_path="mocs/results/sharedspecific--train_MocsSparseVAE--MocsSparseVAE/dandy-smoke-8")
if __name__ == '__main__':
    main()  
