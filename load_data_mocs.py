import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_data_mocs(disease):
    # Load BRCA_data
    # Dataset    mRNA expression        DNA methylation    miRNA expression  Samples     Subtypes
    # BRCA         1000                   1000               503              875           5
    print(sys.path)
    base_path = os.path.join('data', disease)
    print(base_path)
    if disease == 'brca':
        view1_data = pd.read_csv(os.path.join(sys.path[1],'data',disease,'1_all.csv'), header=None)  # mRNA
        view2_data = pd.read_csv(os.path.join(sys.path[1],'data',disease,'2_all.csv'), header=None)  # DNAMeth
        view3_data =pd.read_csv(os.path.join(sys.path[1],'data',disease,'3_all.csv'), header=None)  # miRNA
        label = pd.read_csv(os.path.join(sys.path[1],'data',disease,'labels_all.csv'), header=None) # labels
        y_true = label.to_numpy().flatten()
        y_true = y_true.astype(int)
        n_clusters = 5
        X1 = view1_data.to_numpy() # all [0,1]
        X2 = view2_data.to_numpy() # all [0,1]
        X3 = view3_data.to_numpy() # all [0,1]
    elif disease == 'kirc':
        # Load omic data with headers
        view1_df = pd.read_csv(os.path.join(sys.path[1],'data',disease,"gene1.csv.gz"),index_col=0).T # mRNA
        view2_df = pd.read_csv(os.path.join(sys.path[1],'data',disease,"methyl.csv.gz"),index_col=0).T # DNAMeth
        view3_df = pd.read_csv(os.path.join(sys.path[1],'data',disease,"miRNA1.csv"),index_col=0).T # miRNA
        # Only choose the most varying genes
        q = 0.75
        variances = view1_df.var(axis=0)
        view1_df = view1_df[variances[variances >= np.quantile(variances, q)].index]
        variances = view2_df.var(axis=0)
        view2_df = view2_df[variances[variances >= np.quantile(variances, q)].index]
        variances = view3_df.var(axis=0)
        view3_df = view3_df[variances[variances >= np.quantile(variances, q)].index]
        # Store sample and gene names for future use
        sample_names = view1_df.index.to_list()
        gene_names_1 = view1_df.columns.to_list()
        gene_names_2 = view2_df.columns.to_list()
        gene_names_3 = view3_df.columns.to_list()
        # Load labels and convert one-hot to class index
        label_df = pd.read_csv(os.path.join(sys.path[1],'data',disease, 'label.csv'), index_col=0)
        y_true = label_df.to_numpy().argmax(axis=1).astype(int)
        n_clusters = 2
        # Transpose to (samples x features)
        scaler = MinMaxScaler()
        X1 = scaler.fit_transform(view1_df)
        X2 = scaler.fit_transform(view2_df)
        X3 = scaler.fit_transform(view3_df)
    else:
        raise ValueError(f"No such disease exists")

    assert (X1.min() >= 0-1e8) and (X1.max() <= 1+1e8), "X1 not normalized to [0,1]"
    assert (X2.min() >= 0-1e8) and (X2.max() <= 1+1e8), "X2 not normalized to [0,1]"
    assert (X3.min() >= 0-1e8) and (X3.max() <= 1+1e8), "X3 not normalized to [0,1]"

    Xall = np.concatenate((X1,X2,X3), axis=1)

    X_train_all, X_test_all, y_train, y_test = train_test_split(Xall, y_true, test_size=0.2, random_state=1)
    X_train_all, X_val_all, y_train, y_val = train_test_split(X_train_all, y_train, test_size=0.2, random_state=1)
            
    print("mRNA data shape: ", X1.shape)
    print("DNAmeth data shape: ", X2.shape) 
    print("miRNA data shape: ", X3.shape) 
    print("All data shape: ", Xall.shape) 
    print("Label data shape: ", y_true.shape) 
    print("Number of clusters: ", n_clusters)
        
    return X1.shape[1], X2.shape[1], X3.shape[1], X_train_all, X_val_all, X_test_all, y_train, y_val, y_test, n_clusters
