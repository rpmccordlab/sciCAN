# sciCAN
Integration of single-cell chromatin accessibility and gene expression data via cycle-consistent adversarial network

# Requirements
###
* numpy
* scipy
* pandas
* scikit-learn
* scanpy
* anndata
* pytorch

# Tutorial

### For quick start
    ##A processed scRNA-seq/scATAC-seq can be found at https://figshare.com/articles/dataset/Mouse_skin_data_by_SHARE-seq/16620367
    ##rna_X: RNA-seq data; dna_X: ATAC-seq data
    ##Both rna_X and dna_X are matrices in which each row represents one cell while each column stands for a gene
    ##rna_X and dna_X don't have to have the same number of cells
    ##rna_X as input for source_trainset while dna_X as input for target_trainset
    
    from sciCAN.network import *
    from sciCAN.training import *
    FeatureExtractor, Generator = Cycle_train_wolabel(epoch=100, batch_size=1024, source_trainset=rna_X, target_trainset=dna_X)
    
    ##visualization integration outcome
    X_tensor_a = torch.tensor(rna_X).float()
    X_tensor_b = torch.tensor(dna_X).float()
    FeatureExtractor.to(torch.device("cpu"))
    X_all_tensor = torch.cat((X_tensor_a,X_tensor_b),0)
    y_pred = FeatureExtractor(X_all_tensor)
    y_pred = F.normalize(y_pred, dim=1,p=2)
    y_pred = torch.Tensor.cpu(y_pred).detach().numpy()##visualize y_pred with UMAP
