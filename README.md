# scMVAE
Deep joint-leaning single-cell multi-omics model

![image](https://github.com/cmzuo11/scMVAE/blob/master/scMVAE/Figure%201.png)

Overview of scMVAE model with three joint-learning strategies. (A) Overall framework of the scMVAE model. Given the scRNA-seq data (xi with M variables) and scATAC-seq data (yi with N variables) of the same cell i as input, the scMVAE model learned a nonlinear joint embedding (z) of the cells that can be used for multiple analysis tasks (i.e. cell clustering and visualization) through a multimodal encoder with three learning strategies described as (B), and then reconstructed back to the original dimension as output through a decoder for each omics data. Note: the same cell orders for both omics data ensure that one cell corresponds to a point in the low-dimensional space. (B) Illustration model of three learning strategies: (i) ‘PoE’ framework was used to estimate the joint posterior by a product of posterior of each omics data (detailed in Material S1), (ii) ‘NN’ was used to learn the joint-learning space by using a neural network to combine the features extracted by a sub encoder network for each layer data and (iii) ‘Direct’ strategy was used to learn together by directly using the concatenation of the original features of two-layer data as input. Here, the neural networks: NN − fμy−l , NN − fσ y−l , NN − fμy , NN − fθ y , NN − fπ y , were removed from the total network under this learning condition. (C) The distribution to where each variable of scMVAE model belongs. Each omics data were modeled as one ZINB distribution. The detailed description for each variable is given in datasets and preprocessing.

# Installation

scMVAE is implemented in the Pytorch framework. Please run scMVAE on CUDA if possible. DCCA requires python 3.6.12 or later, and torch 1.6.0 or later. The used packages (described by "used_package.txt") for scMVAE can be automatically installed.

* git clone git://github.com/cmzuo11/scMVAE.git

* cd scMVAE

* python set_up.py install

the detailed usage for using scMVAE-PoE model, please check the 'MVAE_test_Adbrain.py'

# Citation

Chunman Zuo, Luonan Chen. Deep-joint-learning analysis model of single cell transcriptome and open chromatin accessibility data. Briefings in Bioinformatics. 2020.(accept)
