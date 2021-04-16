# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:29:04 2019
@author: chunmanzuo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import scanpy as sc
import os
import time
import argparse
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch import optim

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.io import mmread

from sklearn.model_selection import train_test_split

def parameter_setting():
	
	parser = argparse.ArgumentParser(description='Single cell Multi-omics data analysis')
	
	outPath = '/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Simulated/Simulated_2/new_test/'
	
	parser.add_argument('--File1', '-F1', type=str, default = '5-counts-RNA.tsv',    help='input file name1')
	parser.add_argument('--File2', '-F2', type=str, default = '0.75-0.2-counts-ATAC.tsv', help='input file name2')
	parser.add_argument('--File2_1', '-F2_1', type=str, default = '0.75-0.2-counts-ATAC_binary.tsv', help='input file name2_1')

	parser.add_argument('--File3', '-F3', type=str, default = '5-cellinfo-RNA.tsv',  help='input meta file')
	parser.add_argument('--File_combine', '-F_com', type=str, default = 'Gene_chromatin_order_combine.tsv',    help='input combine file name')
	parser.add_argument('--File_mofa', '-F_mofa', type=str, default = 'MOFA_combine_cluster.csv',    help='cluster for mofa predicted')
	
	parser.add_argument('--latent_fusion', '-olf1', type=str, default = 'First_simulate_fusion.csv',    help='fusion latent code file')
	parser.add_argument('--latent_1', '-ol1', type=str, default = 'scRNA_latent_combine.csv',    help='first latent code file')
	parser.add_argument('--latent_2', '-ol2', type=str, default = 'scATAC_latent.csv',    help='seconde latent code file')
	parser.add_argument('--denoised_1', '-od1', type=str, default = 'scRNA_seq_denoised.csv', help='outfile for denoised file1')
	parser.add_argument('--normalized_1', '-on1', type=str, default = 'scRNA_seq_normalized_combine.tsv',  help='outfile for normalized file1')
	parser.add_argument('--denoised_2', '-od2', type=str, default = 'scATAC_seq_denoised.csv',  help='outfile for denoised file2')
	
	parser.add_argument('--workdir', '-wk', type=str, default = outPath, help='work path')
	parser.add_argument('--outdir', '-od', type=str, default = outPath, help='Output path')
	
	parser.add_argument('--lr', type=float, default = 1E-3, help='Learning rate')
	parser.add_argument('--weight_decay', type=float, default = 1e-6, help='weight decay')
	parser.add_argument('--eps', type=float, default = 0.01, help='eps')
	
	parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size')
	parser.add_argument('--use_cuda', dest='use_cuda', default=True, action='store_true', help=" whether use cuda(default: True)")
	
	parser.add_argument('--seed', type=int, default=200, help='Random seed for repeat results')
	parser.add_argument('--latent', '-l',type=int, default=10, help='latent layer dim')
	parser.add_argument('--max_epoch', '-me', type=int, default=500, help='Max epoches')
	parser.add_argument('--max_iteration', '-mi', type=int, default=3000, help='Max iteration')
	parser.add_argument('--anneal_epoch', '-ae', type=int, default=200, help='Anneal epoch')
	parser.add_argument('--epoch_per_test', '-ept', type=int, default=5, help='Epoch per test')
	parser.add_argument('--max_ARI', '-ma', type=int, default=-200, help='initial ARI')
	
	return parser


def split_train_test_datasets( Data1, trainRate ):
	### split two datasets into training and testing based on trainRate

	N_samples = np.shape(Data1)[0]
	train_size = int( trainRate * N_samples )

	random_state = np.random.RandomState( 11 )
	permutation = random_state.permutation( N_samples )

	sel_pos = permutation[ : train_size ]
	remain_pos = permutation[ train_size : ]

	trainData = Data1[ sel_pos, : ]
	testData = Data1[ remain_pos, : ]

	return trainData, testData, sel_pos, remain_pos

def read_dataset( File1 = None, File3 = None, File2 = None, File4 = None, transpose = True, test_size_prop = 0.15 ):
	
	### File1 for raw reads count 
	adata = sc.read( File1 )
	
	if File2 is not None:
		adata1 = sc.read( File2 )
		if transpose: 
			adata1 = adata1.transpose()
	else:
		adata1 = None

	if File4 is not None:
		adata2 = sc.read( File4 )
		if transpose: 
			adata2 = adata2.transpose()
	else:
		adata2 = None
	
	if transpose: 
		adata  = adata.transpose()
	
	### File2 for cell group information
	label_ground_truth = []
	
	if File3 is not None:
		
		Data2 = pd.read_csv( File3, header=0, index_col=0 )

		## preprocessing for latter evaluation
		group = Data2['Group'].values
		for g in group:
			g = int(g.split('Group')[1])
			label_ground_truth.append(g)
	else:
		label_ground_truth =  np.ones( len( adata.obs_names ) )
	
	if test_size_prop > 0 :
		train_idx, test_idx = train_test_split(np.arange(adata.n_obs), 
											   test_size = test_size_prop, 
											   random_state = 200)
		spl = pd.Series(['train'] * adata.n_obs)
		spl.iloc[test_idx]  = 'test'
		adata.obs['split']  = spl.values
		
		if File2 is not None:
			adata1.obs['split'] = spl.values

		if File4 is not None:
			adata2.obs['split'] = spl.values

	else:
		train_idx, test_idx = list(range( adata.n_obs )), list(range( adata.n_obs ))

		spl = pd.Series(['train'] * adata.n_obs)
		adata.obs['split']       = spl.values
		
		if File2 is not None:
			adata1.obs['split']  = spl.values

		if File4 is not None:
			adata2.obs['split']  = spl.values
		
		
	adata.obs['split'] = adata.obs['split'].astype('category')
	adata.obs['Group'] = label_ground_truth
	adata.obs['Group'] = adata.obs['Group'].astype('category')
	
	if File2 is not None:
		adata1.obs['split'] = adata1.obs['split'].astype('category')
		adata1.obs['Group'] = label_ground_truth
		adata1.obs['Group'] = adata1.obs['Group'].astype('category')

	if File4 is not None:
		adata2.obs['split'] = adata2.obs['split'].astype('category')
		adata2.obs['Group'] = label_ground_truth
		adata2.obs['Group'] = adata2.obs['Group'].astype('category')
	
	print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))
	
	### here, adata with cells * features
	return adata, adata1, adata2, train_idx, test_idx, label_ground_truth

def normalize( adata, filter_min_counts = True, size_factors = False, 
			   normalize_input = False, logtrans_input = True):
	
	# here, adata.x is raw readc count matrix with cells * Features 
	if filter_min_counts:
		sc.pp.filter_genes(adata, min_counts=1)
		sc.pp.filter_cells(adata, min_counts=1)

	if size_factors or normalize_input or logtrans_input:
		adata.raw = adata.copy()
	else:
		adata.raw = adata

	if size_factors:
		sc.pp.normalize_per_cell(adata)
		adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
	else:
		adata.obs['size_factors'] = 1.0

	if logtrans_input:
		sc.pp.log1p(adata)

	if normalize_input:
		sc.pp.scale(adata)

	return adata

def calculate_ARI( File1, File2 ):
	
	### here File1 is the cell info, File2 is the predited group
	Data1 = pd.read_csv( File1, header=0, index_col=0 )
	## preprocessing for latter evaluation
	group = Data1['Group'].values
	label_ground_truth = []
	for g in group:
		g = int(g.split('Group')[1])
		label_ground_truth.append(g)
		
	Data1 = pd.read_csv( File2, header=0, index_col=0, sep = "\t" )
	
	pre_group = Data1.values[:,0]
	
	ARI_pred = round( metrics.adjusted_rand_score( pre_group, label_ground_truth ), 3 )
	
	return ARI_pred

def calculate_log_library_size( Dataset ):
	
	### Dataset is raw read counts, and should be cells * features
	
	Nsamples     =  np.shape(Dataset)[0]
	library_sum  =  np.log( np.sum( Dataset, axis = 1 ) )
	
	lib_mean     =  np.full( (Nsamples, 1), np.mean(library_sum) )
	lib_var      =  np.full( (Nsamples, 1), np.var(library_sum) ) 
	
	return lib_mean, lib_var

def load_mtx( barcodes_tsv, genes_tsv, matrix_mtx ):
	
		for filepath in [barcodes_tsv, genes_tsv, matrix_mtx]:
			
			if not os.path.exists(filepath):
				
				raise IOError("Required file not found: %s" % filepath)
				
		barcodes = pd.read_csv( barcodes_tsv, delimiter='\t', header=None, usecols=[0] ).values.squeeze()
		genes = pd.read_csv( genes_tsv, delimiter='\t', header=None, usecols=[0] ).values.squeeze()
		
		matrix = mmread( matrix_mtx ).todense()
		
		return matrix,  genes, barcodes

def adjust_learning_rate(init_lr, optimizer, iteration, max_lr, adjust_epoch):

	lr = max(init_lr * (0.9 ** (iteration//adjust_epoch)), max_lr)
	for param_group in optimizer.param_groups:
		param_group["lr"] = lr

	return lr   

def getFinalResult( Result, Result1, Mode = 0 ):

	latent_joint, latent_rna = Result["latent_z"].data.cpu().numpy(), Result1["latent_z"].data.cpu().numpy()
	norm_x1_joint, norm_x1_rna = Result["norm_x1"].data.cpu().numpy(), Result1["norm_x1"].data.cpu().numpy()
	recon_x1_joint, recon_x1_rna = Result["recon_x1"].data.cpu().numpy(), Result1["recon_x1"].data.cpu().numpy()
	recon_x2_joint, recon_x2_rna = Result["recon_x_2"].data.cpu().numpy(), Result1["recon_x_2"].data.cpu().numpy()

	latent_temp, norm_x1_temp, recon_x1_temp, recon_x2_temp = latent_joint, norm_x1_joint, recon_x1_joint, recon_x2_joint

	if Mode ==0 :
		latent_temp, norm_x1_temp = (latent_joint+latent_rna)/2.0, (norm_x1_joint+norm_x1_rna)/2.0
		recon_x1_temp, recon_x2_temp = (recon_x1_joint +recon_x1_rna)/2.0, (recon_x2_joint+recon_x2_rna)/2.0

	return latent_temp, norm_x1_temp, recon_x1_temp, recon_x2_temp


def save_checkpoint(state, folder='./saved_model/', filename='model_best.pth.tar'):

	if not os.path.isdir(folder):

		os.mkdir(folder)

	torch.save(state, os.path.join(folder, filename))


def load_checkpoint(file_path, model, device):

	checkpoint = torch.load(file_path)
	model.load_state_dict(checkpoint['model_state_dict'])

	model.to(device)

	return model
