#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:07:52 2019

@author: chunmanzuo
"""

import numpy as np
import pandas as pd
import os
import time
import torch
import math
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch import optim
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from tqdm import trange
#from MVAE_model import VAE, AE, MVAE_POE
from utilities import read_dataset, normalize, calculate_log_library_size, parameter_setting, save_checkpoint, adjust_learning_rate
from MVAE_model import AE, MAE, VAE, MVAE, MVAE_POE, ProductOfExperts


def train(args, adata, adata1, model, train_index, test_index, lib_mean, lib_var, lib_mean1, lib_var1, real_groups, final_rate, file_fla, Type1, Type, device, scale_factor ):

	trainData_raw_1 = adata.raw[train_index].X
	testData_raw_1  = adata.raw[test_index].X

	if Type == "Bernoulli":
		used_data = adata1.X
		trainData_2 = adata1[train_index].X
		testData_2  = adata1[test_index].X

	elif Type == "Gaussian":
		used_data1   = Normalized_0_1(adata1.raw.X)
		used_data    = used_data1.X
		trainData_2  = used_data[train_index, : ]
		testData_2   = used_data[test_index, : ]

	else:
		used_data   = adata1.raw.X
		trainData_2 = used_data[train_index, : ]
		testData_2  = used_data[test_index, : ]


	train         = data_utils.TensorDataset( torch.from_numpy( trainData_raw_1 ),
											  torch.from_numpy( lib_mean[train_index] ), 
											  torch.from_numpy( lib_var[train_index] ),
											  torch.from_numpy( lib_mean1[train_index] ), 
											  torch.from_numpy( lib_mean1[train_index] ),
											  torch.from_numpy( trainData_2 ))
	train_loader  = data_utils.DataLoader( train, batch_size = args.batch_size, shuffle = True )

	test          = data_utils.TensorDataset( torch.from_numpy( testData_raw_1 ),
											  torch.from_numpy( lib_mean[test_index] ), 
											  torch.from_numpy( lib_var[test_index] ),
											  torch.from_numpy( lib_mean1[test_index] ), 
											  torch.from_numpy( lib_var1[test_index] ),
											  torch.from_numpy( testData_2 ))
	test_loader   = data_utils.DataLoader( test, batch_size = len(test_index), shuffle = False )

	
	total         = data_utils.TensorDataset( torch.from_numpy( adata.raw.X  ),
											  torch.from_numpy( used_data ))
	total_loader  = data_utils.DataLoader( total, batch_size = (len(test_index)+len(train_index)) , shuffle = False )
	
	args.max_epoch = 500

	train_loss_list  = []

	flag_break = 0
	epoch_count = 0
	reco_epoch_test  = 0
	test_like_max    = 100000
	status = ""

	latent_z  = None
	norm_x1   = None
	recon_x_2 = None
	recon_x1  = None
	norm_x2   = None

	cohen_score1  = 0
	cohen_score2  = 0
	cohen_score3  = 0
	ARI_list1     = 0
	ARI_list2     = 0
	ARI_list3     = 0
	max_iteration = 10000
	args.epoch_per_test = 10

	params = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = optim.Adam( params, lr = args.lr, weight_decay = args.weight_decay, eps = args.eps )

	epoch = 0
	iteration = 0
	start = time.time()

	#model.init_gmm_params( total_loader, device )

	with trange( args.max_epoch, disable=True ) as pbar:

		while True:

			model.train()

			epoch +=  1
			#epoch_lr = adjust_learning_rate( args.lr, optimizer, epoch, final_rate, 10 )
			kl_weight = min( 1, epoch / args.anneal_epoch )

			for batch_idx, ( X1, lib_m, lib_v, lib_m1, lib_v1, X2 ) in enumerate(train_loader):

				X1     = X1.float().to(device)
				lib_m  = lib_m.to(device)
				lib_v  = lib_v.to(device)
				X2     = X2.float().to(device)
				lib_m1 = lib_m1.to(device)
				lib_v1 = lib_v1.to(device)

				X1     = Variable( X1 )
				lib_m  = Variable( lib_m )
				lib_v  = Variable( lib_v )
				lib_m1 = Variable( lib_m1 )
				lib_v1 = Variable( lib_v1 )
				X2     = Variable( X2 )

				optimizer.zero_grad()
				loss1, loss2, pear_loss, kl_divergence_l, kl_divergence_l1, kl_divergence_z = model( X1.float(), X2.float(), lib_m, lib_v, lib_m1, lib_v1 )
				loss = torch.mean( ( scale_factor * loss1 + loss2 + kl_divergence_l + kl_divergence_l1) + (kl_weight*(kl_divergence_z)) )  

				loss.backward()
				optimizer.step()

				iteration += 1 

			epoch_count += 1

			if epoch % args.epoch_per_test == 0 and epoch > 0: 

				model.eval()

				with torch.no_grad():

					for batch_idx, ( X1, lib_m, lib_v, lib_m1, lib_v1, X2 ) in enumerate(test_loader): 

						X1     = X1.float().to(device)
						lib_m  = lib_m.to(device)
						lib_v  = lib_v.to(device)
						X2     = X2.float().to(device)
						lib_m1 = lib_m1.to(device)
						lib_v1 = lib_v1.to(device)

						X1     = Variable( X1 )
						lib_m  = Variable( lib_m )
						lib_v  = Variable( lib_v )
						lib_m1 = Variable( lib_m1 )
						lib_v1 = Variable( lib_v1 )
						X2     = Variable( X2 )

						loss1, loss2, pear_loss, kl_divergence_l, kl_divergence_l1, kl_divergence_z = model( X1.float(), X2.float(), lib_m, lib_v, lib_m1, lib_v1 )
						test_loss = torch.mean( ( scale_factor * loss1 + loss2 + kl_divergence_l + kl_divergence_l1) + (kl_weight*(kl_divergence_z)) )  

						train_loss_list.append( test_loss.item() )

						if math.isnan(test_loss.item()):
							flag_break = 1
							break

						if test_like_max >  test_loss.item():
							test_like_max   = test_loss.item()
							epoch_count  = 0

							for batch_idx, ( X1, X2 ) in enumerate(total_loader): 
								X1     = X1.float().to(device)
								X2     = X2.float().to(device)

								X1     = Variable( X1 )
								X2     = Variable( X2 )

								result    = model.inference( X1.float(), X2.float() )
								latent_z  = result["latent_z"].data.cpu().numpy()
								recon_x_2 = result["recon_x_2"].data.cpu().numpy()
								recon_x1  = result["recon_x1"].data.cpu().numpy()
								norm_x1   = result["norm_x1"].data.cpu().numpy()

								#print(np.shape(latent_z))

								if Type == "ZINB":
									norm_x2 = result["norm_x2"].data.cpu().numpy()

							print( str(epoch)+ "   " + str(loss.item()) +"   " + str(test_loss.item()) +"   " + 
								   str(torch.mean(loss1).item()) +"   "+ str(torch.mean(loss2).item()) +
								   "  kl_divergence_l:  " + str(torch.mean(kl_divergence_l).item()) + " kl_weight: " + str( kl_weight )+
								   " kl_divergence_z: " + str( torch.mean(kl_divergence_z).item() ) )

			if epoch_count >= 30:
				reco_epoch_test = epoch
				status = " larger than 30 "
				break

			if flag_break == 1:
				reco_epoch_test = epoch
				status = " with NA "
				break

			if epoch >= args.max_epoch:
				reco_epoch_test = epoch
				status = " larger than 500 epoch "
				break
			
			if len(train_loss_list) >= 2 :
				if abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] < 1e-4 :
					reco_epoch_test = epoch
					status = " training for the train dataset is converged! "
					break

	duration = time.time() - start
	print('Finish training, total time: ' + str(duration) + 's' + " epoch: " + str(reco_epoch_test) + " status: " + status )

	if latent_z is not None:
		imputed_val  = pd.DataFrame( latent_z, index= adata.obs_names ).to_csv( os.path.join( args.outdir, 
									 str(file_fla) + '_latent_ZINB_final.csv' ) ) 
	if norm_x1 is not None:
		norm_x1_1    = pd.DataFrame( norm_x1, columns =  adata.var_names, 
									 index= adata.obs_names ).to_csv( os.path.join( args.outdir,
									 str(file_fla) + '_scRNA_norm_ZINB_final.csv' ) )

	if recon_x1 is not None:
		recon_x1_1   = pd.DataFrame( recon_x1, columns =  adata.var_names, 
								  index= adata.obs_names ).to_csv( os.path.join( args.outdir,
								  str(file_fla) + '_scRNA_recon_ZINB_final.csv' ) )

	if recon_x_2 is not None:
		recon_x2_1   = pd.DataFrame( recon_x_2, columns =  adata1.var_names, 
								 index= adata1.obs_names ).to_csv( os.path.join( args.outdir, 
								 str(file_fla)+ '_scATAC_recon_ZINB_final.csv') )

	if norm_x2 is not None:
		norm_x2_1   = pd.DataFrame( norm_x2, columns =  adata1.var_names, 
									index= adata1.obs_names ).to_csv( os.path.join( args.outdir, 
									str(file_fla)+ '_scATAC_norm_ZINB_final.csv') )

	print( 'train likelihood is :  '+ str(test_like_max) + ' epoch: ' + str(reco_epoch_test) + " cohen_score1:" + str(cohen_score1) +
		   ' cohen_score2: ' + str(cohen_score2) + ' cohen_score3: ' + str(cohen_score3) + " ARI_list1:" + str(ARI_list1) +
		   ' ARI_list2: ' + str(ARI_list2) + ' ARI_list3: ' + str(ARI_list3) + ' Type: ' + Type )
	
	return test_like_max, reco_epoch_test, cohen_score1, cohen_score2, cohen_score3, ARI_list1, ARI_list2, ARI_list3


def train_with_argas( args ):

	adata, adata1, adata2, train_index, test_index = read_dataset( File1 = os.path.join( args.workdir, args.File1 ),
																   File2 = os.path.join( args.workdir, args.File2 ),  
																   File3 = None,
																   File4 = os.path.join( args.workdir, args.File2_1 ),
																   test_size_prop = 0.1
																	)

	adata  = normalize( adata, 
						size_factors = False, 
						normalize_input = False, 
						logtrans_input = True ) ## here, juts log(x+1)

	adata1 = normalize( adata1, 
						size_factors = False, 
						normalize_input = False, 
						logtrans_input = True ) ## here, juts log(x+1)

	args.batch_size     = 64
	args.epoch_per_test = 10
	
	lib_mean, lib_var   = calculate_log_library_size( adata.X )
	lib_mean1, lib_var1 = calculate_log_library_size( adata1.X )

	Nsample, Nfeature   = np.shape( adata.X )
	Nsample1, Nfeature1 = np.shape( adata1.X )

	device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
	
	encoder_1_list = [ [ Nfeature, 1024, 128, 128 ] ]
	encoder_2_list = [ [ Nfeature1, 1024, 128, 128 ] ]

	share_e_list   = [ [256, 128] ]
	hidden_list    = [ 128 ]
	zdim_list      = [ 22 ]
	share_d_list   = [ [22, 128, 256] ]
	laste_hidden_list = [ 128 ]

	decoder_1_list = [ [128, 128, 1024] ]
	hidden1_list   = [ 1024 ]
	decoder_2_list = [ [128, 128, 1024] ]
	hidden2_list   = [ 1024 ]

	mode_l         = [ 4 ]

	Type1_list     = [ "ZINB" ]
	Type_list      = "ZINB"
	
	file_fla = 10
	
	l_rate         = [ 0.001 ]
	final_rate     = [ 0.0001 ]
	drop_rate_l    = [ 0.1 ]
	drop_rate_dl   = [ 0.1 ]
	anneal_epco_l  = [ 200 ]
	scale_factor_l = [  4 ]

	test_like_max_list   = []
	train_like_max_list  = []
	reco_epoch_test_list = []

	encoder_1_save = []
	encoder_2_save = []
	share_e_save   = []
	hidden_save    = []
	zdim_save      = []
	share_d_save   = []
	decoder_1_save = []
	hidden1_save   = []
	decoder_2_save = []
	hidden2_save   = [] 
	file_fla_save  = []

	drop_save = []
	lar_save  = []

	ARI_list1  = []
	ARI_list2  = []
	ARI_list3  = []

	cohen_list1  = []
	cohen_list2  = []
	cohen_list3  = []
	type_saves   = []
	type1_saves  = []
	anneal_epoch_save = []
	type_mode  = []
	drop_rate_dl_l  = []
	scale_list_l = []
	
	for kd in range( len(drop_rate_l) ):

		for scal in range( len(scale_factor_l) ):

			for kl in range( len(l_rate) ): 

				for kt in range( len(anneal_epco_l) ): 

					for k in range( len(encoder_1_list) ): 

							file_fla = file_fla + 1
							
							args.lr = l_rate[kl]
							args.anneal_epoch = anneal_epco_l[kt]

							file_fla_save.append(file_fla)
							print( str(file_fla) + "  "+ str(l_rate[kl]) + "  " + str( drop_rate_l[kd] )+ "  " + str( drop_rate_dl[kd] ) )

							encoder_1_save.append( '-'.join( map(str, encoder_1_list[k]  ) ))
							encoder_2_save.append( '-'.join( map(str, encoder_2_list[k]) )) 
							share_e_save.append( '-'.join( map(str, share_e_list[k]) ) )
							hidden_save.append( hidden_list[k] )
							zdim_save.append( zdim_list[k] )
							share_d_save.append( '-'.join( map(str, share_d_list[k]) ) )
							decoder_1_save.append( '-'.join( map(str, decoder_1_list[k]) ) )
							hidden1_save.append( hidden1_list[k] ) 
							decoder_2_save.append( '-'.join( map(str, decoder_2_list[k]) ) )
							hidden2_save.append( hidden2_list[k] ) 

							drop_save.append( drop_rate_l[kd] )
							lar_save.append( l_rate[kl] )
							anneal_epoch_save.append( anneal_epco_l[kt] )
							drop_rate_dl_l.append( drop_rate_dl[kd] )
							scale_list_l.append( scale_factor_l[scal] )

							model =    MVAE ( encoder_1 = encoder_1_list[k], encoder_2 = encoder_2_list[k],
											  share_e = share_e_list[k], hidden = hidden_list[k], zdim = zdim_list[k], 
											  share_d = share_d_list[k], decoder_1 = decoder_1_list[k], hidden1 = hidden1_list[k], 
											  decoder_2 = decoder_2_list[k], hidden2 = hidden2_list[k], laste_hidden = laste_hidden_list[k],
											  encoder_l = [ Nfeature, 128 ], hidden_l = 128, encoder_l1 = [ Nfeature1, 128 ], hidden_l1 = 128, 
											  logvariantional = True, drop_rate = drop_rate_l[kd], drop_rate_d = drop_rate_dl[kd], 
											  Type1 = "ZINB", Type = "ZINB", pair = False, mode = mode_l[k], library_mode = 0,
											  n_centroids = 22, penality = "GMM" )

							model.to(device)

							if Type_list == "Bernoulli":
								infer_data = adata2 

							else:
								infer_data = adata1

							test_like_max, epoch, cohen_score1, cohen_score2, cohen_score3, ARI_1, ARI_2, ARI_3 = train( args, adata, 
														   infer_data, model, train_index, test_index, lib_mean, lib_var, lib_mean1, lib_var1, 
														   adata.obs['Group'], final_rate[kl], file_fla, "ZINB", "ZINB", device,
														   scale_factor = scale_factor_l[scal] )

							test_like_max_list.append(test_like_max)
							reco_epoch_test_list.append( epoch )

							ARI_list1.append( ARI_1 )
							ARI_list2.append( ARI_2 )
							ARI_list3.append( ARI_3 )

							cohen_list1.append( cohen_score1 )
							cohen_list2.append( cohen_score2 )
							cohen_list3.append( cohen_score3 )
							type_saves.append("ZINB")
							type1_saves.append("ZINB")
							type_mode.append( mode_l[k] )

	data_three_save  = { "file_labbel": file_fla_save, "encoder_1": encoder_1_save, "encoder_2": encoder_2_save, 
						 "share_e": share_e_save, "hidden": hidden_save, "zdim": zdim_save, "share_d": share_d_save,
						 "decoder_1": decoder_1_save, "hidden1": hidden1_save, "decoder_2": decoder_2_save,
						 "hidden2": hidden2_save, "Type1_list": type1_saves, "Type_list": type_saves, "drop_rate": drop_save, "drop_rate_dl_l": drop_rate_dl_l,
						 "l_rate": lar_save , "scale_list": scale_list_l, "type_mode": type_mode, "anneal_epoch_save": anneal_epoch_save, 
						 "test_like_max_list": test_like_max_list, "reco_epoch_test_list": reco_epoch_test_list , "cohen_score1": cohen_list1, 
						 "cohen_score2": cohen_list2, "cohen_score3": cohen_list3, "ARI_list1": ARI_list1, "ARI_list2": ARI_list2, "ARI_list3": ARI_list3 }

	data_three_save1 = pd.DataFrame(data_three_save).to_csv(os.path.join( args.outdir, 'MVAE_Cellline_adjust_parameters_ZINB_final_10.csv' ))

if __name__ == "__main__":

	parser = parameter_setting()
	args   = parser.parse_args()

	args.workdir  =  '/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Real/SNARE-seq/AdBrainCortex/lap_combine/POE_3000/'
	args.outdir   =  '/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Real/SNARE-seq/AdBrainCortex/lap_combine/POE_3000/MVAE/'
	args.File1    =  'Gene_order_99_3000.tsv'
	args.File2    =  'Gene_order_95_3000_atac.tsv'
	#args.File3   =  '5-cellinfo-RNA.tsv'
	args.File2_1  =  'Gene_order_95_3000_atac_binary.tsv'

	train_with_argas(args)
	