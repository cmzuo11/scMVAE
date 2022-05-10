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

from scMVAE.utilities import read_dataset, normalize, calculate_log_library_size, parameter_setting, save_checkpoint, load_checkpoint, adjust_learning_rate
from scMVAE.MVAE_model import scMVAE_Concat, scMVAE_NN, scMVAE_POE


def train(args, adata, adata1, model, train_index, test_index, lib_mean, lib_var, lib_mean1, lib_var1, real_groups, 
	      final_rate, file_fla, Type1, Type, device, scale_factor ):

	train         = data_utils.TensorDataset( torch.from_numpy( adata.raw[train_index].X ),
											  torch.from_numpy( lib_mean[train_index] ), 
											  torch.from_numpy( lib_var[train_index] ),
											  torch.from_numpy( lib_mean1[train_index] ), 
											  torch.from_numpy( lib_var1[train_index] ),
											  torch.from_numpy( adata1.raw[train_index].X ))
	train_loader  = data_utils.DataLoader( train, batch_size = args.batch_size, shuffle = True )

	test          = data_utils.TensorDataset( torch.from_numpy( adata.raw[test_index].X ),
											  torch.from_numpy( lib_mean[test_index] ), 
											  torch.from_numpy( lib_var[test_index] ),
											  torch.from_numpy( lib_mean1[test_index] ), 
											  torch.from_numpy( lib_var1[test_index] ),
											  torch.from_numpy( adata1.raw[test_index].X ))
	test_loader   = data_utils.DataLoader( test, batch_size = len(test_index), shuffle = False )

	
	total         = data_utils.TensorDataset( torch.from_numpy( adata.raw.X  ),
											  torch.from_numpy( adata1.raw.X ))
	total_loader  = data_utils.DataLoader( total, batch_size = args.batch_size , shuffle = False )
	
	args.max_epoch   = 500
	train_loss_list  = []

	flag_break       = 0
	epoch_count      = 0
	reco_epoch_test  = 0
	test_like_max    = 100000
	status = ""

	max_iteration = 10000
	args.epoch_per_test = 10

	params    = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = optim.Adam( params, lr = args.lr, weight_decay = args.weight_decay, eps = args.eps )

	epoch     = 0
	iteration = 0
	start     = time.time()

	model.init_gmm_params( total_loader )

	with trange( args.max_epoch, disable=True ) as pbar:

		while True:

			model.train()

			epoch +=  1
			epoch_lr = adjust_learning_rate( args.lr, optimizer, epoch, final_rate, 10 )
			kl_weight = min( 1, epoch / args.anneal_epoch )

			for batch_idx, ( X1, lib_m, lib_v, lib_m1, lib_v1, X2 ) in enumerate(train_loader):

				X1, X2         = X1.float().to(device), X2.float().to(device)
				lib_m,lib_v    = lib_m.to(device),      lib_v.to(device)
				lib_m1, lib_v1 = lib_m1.to(device),     lib_v1.to(device)

				X1, X2         = Variable( X1 ),    Variable( X2 )
				lib_m, lib_v   = Variable( lib_m ), Variable( lib_v )
				lib_m1, lib_v1 = Variable( lib_m1 ),Variable( lib_v1 )

				optimizer.zero_grad()

				loss1, loss2, kl_divergence_l, kl_divergence_l1, kl_divergence_z = model( X1.float(), X2.float(), lib_m, lib_v, lib_m1, lib_v1 )
				loss = torch.mean( ( scale_factor * loss1 + loss2 + kl_divergence_l + kl_divergence_l1) + (kl_weight*(kl_divergence_z)) )  

				loss.backward()
				optimizer.step()

				iteration += 1 

			epoch_count += 1

			if epoch % args.epoch_per_test == 0 and epoch > 0: 

				model.eval()

				with torch.no_grad():

					for batch_idx, ( X1, lib_m, lib_v, lib_m1, lib_v1, X2 ) in enumerate(test_loader): 

						X1, X2         = X1.float().to(device), X2.float().to(device)
						lib_v, lib_m   = lib_v.to(device),      lib_m.to(device)
						lib_v1, lib_m1 = lib_v1.to(device),     lib_m1.to(device)

						X1, X2         = Variable( X1 ),     Variable( X2 )
						lib_m, lib_v   = Variable( lib_m ),  Variable( lib_v )
						lib_m1, lib_v1 = Variable( lib_m1 ), Variable( lib_v1 )
				

						loss1, loss2, kl_divergence_l, kl_divergence_l1, kl_divergence_z = model( X1.float(), X2.float(), lib_m, lib_v, lib_m1, lib_v1 )
						test_loss = torch.mean( ( scale_factor * loss1 + loss2 + kl_divergence_l + kl_divergence_l1) + (kl_weight*(kl_divergence_z)) )  

						train_loss_list.append( test_loss.item() )

						if math.isnan(test_loss.item()):
							flag_break = 1
							break

						if test_like_max >  test_loss.item():
							test_like_max   = test_loss.item()
							epoch_count  = 0

							save_checkpoint(model)

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

	load_checkpoint( './saved_model/model_best.pth.tar', model, device)

	latent_z, recon_x1, norm_x1, recon_x_2, norm_x2 = model.Denoise_batch(total_loader)

	if latent_z is not None:
		imputed_val  = pd.DataFrame( latent_z, index= adata.obs_names ).to_csv( os.path.join( args.outdir, 
									 str(file_fla) + '_latent_ZINB_final.csv' ) ) 
	if norm_x1 is not None:
		norm_x1_1    = pd.DataFrame( norm_x1, columns =  adata.var_names, 
									 index= adata.obs_names ).to_csv( os.path.join( args.outdir,
									 str(file_fla) + '_scRNA_norm_ZINB_final.csv' ) )
	if norm_x2 is not None:
		norm_x2_1   = pd.DataFrame( norm_x2, columns =  adata1.var_names, 
									index= adata1.obs_names ).to_csv( os.path.join( args.outdir, 
									str(file_fla)+ '_scATAC_norm_ZINB_final.csv') )

def train_with_argas( args ):

	args.workdir  =  '/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Real/SNARE-seq/AdBrainCortex/lap_combine/POE_3000/'
	args.outdir   =  '/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Real/SNARE-seq/AdBrainCortex/lap_combine/POE_3000/MVAE/'
	args.File1    =  'Gene_order_99_3000.tsv'
	args.File2    =  'Gene_order_95_3000_atac.tsv'
	args.File2_1  =  'Gene_order_95_3000_atac_binary.tsv'

	adata, adata1, adata2, train_index, test_index,_ = read_dataset( File1 = os.path.join( args.workdir, args.File1 ),
																     File2 = os.path.join( args.workdir, args.File2 ),  
																     File3 = None,
																     File4 = os.path.join( args.workdir, args.File2_1 ),
																     test_size_prop = 0.1
																    )

	adata  = normalize( adata,  size_factors = False, 
						normalize_input = False,  logtrans_input = True ) 

	adata1 = normalize( adata1, size_factors = False, 
						normalize_input = False, logtrans_input = True )

	args.batch_size     = 64
	args.epoch_per_test = 10
	
	lib_mean, lib_var   = calculate_log_library_size( adata.X )
	lib_mean1, lib_var1 = calculate_log_library_size( adata1.X )

	Nsample, Nfeature   = np.shape( adata.X )
	Nsample1, Nfeature1 = np.shape( adata1.X )

	device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
	
	model  = scMVAE_POE ( encoder_1       = [Nfeature, 1024, 128, 128],
		                  hidden_1        = 128, 
		                  Z_DIMS          = 22, 
		                  decoder_share   = [22, 128, 256],
		                  share_hidden    = 128, 
		                  decoder_1       = [128, 128, 1024], 
		                  hidden_2        = 1024, 
		                  encoder_l       = [ Nfeature, 128 ],
		                  hidden3         = 128, 
		                  encoder_2       = [Nfeature1, 1024, 128, 128], 
		                  hidden_4        = 128,
		                  encoder_l1      = [Nfeature1, 128], 
		                  hidden3_1       = 128, 
		                  decoder_2       = [128, 128, 1024],
		                  hidden_5        = 1024, 
		                  drop_rate       = 0.1, 
		                  log_variational = True,
			          Type            = "ZINB", 
			          device          = device, 
				  n_centroids     = 22, 
				  penality        = "GMM",
				  model           = 1,  )

	args.lr           = 0.001
	args.anneal_epoch = 200

	model.to(device)
	infer_data = adata1

	train( args, adata, infer_data, model, train_index, test_index, lib_mean, lib_var, 
		   lib_mean1, lib_var1, adata.obs['Group'], 0.0001, 1, "ZINB", "ZINB", device, 
		   scale_factor = 4 )


if __name__ == "__main__":

	parser = parameter_setting()
	args   = parser.parse_args()

	train_with_argas(args)
	
