# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:22:26 2019
@author: chunmanzuo
"""

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.autograd import Variable
from torch.distributions import Normal, kl_divergence as kl
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
from layers import build_multi_layers
from layers import Encoder, Decoder_ZINB, Decoder
from loss_function import log_zinb_positive, binary_cross_entropy, mse_loss, poisson_loss, GMM_loss

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class scMVAE_Concat(nn.Module):
	def __init__( self, layer_e, hidden1, Zdim, layer_l, hidden3, layer_d, 
				  hidden4, logchange = True, Type = 'ZINB', n_centroids = 4, 
				  penality = "GMM" ):

		super(scMVAE_Concat, self).__init__()
		### function definition
		self.encoder_x = Encoder( layer_e, hidden1, Zdim )
		self.encoder_l = Encoder( layer_l, hidden3, 1 )
		
		if Type == 'ZINB':
			self.decoder_x = Decoder_ZINB( layer_d, hidden4, layer_e[0] )

		else:
			self.decoder_x = Decoder( layer_d, hidden4, layer_e[0], Type )
		
		### parameters definition
		self.logchange   =  logchange
		self.Type        =  Type
		self.penality    =  penality
		self.n_centroids =  n_centroids

		self.pi    = nn.Parameter(torch.ones(n_centroids)/n_centroids)  # pc
		self.mu_c  = nn.Parameter(torch.zeros(Zdim, n_centroids)) # mu
		self.var_c = nn.Parameter(torch.ones(Zdim, n_centroids)) # sigma^2

	def reparametrize(self, means, logvar):

		if self.training:

			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(means)
		else:
		  return means
	
	def inference(self, X):
		
		X_ = X
		if self.logchange:
			X_ = torch.log( X_ + 1 )
			
		### latent encoder for x
		mean_z, logvar_z, latent_z = self.encoder_x( X_ )
		mean_l, logvar_l, library  = self.encoder_l( X_ ) ### library scale factor
		
		### decoder for x latent
		if self.Type == 'ZINB' :
			output        =  self.decoder_x( latent_z, library )
			normalized_x  =  output["normalized"]
			disper_x      =  output["disperation"]
			dropout_rate  =  output["dropoutrate"]
			recon_final   =  output["imputation"]
		else:
			recons_x      =  self.decoder_x( latent_z )
			recon_final   =  recons_x
			normalized_x  =  None
			disper_x      =  None
			dropout_rate  =  None
		
		return dict( latent_z_mu     =  mean_z,
					 latent_z_logvar =  logvar_z,
					 latent_z        =  latent_z,
					 latent_l_mu     =  mean_l,
					 latent_l_logvar =  logvar_l,
					 normalized      =  normalized_x,
					 disperation     =  disper_x,
					 imputation      =  recon_final,
					 dropoutrate     =  dropout_rate
				   )
	
	def get_reconstruction_loss( self, x, px_rate, px_r, px_dropout ):
		
		if self.Type == 'ZINB':
			loss = log_zinb_positive(x, px_rate, px_r, px_dropout)
			
		elif self.Type == 'Bernoulli':
			loss = binary_cross_entropy( x, px_rate )
			
		else:
			loss = mse_loss( x, px_rate )
			
		return loss
		
	def forward( self, X, local_l_mean = None, local_l_var = None ):
		
		result = self.inference(X)
		
		latent_z_mu     = result["latent_z_mu"]
		latent_z_logvar = result["latent_z_logvar"]
		latent_z        = result["latent_z"]

		latent_l_mu     = result["latent_l_mu"]
		latent_l_logvar = result["latent_l_logvar"]
		
		imputation  = result["imputation"]
		disperation = result["disperation"]
		dropoutrate = result["dropoutrate"]

		# KL Divergence for library factor
		if local_l_mean is not None:
			kl_divergence_l =  kl( Normal( latent_l_mu, latent_l_logvar ),
								   Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)
		else:
			kl_divergence_l = torch.tensor(0.0)

		# KL Divergence for latent code
		if self.penality == "GMM":
			gamma, mu_c, var_c, pi =  self.get_gamma(latent_z) #, self.n_centroids, c_params)
			kl_divergence_z        =  GMM_loss( gamma, (mu_c, var_c, pi), (latent_z_mu, latent_z_logvar) )

		else:
			mean             = torch.zeros_like(latent_z_mu)
			scale            = torch.ones_like(latent_z_logvar)
			kl_divergence_z  =  kl( Normal(latent_z_mu, latent_z_logvar), 
									Normal(mean, scale)).sum(dim=1)
		
		reconst_loss  =  self.get_reconstruction_loss( X, imputation, disperation, dropoutrate )
		
		return reconst_loss, kl_divergence_l, kl_divergence_z


	def get_gamma(self, z):
		
		n_centroids = self.n_centroids

		N     =  z.size(0)
		z     =  z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
		pi    =  self.pi.repeat(N,1) # NxK
		mu_c  =  self.mu_c.repeat(N,1,1) # NxDxK
		var_c =  self.var_c.repeat(N,1,1) # NxDxK

		# p(c,z) = p(c)*p(z|c) as p_c_z
		p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*var_c) + (z-mu_c)**2/(2*var_c), dim=1)) + 1e-10
		gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

		return gamma, mu_c, var_c, pi

	def out_Batch(self, Dataloader, device = 'cpu', out='Z', transforms=None):
		output = []

		for i, (X,_) in enumerate(Dataloader):

			X = X.view(X.size(0), -1).float().to(device)
			result = self.inference( X )

			if out == 'Z':
				output.append(result["latent_z"].detach().cpu())
			elif out == 'imputation':
				output.append(result["imputation"].detach().cpu().data)
			elif out == 'normalized':
				output.append(result["normalized"].detach().cpu().data)

			elif out == 'logit':
				output.append(self.get_gamma(z)[0].cpu().detach())

		output = torch.cat(output).numpy()

		return output

	def init_gmm_params(self, Dataloader, device ):

		gmm      = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
		latent_z = self.out_Batch(Dataloader, device, out='Z' )
		gmm.fit(latent_z)

		self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
		self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))


class scMVAE_NN(nn.Module):
	# scMVAE-NN
	def __init__( self, encoder_1, encoder_2, share_e, hidden, zdim, share_d, 
				 decoder_1, hidden1, decoder_2, hidden2,
				 encoder_l, hidden_l, encoder_l1, hidden_l1, laste_hidden, logvariantional = True, 
				 drop_rate = 0.1, drop_rate_d = 0.1, Type1 = 'ZINB', Type = 'ZINB', pair = False, 
				 mode = 0, library_mode = 0, n_centroids = 19, penality = "GMM" ):

		super(scMVAE_NN, self).__init__()

		self.encoder_x1    =  build_multi_layers( encoder_1, dropout_rate = drop_rate )
		self.encoder_x2    =  build_multi_layers( encoder_2, dropout_rate = drop_rate )

		self.encoder_share =  Encoder( share_e, hidden, zdim , dropout_rate = drop_rate )
		self.decoder_share =  build_multi_layers( share_d, dropout_rate = drop_rate_d )

		
		self.decoder_x1    =  Decoder_ZINB( decoder_1, hidden1, encoder_1[0], dropout_rate = drop_rate_d )
		
		#self.decoder_x1  =  Decoder( decoder_1, hidden1, encoder_1[0], Type1, dropout_rate = drop_rate )

		if library_mode ==0:
			self.encoder_l   =  Encoder( encoder_l, hidden_l, 1 )
		else:
			self.encoder_l   =  Encoder( [128], encoder_1[-1], 1 )

		if Type == "ZINB":
			self.encoder_l2  =  Encoder( encoder_l1, hidden_l1, 1, dropout_rate = drop_rate )
			self.decoder_x2  =  Decoder_ZINB( decoder_2, hidden2, encoder_2[0], dropout_rate = drop_rate_d )
		else:
			self.decoder_x2  =  Decoder( decoder_2, hidden2, encoder_2[0], Type, dropout_rate = drop_rate_d )
		
		###parameters
		self.logvariantional =  logvariantional
		self.hidden          =  laste_hidden
		self.Type            =  Type
		self.Type1           =  Type1
		self.pair            =  pair
		self.mode            =  mode
		self.library_mode    =  library_mode
		self.n_centroids     =  n_centroids
		self.penality        =  penality

		self.pi    = nn.Parameter(torch.ones(n_centroids)/n_centroids)  # pc
		self.mu_c  = nn.Parameter(torch.zeros(zdim, n_centroids)) # mu
		self.var_c = nn.Parameter(torch.ones(zdim, n_centroids)) # sigma^2
	
	def reparametrize(self, means, logvar):

		if self.training:

			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(means)
		else:
		  return means
	  
	def inference( self, X1, X2 ):
		X1_ = X1
		X2_ = X2
		
		if self.logvariantional:
			X1_ = torch.log( X1_ + 1 )
			
			if self.Type == "ZINB":
				X2_ = torch.log( X2_ + 1 )
				mean_l2, logvar_l2, library2  = self.encoder_l2( X2_ )
			else:
				mean_l2   = None
				logvar_l2 = None
				library2  = None
		
		latent_x1 = self.encoder_x1( X1_ )
		latent_x2 = self.encoder_x2( X2_ )

		if self.library_mode ==0: ### for library out
			mean_l, logvar_l, library  = self.encoder_l( X1_ )
		else:
			mean_l, logvar_l, library  = self.encoder_l( latent_x1 )
		
		latent = torch.cat( (latent_x1, latent_x2), 1)
		mean_z, logvar_z, latent_z = self.encoder_share( latent )

		### uodate network
		decoder_z    =  self.decoder_share(latent_z)

		if self.mode == 0: ## normal network
			decoder_z_x1 = decoder_z
			decoder_z_x2 = decoder_z

		elif self.mode == 2: 
			decoder_z_x1 = decoder_z
			decoder_z_x2 = latent_z

		elif self.mode == 3: 
			decoder_z_x1 = torch.cat( ( decoder_z, latent_z ), 1)
			decoder_z_x2 = latent_z
			
		elif self.mode == 1: 
			decoder_z_x1 = torch.cat( ( decoder_z, latent_z ), 1)
			decoder_z_x2 = torch.cat( ( decoder_z, latent_z ), 1)

		elif self.mode == 5: 
			decoder_z_x1 = torch.cat( ( decoder_z[ : , :self.hidden ], latent_z ), 1)
			decoder_z_x2 = torch.cat( ( decoder_z[ : , self.hidden: ], latent_z ), 1)

		elif self.mode == 6: 
			decoder_z_x1 = torch.cat( ( decoder_z[ : , :self.hidden ], latent_z ), 1)
			decoder_z_x2 = decoder_z[ : , self.hidden: ]

		elif self.mode == 7: 
			decoder_z_x1 = torch.cat( ( decoder_z, latent_z ), 1)
			decoder_z_x2 = decoder_z

		else: ### 4
			decoder_z_x1 = decoder_z[ : , :self.hidden ]
			decoder_z_x2 = decoder_z[ : , self.hidden: ]
		
		## library size factor
		recon_output =  self.decoder_x1( decoder_z_x1, library )
		norm_x1      =  recon_output["normalized"]
		disper_x     =  recon_output["disperation"]
		recon_x1     =  recon_output["imputation"]
		dropout_rate =  recon_output["dropoutrate"]

		if self.Type == "ZINB":
			recon_x2       = self.decoder_x2( decoder_z_x2, library2 )
			norm_x2        =  recon_x2["normalized"]
			disper_x2      =  recon_x2["disperation"]
			recon_x_2      =  recon_x2["imputation"]
			dropout_rate_2 =  recon_x2["dropoutrate"]

		else:
			recon_x21      = self.decoder_x2( decoder_z_x2 )
			norm_x2        =  None
			disper_x2      =  None
			recon_x_2      =  recon_x21
			dropout_rate_2 =  None
		
		return dict( norm_x1        =  norm_x1,   disper_x       =  disper_x,
					 recon_x1       =  recon_x1,  dropout_rate   =  dropout_rate,
					 norm_x2        =  norm_x2,   disper_x2      =  disper_x2,
					 recon_x_2      =  recon_x_2, dropout_rate_2 =  dropout_rate_2,
					 mean_l         =  mean_l,    logvar_l       =  logvar_l,
					 library        =  library,   mean_l2        =  mean_l2,
					 logvar_l2      =  logvar_l2, library2       =  library2,
					 mean_z         =  mean_z,    logvar_z       =  logvar_z,
					 latent_z       =  latent_z,
				   )
		
	def forward( self, X1, X2, local_l_mean, local_l_var, local_l_mean1, local_l_var1 ):
		
		result         =  self.inference( X1, X2 )
		disper_x       =  result["disper_x"]
		recon_x1       =  result["recon_x1"]
		dropout_rate   =  result["dropout_rate"]

		disper_x2      =  result["disper_x2"]
		recon_x_2      =  result["recon_x_2"]
		dropout_rate_2 =  result["dropout_rate_2"]
		
		mean_z         =  result["mean_z"]
		logvar_z       =  result["logvar_z"]

		# KL Divergence
		# for X1 library
		mean_l         =  result["mean_l"]
		logvar_l       =  result["logvar_l"]

		kl_divergence_l = kl( Normal(mean_l, logvar_l),
							  Normal(local_l_mean,torch.sqrt(local_l_var))).sum(dim=1)

		# for X2 library
		if self.Type == "ZINB":
			mean_l2          =  result["mean_l2"]
			logvar_l2        =  result["logvar_l2"]
			kl_divergence_l2 =  kl( Normal(mean_l2, logvar_l2),
									Normal(local_l_mean,torch.sqrt(local_l_var))).sum(dim=1)
		else:
			kl_divergence_l2 = torch.tensor(0.0)

		mean_z    =  result["mean_z"]
		logvar_z  =  result["logvar_z"]
		latent_z  =  result["latent_z"]

		if self.penality == "GMM" :
			gamma, mu_c, var_c, pi =  self.get_gamma(latent_z) #, self.n_centroids, c_params)
			kl_divergence_z        =  GMM_loss( gamma, (mu_c, var_c, pi), (mean_z, logvar_z) )

		else:
			mean  = torch.zeros_like(mean_z)
			scale = torch.ones_like(logvar_z)
			kl_divergence_z  = kl( Normal(mean_z, logvar_z), 
								   Normal(mean, scale)).sum(dim=1)

		### maximize the negative likelihood fuction
		loss1, loss2  =  get_both_recon_loss( X1, recon_x1, disper_x, dropout_rate, 
											  X2, recon_x_2, disper_x2, dropout_rate_2, 
											  self.Type1, self.Type )
			
		return loss1, loss2, kl_divergence_l, kl_divergence_l2, kl_divergence_z

	def get_gamma(self, z):
		
		n_centroids = self.n_centroids

		N     =  z.size(0)
		z     =  z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
		pi    =  self.pi.repeat(N,1) # NxK
		mu_c  =  self.mu_c.repeat(N,1,1) # NxDxK
		var_c =  self.var_c.repeat(N,1,1) # NxDxK

		# p(c,z) = p(c)*p(z|c) as p_c_z
		p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*var_c) + (z-mu_c)**2/(2*var_c), dim=1)) + 1e-10
		gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

		return gamma, mu_c, var_c, pi

	def out_Batch(self, Dataloader, device, out='Z', transforms=None):
		output = []

		for i, (X1, X2) in enumerate(Dataloader):

			X1 = X1.view(X1.size(0), -1).float().to(device)
			X2 = X2.view(X2.size(0), -1).float().to(device)

			result = self.inference(X1, X2)

			if out == 'Z':
				output.append(result["latent_z"].detach().cpu())
			elif out == 'recon_X1':
				output.append(result["recon_x1"].detach().cpu().data)
			elif out == 'Norm_X1':
				output.append(result["norm_x1"].detach().cpu().data)
			elif out == 'recon_X2':
				output.append(result["recon_x_2"].detach().cpu().data)

			elif out == 'logit':
				output.append(self.get_gamma(z)[0].cpu().detach())

		output = torch.cat(output).numpy()

		return output

	def init_gmm_params(self, Dataloader, device):

		gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
	   
		latent_z  =  self.out_Batch(Dataloader, device, out='Z' )
		gmm.fit(latent_z)

		self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
		self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))

	def Denoise_batch(self, total_loader):
		# processing large-scale datasets
		latent_z  = []
		norm_x1   = []
		recon_x_2 = []
		recon_x1  = []
		norm_x2   = []

		for batch_idx, ( X1, X2 ) in enumerate(total_loader): 

			X1     = X1.to(self.device)   
			X2     = X2.to(self.device)

			X1     = Variable( X1 )
			X2     = Variable( X2 )

			result = self.inference(X1, X2)

			latent_z.append( result1["latent_z"].data.cpu().numpy() )
			recon_x_2.append( result1["recon_x_2"].data.cpu().numpy() )
			recon_x1.append( result1["recon_x1"].data.cpu().numpy() )
			norm_x1.append( result1["norm_x1"].data.cpu().numpy() )
			norm_x2.append( result1["norm_x2"].data.cpu().numpy() )

		latent_z  = np.concatenate(latent_z)
		recon_x_2 = np.concatenate(recon_x_2)
		recon_x1  = np.concatenate(recon_x1)
		norm_x1   = np.concatenate(norm_x1)
		norm_x2   = np.concatenate(norm_x2)

		return latent_z, recon_x1, norm_x1, recon_x_2, norm_x2
	

class scMVAE_POE(nn.Module):
	## scMVAE-PoE
	
	def __init__(self, encoder_1, hidden_1, Z_DIMS, decoder_share, share_hidden,
				 decoder_1, hidden_2, encoder_l, hidden3, encoder_2, hidden_4, 
				 encoder_l1, hidden3_1, decoder_2, hidden_5, drop_rate,
				 log_variational = True, Type = 'Bernoulli', device = 'cpu', 
				 n_centroids = 19, penality = "GMM", model = 2
				):
		
		super(scMVAE_POE, self).__init__()

		self.X1_encoder    = Encoder( encoder_1, hidden_1, Z_DIMS, dropout_rate = drop_rate) 
		self.X1_encoder_l  = Encoder( encoder_l, hidden3, 1 , dropout_rate = drop_rate)

		self.X1_decoder    = Decoder_ZINB( decoder_1, hidden_2, encoder_1[0], dropout_rate = drop_rate )
		 
		self.X2_encoder    = Encoder( encoder_2, hidden_4, Z_DIMS, dropout_rate = drop_rate )

		self.decode_share  = build_multi_layers( decoder_share, dropout_rate = drop_rate )

		if Type == 'ZINB':
			self.X2_encoder_l =  Encoder( encoder_l1, hidden3_1, 1 , dropout_rate = drop_rate)
			self.decoder_x2   =  Decoder_ZINB( decoder_2, hidden_5, encoder_2[0], 
											   dropout_rate = drop_rate )
		elif Type == 'Bernoulli':
			self.decoder_x2   =  Decoder( decoder_2, hidden_5, encoder_2[0], Type, 
										 dropout_rate = drop_rate )
		elif Type == "Possion":
			self.decoder_x2   =  Decoder( decoder_2, hidden_5, encoder_2[0], Type, 
										  dropout_rate = drop_rate )
		else:
			self.decoder_x2   =  Decoder( decoder_2, hidden_5, encoder_2[0], Type, 
										  dropout_rate = drop_rate )

		self.experts         =  ProductOfExperts()
		self.Z_DIMS          =  Z_DIMS
		self.share_hidden    =  share_hidden
		self.log_variational =  log_variational
		self.Type            =  Type
		self.decoder_share   =  decoder_share
		self.decoder_1       =  decoder_1
		self.n_centroids     =  n_centroids
		self.penality        =  penality
		self.device          =  device
		self.model           =  model

		self.pi    = nn.Parameter(torch.ones(n_centroids)/n_centroids)  # pc
		self.mu_c  = nn.Parameter(torch.zeros(Z_DIMS, n_centroids)) # mu
		self.var_c = nn.Parameter(torch.ones(Z_DIMS, n_centroids)) # sigma^2

	def reparametrize(self, means, logvar):

		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(means)
		else:
		  return means
	
	def encode_modalities(self, X1 = None, X2 = None):

		if X1 is not None:
			batch_size = X1.size(0)
		else:
			batch_size = X2.size(0)

		# Initialization
		means, logvar = prior_expert( (1, batch_size, self.Z_DIMS) )

		means  = means.to(self.device)
		logvar = logvar.to(self.device)
	
		# Support for weak supervision setting
		if X1 is not None:
			X1_mean, X1_logvar, _ = self.X1_encoder(X1)
			means                 = torch.cat((means, X1_mean.unsqueeze(0)), dim=0)
			logvar                = torch.cat((logvar, X1_logvar.unsqueeze(0)), dim=0)

		if X2 is not None:
			X2_mean, X2_logvar, _ = self.X2_encoder(X2)
			means                 = torch.cat((means, X2_mean.unsqueeze(0)), dim=0)
			logvar                = torch.cat((logvar, X2_logvar.unsqueeze(0)), dim=0)

		# Combine the gaussians
		means, logvar = self.experts( means, logvar )
		return means, logvar
	  
	def inference(self, X1 = None, X2 = None):

		X1_ = X1
		X2_ = X2
		### X1 processing
		mean_l, logvar_l, library = None, None, None
		
		if X1 is not None:
			if self.log_variational:
				X1_ = torch.log( X1_ + 1 )

			mean_l, logvar_l, library = self.X1_encoder_l( X1_ )
	  
		### X2 processing
		mean_l2, logvar_l2, library2 = None, None, None

		if X2 is not None:

			if self.Type == 'ZINB':
				if self.log_variational:
					X2_ = torch.log( X2_ + 1 )
					mean_l2, logvar_l2, library2 = self.X2_encoder_l( X2_ )


		means, logvar = self.encode_modalities( X1_, X2_ )

		z = self.reparametrize( means, logvar )

		if len(self.decoder_share) > 1 :
			latents  = self.decode_share(z)

			if self.model == 0:
				latent_1 = latents
				latent_2 = latents

			elif self.model == 1:
				latent_1 = latents[:, :self.share_hidden]
				latent_2 = latents[:, self.share_hidden:]

			elif self.model == 2:
				latent_1 = torch.cat( ( z, latents[:, :self.share_hidden] ), 1)
				latent_2 = latents[:, self.share_hidden:]

			else:
				latent_1 = torch.cat( ( z, latents ), 1)
				latent_2 = latents
		else:
			latent_1 = z
			latent_2 = z

		# Reconstruct
		output       = self.X1_decoder( latent_1, library )
		
		normalized_x = output["normalized"]
		recon_X1     = output["imputation"]
		disper_x     = output["disperation"]
		dropout_rate = output["dropoutrate"]

		if self.Type == 'ZINB':
			results        = self.decoder_x2( latent_2, library2 )

			norma_x2       = results["normalized"]
			recon_X2       = results["imputation"]
			disper_x2      = results["disperation"]
			dropout_rate_2 = results["dropoutrate"]

		else:
			recon_X2       = self.decoder_x2( latent_2 )
			norma_x2, disper_x2, dropout_rate_2 = None, None, None

		return dict( norm_x1        =  normalized_x, disper_x       =  disper_x,
					 recon_x1       =  recon_X1,     dropout_rate   =  dropout_rate,
					 norm_x2        =  norma_x2,     disper_x2      =  disper_x2,
					 recon_x_2      =  recon_X2,     dropout_rate_2 =  dropout_rate_2,
					 mean_l         =  mean_l,       logvar_l       =  logvar_l,
					 library        =  library,      mean_l2        =  mean_l2,
					 logvar_l2      =  logvar_l2,    library2       =  library2,
					 mean_z         =  means,        logvar_z       =  logvar,
					 latent_z       =  z,
				   )
		

	def forward( self, X1, X2, local_l_mean, local_l_var, 
				 local_l_mean1, local_l_var1 ):
		
		result         =  self.inference(X1, X2 )

		disper_x       =  result["disper_x"]
		recon_x1       =  result["recon_x1"]
		dropout_rate   =  result["dropout_rate"]

		disper_x2      =  result["disper_x2"]
		recon_x_2      =  result["recon_x_2"]
		dropout_rate_2 =  result["dropout_rate_2"]

		if X1 is not None:
			mean_l           =  result["mean_l"]
			logvar_l         =  result["logvar_l"]

			kl_divergence_l  = kl( Normal(mean_l, logvar_l),
								   Normal(local_l_mean,torch.sqrt(local_l_var))).sum(dim=1)
		else:
			kl_divergence_l  = torch.tensor(0.0)

		if X2 is not None:
			if self.Type == 'ZINB':
				mean_l2           =  result["mean_l2"]
				logvar_l2         =  result["library2"]
				kl_divergence_l2  = kl( Normal(mean_l2, logvar_l2),
										Normal(local_l_mean,torch.sqrt(local_l_var))).sum(dim=1)
			else:
				kl_divergence_l2 = torch.tensor(0.0)
		else:
			kl_divergence_l2 = torch.tensor(0.0)

		mean_z    =  result["mean_z"]
		logvar_z  =  result["logvar_z"]
		latent_z  = result["latent_z"]

		if self.penality == "GMM" :
			gamma, mu_c, var_c, pi = self.get_gamma(latent_z) #, self.n_centroids, c_params)
			kl_divergence_z = GMM_loss( gamma, (mu_c, var_c, pi), (mean_z, logvar_z) )

		else:
			mean  = torch.zeros_like(mean_z)
			scale = torch.ones_like(logvar_z)
			kl_divergence_z  = kl( Normal(mean_z, logvar_z), 
								   Normal(mean, scale)).sum(dim=1)

		loss1, loss2  =  get_both_recon_loss( X1, recon_x1, disper_x, dropout_rate, 
											  X2, recon_x_2, disper_x2, dropout_rate_2, 
											  "ZINB", self.Type )
			
		return loss1, loss2, kl_divergence_l, kl_divergence_l2, kl_divergence_z

	def out_Batch(self, Dataloader, out='Z', transforms=None):
		output = []

		for i, (X1, X2) in enumerate(Dataloader):

			X1 = X1.view(X1.size(0), -1).float().to(self.device)
			X2 = X2.view(X2.size(0), -1).float().to(self.device)

			result = self.inference(X1, X2)

			if out == 'Z':
				output.append(result["latent_z"].detach().cpu())
			elif out == 'recon_X1':
				output.append(result["recon_x1"].detach().cpu().data)
			elif out == 'Norm_X1':
				output.append(result["norm_x1"].detach().cpu().data)
			elif out == 'recon_X2':
				output.append(result["recon_x_2"].detach().cpu().data)

			elif out == 'logit':
				output.append(self.get_gamma(z)[0].cpu().detach())

		output = torch.cat(output).numpy()

		return output

	def get_gamma(self, z):
		
		n_centroids = self.n_centroids

		N = z.size(0)
		z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
		pi = self.pi.repeat(N,1) # NxK
		mu_c = self.mu_c.repeat(N,1,1) # NxDxK
		var_c = self.var_c.repeat(N,1,1) # NxDxK

		# p(c,z) = p(c)*p(z|c) as p_c_z
		p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*var_c) + (z-mu_c)**2/(2*var_c), dim=1)) + 1e-10
		gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

		return gamma, mu_c, var_c, pi

	def init_gmm_params(self, Dataloader):
		
		gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
		
		latent_z  =  self.out_Batch(Dataloader, out='Z' )
		gmm.fit(latent_z)

		self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
		self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))

	def Denoise_batch(self, total_loader):
		# processing large-scale datasets
		latent_z  = []
		norm_x1   = []
		recon_x_2 = []
		recon_x1  = []
		norm_x2   = []

		for batch_idx, ( X1, X2 ) in enumerate(total_loader): 

			X1     = X1.to(self.device)   
			X2     = X2.to(self.device)

			X1     = Variable( X1 )
			X2     = Variable( X2 )

			result = self.inference(X1, X2)

			latent_z.append( result1["latent_z"].data.cpu().numpy() )
			recon_x_2.append( result1["recon_x_2"].data.cpu().numpy() )
			recon_x1.append( result1["recon_x1"].data.cpu().numpy() )
			norm_x1.append( result1["norm_x1"].data.cpu().numpy() )
			norm_x2.append( result1["norm_x2"].data.cpu().numpy() )

		latent_z  = np.concatenate(latent_z)
		recon_x_2 = np.concatenate(recon_x_2)
		recon_x1  = np.concatenate(recon_x1)
		norm_x1   = np.concatenate(norm_x1)
		norm_x2   = np.concatenate(norm_x2)

		return latent_z, recon_x1, norm_x1, recon_x_2, norm_x2

class ProductOfExperts(nn.Module):
	"""Return parameters for product of independent experts.
	See https://arxiv.org/pdf/1410.7827.pdf for equations.

	@param mu: M x D for M experts
	@param logvar: M x D for M experts
	"""
	def forward(self, mu, logvar, eps=1e-8):
		
		var       = torch.exp(logvar) + eps
		# precision of i-th Gaussian expert at point x
		T         = 1. / var
		pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
		pd_var    = 1. / torch.sum(T, dim=0)
		pd_logvar = torch.log(pd_var)
		return pd_mu, pd_logvar


def prior_expert(size):
	"""Universal prior expert. Here we use a spherical
	Gaussian: N(0, 1).

	@param size: integer
				 dimensionality of Gaussian
	"""
	mu     = Variable(torch.zeros(size))
	logvar = Variable(torch.log(torch.ones(size)))
	
	return mu, logvar

def get_triple_recon_loss(x1 = None, px1_rate = None, px1_r = None, px1_dropout = None, 
						  x2 = None, px2_rate = None, px2_r = None, px2_dropout = None,
						  px12_rate = None, px12_r = None, px12_dropout = None, 
						  Type1 = "ZINB", Type = "Bernoulli" ):

	reconst_loss1  = log_zinb_positive( x1, px1_rate, px1_r, px1_dropout )
	reconst_loss12 = binary_cross_entropy( px12_rate, x2 )
	
	if x2 is not None:
		reconst_loss2  = binary_cross_entropy( px2_rate, x2 )
	else:
		reconst_loss2  = torch.tensor(0.0)
	
	return reconst_loss1, reconst_loss2, reconst_loss12



def get_both_recon_loss( x1 = None, px1_rate = None, px1_r = None, px1_dropout = None, 
						 x2 = None, px2_rate = None, px2_r = None, px2_dropout = None, 
						 Type1 = "ZINB", Type = "Bernoulli" ):

	# Reconstruction Loss
	## here Type1 for rna-seq, Type for atac-seq
	#reconst_loss1 = log_zinb_positive( x1, px1_rate, px1_r, px1_dropout )
	if x1 is not None:
		reconst_loss1 = log_zinb_positive( x1, px1_rate, px1_r, px1_dropout )
	else:
		reconst_loss1 = torch.tensor(0.0)

	if x2 is not None:

		if Type == "ZINB":
			reconst_loss2 = log_zinb_positive( x2, px2_rate, px2_r, px2_dropout )

		elif Type == 'Bernoulli':
			reconst_loss2 = binary_cross_entropy( px2_rate, x2 )

		elif Type == "Possion":
			reconst_loss2 = poisson_loss( x2, px2_rate )

		else:
			reconst_loss2 = mse_loss( x2 , px2_rate  )
	else:
		reconst_loss2 = torch.tensor(0.0)
	
	return reconst_loss1, reconst_loss2
