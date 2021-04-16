# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:27:03 2019
@author: chunmanzuo
"""

import torch
from torch.nn import functional as F
import numpy as np
import math
import matplotlib

def binary_cross_entropy(recon_x, x):
	return - torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=1)

def log_zinb_positive(x, mu, theta, pi, eps=1e-8):

	x = x.float()

	if theta.ndimension() == 1:

		theta = theta.view( 1, theta.size(0) ) 

	softplus_pi = F.softplus(-pi)

	log_theta_eps = torch.log( theta + eps )

	log_theta_mu_eps = torch.log( theta + mu + eps )

	pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

	case_zero = F.softplus(pi_theta_log) - softplus_pi
	mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

	case_non_zero = ( -softplus_pi + pi_theta_log
		+ x * ( torch.log(mu + eps) - log_theta_mu_eps )
		+ torch.lgamma(x + theta)
		- torch.lgamma(theta)
		- torch.lgamma(x + 1) )

	mul_case_non_zero = torch.mul( (x > eps).type(torch.float32), case_non_zero )

	res = mul_case_zero + mul_case_non_zero

	return - torch.sum( res, dim = 1 )

def NB_loss( y_true, y_pred, theta , eps = 1e-10 ):

	y_true = y_true.float()
	y_pred = y_pred.float()

	t1 = torch.lgamma( theta + eps ) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
	t2 = (theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))

	final = t1 + t2
	
	return - torch.sum( final, dim = 1 )

def mse_loss(y_true, y_pred):

	mask = torch.sign(y_true)

	y_pred = y_pred.float()
	y_true = y_true.float()

	ret = torch.pow( (y_pred - y_true) * mask , 2)

	return torch.sum( ret, dim = 1 )

def poisson_loss(y_true, y_pred):

	y_pred = y_pred.float()
	y_true = y_true.float()
	
	ret = y_pred - y_true * torch.log(y_pred+1e-10) + torch.lgamma(y_true+1.0)

	return  torch.sum( ret, dim=1 )


def GMM_loss(gamma, c_params, z_params):
	"""
	L elbo(x) = Eq(z,c|x)[ log p(x|z) ] - KL(q(z,c|x)||p(z,c))
			  = Eq(z,c|x)[ log p(x|z) + log p(z|c) + log p(c) - log q(z|x) - log q(c|x) ]
	"""
	mu_c, var_c, pi = c_params; #print(mu_c.size(), var_c.size(), pi.size())
	n_centroids = pi.size(1)
	mu, logvar = z_params
	mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_centroids)
	logvar_expand = logvar.unsqueeze(2).expand(logvar.size(0), logvar.size(1), n_centroids)


	# log p(z|c)
	logpzc = -0.5*torch.sum(gamma*torch.sum(math.log(2*math.pi) + \
										   torch.log(var_c) + \
										   torch.exp(logvar_expand)/var_c + \
										   (mu_expand-mu_c)**2/var_c, dim=1), dim=1)
	# log p(c)
	logpc = torch.sum(gamma*torch.log(pi), 1)

	# log q(z|x) or q entropy    
	qentropy = -0.5*torch.sum(1+logvar+math.log(2*math.pi), 1)

	# log q(c|x)
	logqcx = torch.sum(gamma*torch.log(gamma), 1)

	kld = -logpzc - logpc + qentropy + logqcx

	return  kld
