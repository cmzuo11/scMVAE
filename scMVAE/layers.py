# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:39:39 2019
@author: chunmanzuo
"""

import torch
import torch.nn as nn
import collections
from torch.nn import functional as F
from torch.autograd import Variable


def build_multi_layers(layers, use_batch_norm=True, dropout_rate = 0.1 ):
    """Build multilayer linear perceptron"""
    if dropout_rate > 0:
        fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in, n_out),
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.Dropout(p=dropout_rate),
                        ),
                    )

                    for i, (n_in, n_out) in enumerate( zip(layers[:-1], layers[1:] ) )
                ]
            )
        )

    else:
        fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in, n_out),
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                        ),
                    )

                    for i, (n_in, n_out) in enumerate( zip(layers[:-1], layers[1:] ) )
                ]
            )
        )
        
        
    
    return fc_layers


class Encoder(nn.Module):
    ## for one modulity
    def __init__(self, layer, hidden, Z_DIMS, dropout_rate = 0.1):
        super(Encoder, self).__init__()
        
        if len(layer) > 1:
            self.fc1   =  build_multi_layers( layers = layer, dropout_rate =dropout_rate )
            
        self.layer = layer
        self.fc_means  =  nn.Linear(hidden, Z_DIMS)
        self.fc_logvar =  nn.Linear(hidden, Z_DIMS)
        
    def reparametrize(self, means, logvar):

        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(means)
        else:
          return means
        
    def forward(self, x):
        
        if len(self.layer) > 1:
            h = self.fc1(x)
        else:
            h = x
        mean_x = self.fc_means(h)
        logvar_x = self.fc_logvar(h)
        latent = self.reparametrize(mean_x, logvar_x)
        
        return mean_x, logvar_x, latent

class Decoder_ZINB(nn.Module):
    ### for scRNA-seq
    
    def __init__(self, layer, hidden, input_size, dropout_rate = 0.1):
        
        super(Decoder_ZINB, self).__init__()
        
        if len(layer) >1 :
            self.decoder =  build_multi_layers( layer, dropout_rate = dropout_rate)

        self.decoder_scale = nn.Linear(hidden, input_size)
        self.decoder_r = nn.Linear(hidden, input_size)
        self.dropout = nn.Linear(hidden, input_size)
        
        self.layer = layer

    def forward(self, z, library):
        
        if len(self.layer) > 1:
            latent = self.decoder(z)
        else:
            latent = z
        
        normalized_x = F.softmax( self.decoder_scale( latent ), dim = 1 ) ## mean gamma

        recon_final = torch.exp(library) * normalized_x ##mu
        disper_x = self.decoder_r( latent )  ### theta
        disper_x = torch.exp( disper_x )
        dropout_rate = self.dropout(latent)
        
        return dict( normalized      =  normalized_x,
                     disperation     =  disper_x,
                     imputation      =  recon_final,
                     dropoutrate     =  dropout_rate
                   )

class Decoder_NB(nn.Module):
    
    ### for scRNA-seq
    
    def __init__(self, layer, hidden, input_size):
        
        super(Decoder_NB, self).__init__()

        self.decoder =  build_multi_layers( layers = layer )
        
        self.decoder_scale = nn.Linear(hidden, input_size)
        self.decoder_r = nn.Linear(hidden, input_size)

    def forward(self, z, library):
        
        latent = self.decoder(z)
        
        normalized_x = F.softmax( self.decoder_scale( latent ), dim = 1 ) ## mean gamma

        recon_final = torch.exp(library) * normalized_x ##mu
        disper_x = self.decoder_r( latent )  ### theta
        disper_x = torch.exp( disper_x )
        
        return dict( normalized      =  normalized_x,
                     disperation     =  disper_x,
                     imputation      =  recon_final
                   )

class Decoder(nn.Module):
    ### for scATAC-seq
    def __init__(self, layer, hidden, input_size, Type = "Bernoulli" , dropout_rate = 0.1 ):
        super(Decoder, self).__init__()
        
        if len(layer) >1 :
            self.decoder   =  build_multi_layers( layer, dropout_rate = dropout_rate )
        
        self.decoder_x = nn.Linear( hidden, input_size )
        self.Type      = Type
        self.layer     = layer

    def forward(self, z):
        
        if len(self.layer) >1 :
            latent  = self.decoder( z )
        else:
            latent = z
            
        recon_x = self.decoder_x( latent )
        
        if self.Type == "Bernoulli":
            Final_x = torch.sigmoid(recon_x)

        elif self.Type == "Gaussian":
            Final_x = F.softmax( recon_x, dim = 1 )

        elif self.Type == "Gaussian1":
            Final_x = torch.sigmoid(recon_x)
            
        else:
            Final_x = F.relu(recon_x)
        
        return Final_x
