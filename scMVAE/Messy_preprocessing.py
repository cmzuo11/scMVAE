from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import pandas as pd
import os
import os.path
from os import path
import sys
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.cluster import normalized_mutual_info_score

def eps2C(x, c=10000):
	return np.where(x>=1e-12, x, c*np.ones(shape=x.shape))

def cosine(data):
	print('Start for cosine similarity ')
	cosine = 1 - pairwise_distances(data, metric='cosine')
	return np.where((cosine<=1) & (cosine>=0), cosine, np.zeros(cosine.shape))

def euclidean(data):
	print('Start for euclidean similarity ')
	dist = 1 - pairwise_distances(data, metric='euclidean')
	return np.where((dist<=1)&(dist>=0), dist, np.zeros(dist.shape))

def pearson(data):
	print('Start for pearson similarity ')
	df = pd.DataFrame(data.T)
	pear_ = df.corr(method='pearson')
	return np.where(pear_>=0, pear_, np.zeros(shape=(pear_.shape)))

def spearman(data):
	print('Start for spearman similarity ')
	df = pd.DataFrame(data.T)
	spear_ = df.corr(method='spearman')
	return np.where(spear_>=0, spear_, np.zeros(shape=(spear_.shape)))

def all_similarities(data):
	return pearson(data), spearman(data), cosine(data), euclidean(data)

def l_gene_select( pear_score, spear_score, cos_score):
	score_set = [ pear_score, spear_score, cos_score]
	gene_inter, gene_inter_num = [], []

	for i in range(3):
		score, sort_ind = np.sort(score_set[i]), np.argsort(score_set[i])
		score, sort_ind = score[::-1], sort_ind[::-1]
		gene_num = len(score_set[i])
		thresh1 = int(np.round(0.1 * gene_num))
		thresh2 = int(np.round(0.5 * gene_num))

		gene_var = np.zeros((thresh2+1,))
		for j in np.arange(thresh1, thresh2+1):
			score1 = score[:j]
			score2 = score[j:]
			var1 = score1.var()
			var2 = score2.var()
			gene_var[j] = var1 + var2

		gene_var[:thresh1] = np.inf
		select_index = np.argmin(gene_var)
		gene_inter.append(sort_ind[:select_index])
		gene_inter_num.append(select_index)

	gene_select = list( set(gene_inter[0]) & set(gene_inter[1]) & set(gene_inter[2]) )
	return gene_select, gene_inter, gene_inter_num

def LaplacianScore(x, w):
	# x in (samples, features)
	n_samples, n_feat = x.shape[0], x.shape[1]

	if w.shape[0] != n_samples:
		raise Exception("W.shape not match X.shape")

	D = np.diag(np.sum(w, axis=1)) # (n_samples,)
	D2 = np.sum(w, axis=1) # (n_samples,)
	L = w

	tmp1 = (D2.T).dot(x)
	DPrime = np.sum((x.T.dot(D)).T * x, axis=0) - tmp1 * tmp1/np.sum(D2)
	LPrime = np.sum((x.T.dot(L)).T * x, axis=0) - tmp1 * tmp1/np.sum(D2)

	DPrime = eps2C(DPrime, c=10000)
	a1=np.sum(D)
	a2=np.sum((x.T.dot(D)).T * x, axis=0)
	a3=tmp1 * tmp1/np.sum(D)
	a4=(x.T.dot(D)).T * x
	a7=((x.T).dot(D)).T * x
	a5=tmp1 * tmp1
	a6=x.T.dot(D)
	a9=np.dot(x.T,D)

	Y = LPrime / DPrime
	#Y = Y.T#lzl edit
	return Y


def Gene_selection( path, Data , GeneNames, labels):

	### here, Data is sample * feature
	print('Start for similarity pearson')

	pear_sim0     = pearson(Data)
	pear_score    = LaplacianScore(Data, pear_sim0)

	pear_sim0_1 = pd.DataFrame( pear_sim0 ).to_csv( os.path.join( path, 
													labels + '_pearson_cor.csv') )

	pear_score_2 = pd.DataFrame( pear_score , index= GeneNames).to_csv( os.path.join( path, 
													 labels + '_pearson_laplacianScore.csv') )

	print('Start for similarity pearson')

	spear_sim0    = spearman(Data)
	spear_score   = LaplacianScore(Data, spear_sim0)

	spear_sim0_1  = pd.DataFrame( spear_sim0  ).to_csv( os.path.join( path, 
														labels + '_spearman_cor.csv') )

	spear_score_2 = pd.DataFrame( spear_score , index= GeneNames ).to_csv( os.path.join( path, 
													   labels + '_spearman_laplacianScore.csv') )

	print('Start for similarity cosine')

	cos_sim0    = cosine(Data)
	cos_score   = LaplacianScore(Data, cos_sim0)

	cos_sim0_1  = pd.DataFrame( cos_sim0 ).to_csv( os.path.join( path, 
												   labels + '_cosine_cor.csv') )

	cos_score_2 = pd.DataFrame( cos_score, index= GeneNames ).to_csv( os.path.join( path, 
													 labels + '_cosine_laplacianScore.csv') )

	#print('Start for l_gene_select')

	#gene_select, _ , _ = l_gene_select(pear_score, spear_score, cos_score)
	#gene_select1       = np.sort(gene_select)	
	#data_select        = Data[:, gene_select1]

	#data_select_1      = pd.DataFrame( data_select ).to_csv( os.path.join( path, 
	#												 'high_information_'+ labels + '.csv') )

if __name__ == "__main__":


