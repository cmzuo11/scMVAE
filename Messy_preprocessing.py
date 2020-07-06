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

def calculate_NMI_ARI( latent_file, cluster_file, cluster_no, flag = 0 ):

	if flag == 0:

		Data1  = pd.read_csv( latent_file, header=0, index_col=0 )

		if np.shape(Data1.values)[1] == 0 :
			return 0, 0

		kmeans = KMeans( n_clusters = cluster_no, n_init = 5, random_state = 200 )
		pred_z = kmeans.fit_predict( Data1.values )
	else:
		table  = pd.read_table( latent_file, header=0, index_col=0 )
		pred_z = table['Group'].values

	label_ground_truth = []
	
	Data2 = pd.read_csv( cluster_file, header=0, index_col=0 )
	group = Data2['Group'].values

	for g in group:
		g = int(g.split('Group')[1])
		label_ground_truth.append(g)

	NMI_score = round( normalized_mutual_info_score( pred_z, label_ground_truth, average_method='max' ), 3 )
	ARI_score = round( metrics.adjusted_rand_score( label_ground_truth, pred_z ), 3 )   

	#print( 'NMI score: ' + str(NMI_score) )
	#print( 'ARI score: ' + str(ARI_score) )

	return NMI_score, ARI_score

def predict_clustering( latent_file, out_file, cluster_no):

	table = pd.read_csv( latent_file, header=0, index_col=0 )
	
	kmeans = KMeans( n_clusters = cluster_no, n_init = 5, random_state = 200 )
	pred_z = kmeans.fit_predict( table.values )
	
	imputed_val  = pd.DataFrame( pred_z, index= table.index.values ).to_csv( out_file ) 

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

def calculate_pearson_lascore(path, Data , GeneNames, labels):

	### here, Data is sample * feature
	print('Start for similarity pearson')

	pear_sim0     = pearson(Data)
	pear_score    = LaplacianScore(Data, pear_sim0)

	pear_sim0_1 = pd.DataFrame( pear_sim0 ).to_csv( os.path.join( path, 
													labels + '_pearson_cor.csv') )

	pear_score_2 = pd.DataFrame( pear_score , index= GeneNames).to_csv( os.path.join( path, 
													 labels + '_pearson_laplacianScore.csv') )

def Garbage_function():
	### 1. gene selection processing

	workPath = '/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Real/SNARE-seq/P0_BrainCortex/out/high_median/out2/'
	group_str = ( '' )
	included_extensions = ['_latent.csv']

	file_names = [fn for fn in os.listdir( workPath ) if any(fn.endswith(ext) for ext in included_extensions)]

	for temp_f in file_names:

		input1  =  os.path.join( workPath,  temp_f )
		output1 =  os.path.join( workPath,  temp_f + '-cluster')

		#predict_clustering( input1, output1 )

	### calculate ARI and NMI

	workPath = '/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Simulated/Simulated_2/'

	### PCA region
	print("PCA result")

	latent_file  = os.path.join( workPath,  '5_0.75_0.2_PCA_30.csv' )
	cluster_file = os.path.join( workPath,  '5-cellinfo-RNA.tsv' ) 
	calculate_NMI_ARI( latent_file, cluster_file, 2, 0 )

	## MOFA framework
	print("MOFA result")

	latent_file = os.path.join( workPath, 'mofa_out/MOFA_combine_cluster_1.csv')
	cluster_file = os.path.join( workPath,  '5-cellinfo-RNA.tsv' ) 

	calculate_NMI_ARI( latent_file, cluster_file, 2, 1 )

	#VAE framework
	print("VAE result")
	latent_file  = os.path.join( workPath,  'vae_out/scATAC_latent_VAE.txt' )
	cluster_file = os.path.join( workPath,  '5-cellinfo-RNA.tsv' ) 

	calculate_NMI_ARI( latent_file, cluster_file, 2, 0 )

	#POE framework
	print("MVAE-POE result")
	latent_file  = os.path.join( workPath,  'out/First_simulate_combine_poe.csv' )
	cluster_file = os.path.join( workPath,  '5-cellinfo-RNA.tsv' ) 

	calculate_NMI_ARI( latent_file, cluster_file, 2, 0 )

	### MVAE
	print("MVAE1 result")
	latent_file = os.path.join( workPath, 'out/First_simulate_combine_MVAE.csv')
	cluster_file = os.path.join( workPath,  '5-cellinfo-RNA.tsv' ) 

	calculate_NMI_ARI( latent_file, cluster_file, 2, 0 )

	print("MVAE2 result")

	latent_file = os.path.join( workPath, 'out/First_simulate_combine_MVAE_1.csv')
	cluster_file = os.path.join( workPath,  '5-cellinfo-RNA.tsv' ) 

	calculate_NMI_ARI( latent_file, cluster_file, 2, 0 )

	## for cell line datasest
	workPaths = "/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Real/SNARE-seq/CellLineMixture/sparsity_1009/0.9_0.9/scvi_out/"

	path2 = '/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Real/SNARE-seq/CellLineMixture/sparsity_1009/'

	latent_file = os.path.join( workPaths, '0.9_0.9_latent.csv')
	cluster_file = os.path.join( path2, 'Cell_line_cluster_1009.csv' ) 
	calculate_NMI_ARI( latent_file, cluster_file, 4, 0 )

	workPaths = "/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Real/SNARE-seq/CellLineMixture/sparsity_1009/0.9_0.9/"
	latent_file = os.path.join( workPaths, '5_latent.csv')
	calculate_NMI_ARI( latent_file, cluster_file, 4, 0 )

def MOFA_processing():

	workPath = "/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Real/SNARE-seq/CellLineMixture/out_activity/spasity_1004/"
	cluster_file = os.path.join( workPath, 'Cell_line_cluster_1004.csv' )

	NMI_list1 = []
	ARI_list1 = []
	file_list = []

	file_names = [ '0.7_0.7', '0.7_0.8', '0.7_0.9', '0.8_0.7', '0.8_0.8', '0.8_0.9', '0.9_0.7', '0.9_0.8', '0.9_0.9' ]

	for fn in file_names:

		temp_path = os.path.join( workPath, fn + '/MOFA_out/' )
		temp_file = os.path.join( temp_path, 'MOFA_combine_cluster.csv' )

		if path.exists( temp_file ):

			print(fn)
			NMI, ARI = calculate_NMI_ARI( temp_file, cluster_file, 4, 1 )
			NMI_list1.append(NMI)
			ARI_list1.append(ARI)
			file_list.append(fn)

	data_three_save  = { "file_list": file_list, "NMI_list1": NMI_list1, "ARI_list1": ARI_list1 }
	data_three_save1 = pd.DataFrame(data_three_save).to_csv(os.path.join( workPath, 'MOFA_process_NMI_ARI.txt' ))


def Preidct_mofa_scvi():

	workPath = "/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Real/SNARE-seq/CellLineMixture/sparsity_1009/"
	cluster_file = os.path.join( workPath, 'Cell_line_cluster_1009.csv' )

	file_names = [fn for fn in os.listdir( workPath )]
	NMI_list1 = []
	ARI_list1 = []
	file_list = []

	file_names = [ '0.5_0.5', '0.6_0.5', '0.7_0.5', '0.8_0.5', '0.9_0.5', '0.5_0.6', '0.6_0.6', '0.7_0.6', '0.8_0.6',
				   '0.9_0.6', '0.5_0.7', '0.6_0.7', '0.7_0.7', '0.8_0.7', '0.9_0.7' ]

	for fn in file_names:

		temp_path = os.path.join( workPath, fn + '/' )
		temp_file = os.path.join( temp_path, 'MOFA_combine_cluster.csv' )

		if path.exists( temp_file ):

			print(fn)
			NMI, ARI = calculate_NMI_ARI( temp_file, cluster_file, 4, 1 )
			NMI_list1.append(NMI)
			ARI_list1.append(ARI)
			file_list.append(fn)

	data_three_save  = { "file_list": file_list, "NMI_list1": NMI_list1, "ARI_list1": ARI_list1 }
	data_three_save1 = pd.DataFrame(data_three_save).to_csv(os.path.join( workPath, 'MOFA_process_NMI_ARI_0.567.txt' ))

	print("scvi processing")

	workPath = "/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Real/SNARE-seq/CellLineMixture/sparsity_1009/"

	file_names = [fn for fn in os.listdir( workPath )]
	NMI_list1 = []
	ARI_list1 = []
	file_list = []

	for fn in file_names:

		temp_path = os.path.join( workPath, fn + '/scvi_out/' )
		temp_file = os.path.join( temp_path, fn + '_latent_12.csv' )

		if path.exists( temp_file ):

			print(fn)
			NMI, ARI = calculate_NMI_ARI( temp_file, cluster_file, 4, 0 )

			NMI_list1.append(NMI)
			ARI_list1.append(ARI)
			file_list.append(fn)

	data_three_save  = { "file_list": file_list, "NMI_list1": NMI_list1, "ARI_list1": ARI_list1 }
	data_three_save1 = pd.DataFrame(data_three_save).to_csv(os.path.join( workPath, 'scVI_process_NMI_ARI_12.txt' ))

def MVAE_processing():

	workPath = "/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Real/sci-CAR/mouse_kidney/"
	cluster_file = os.path.join( workPath, 'single_cell_type.csv' )

	temp_path = os.path.join( workPath, 'MVAE_out/' )
	included_extensions = ['_latent.csv']
	file_names = [fn for fn in os.listdir( temp_path ) if any(fn.endswith(ext) for ext in included_extensions)]

	NMI_list1 = []
	ARI_list1 = []
	file_list = []

	for files in file_names:

		temp_file = os.path.join( temp_path, files )
		print(temp_file)

		if path.exists( temp_file ):
			NMI, ARI = calculate_NMI_ARI( temp_file, cluster_file, 14, 0 )
			NMI_list1.append(NMI)
			ARI_list1.append(ARI)
			file_list.append(files)

	data_three_save  = { "file_list": file_list, "NMI_list1": NMI_list1, "ARI_list1": ARI_list1 }
	data_three_save1 = pd.DataFrame(data_three_save).to_csv(os.path.join( temp_path, 'MVAE_process_NMI_ARI.csv' ))


def MVAE_procesing_v2():

	workPath = "/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Real/SNARE-seq/CellLineMixture/out_activity/spasity_1004/"
	cluster_file = os.path.join( workPath, 'Cell_line_cluster_1004.csv' )

	file_dirs = [fn for fn in os.listdir( workPath )]
	included_extensions = ['_latent.csv']

	file_dirs = [ '0.7_0.7', '0.7_0.8', '0.7_0.9', '0.8_0.7', '0.8_0.8', '0.8_0.9', '0.9_0.7', '0.9_0.8', '0.9_0.9' ]
	sub_dirs  = [ 'MVAE_POE_out', 'scVI_concat' ]

	sub_dirs  = [ 'MVAE_out' ]

	for fn in file_dirs:

		temp_path = os.path.join( workPath, fn + '/' )

		for sub in sub_dirs:

			temp_path1 = os.path.join( temp_path, sub + '/' )

			NMI_list1 = []
			ARI_list1 = []
			file_list = []

			if path.exists( temp_path1 ):

				file_names = [fn1 for fn1 in os.listdir( temp_path1 ) if any(fn1.endswith(ext) for ext in included_extensions)]

				for files in file_names:

					temp_file = os.path.join( temp_path1, files )
					print(temp_file)

					if path.exists( temp_file ):

						NMI, ARI = calculate_NMI_ARI( temp_file, cluster_file, 4, 0 )
						NMI_list1.append(NMI)
						ARI_list1.append(ARI)
						file_list.append(files)

				data_three_save  = { "file_list": file_list, "NMI_list1": NMI_list1, "ARI_list1": ARI_list1 }
				data_three_save1 = pd.DataFrame(data_three_save).to_csv(os.path.join( temp_path1, 'NMI_ARI_process.csv' ))

def calculate_gene_selection(self):
	aa_s = sys.argv[0]
	path = sys.argv[1]
	file = sys.argv[2]
	Type = sys.argv[3] ## 'RNA', 'ATAC'

	Data1 = pd.read_table( os.path.join( path, file ), header=0, index_col=0 )
	GeneNames = Data1.index.values
	calculate_pearson_lascore( path, Data1.values.T, GeneNames, Type)

def extract_sequence_by_genomic_location(path, file2, file3):

	from Bio import SeqIO

	f = open(file2, 'r')

	with open(file3,"w") as f1:

		for line in f.readlines():

			line = line.strip()
			
			x = line.split("\t")

			for seq_record in SeqIO.parse(os.path.join( path, x[0] + ".fa" ), "fasta"):

				f1.write(">" + x[3] + "\n")
				f1.write(str(seq_record.seq[int(x[1]) : int(x[2]) ]) + "\n")

	f.close()
	f1.close()

def calculate_Simulation():

	workPath     = '/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Simulated/Simulated_2/GMM_out/'
	cluster_file = os.path.join( '/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Simulated/Simulated_2/', '5-cellinfo-RNA.csv' )

	included_extensions = ['_latent.csv']

	sub_dirs = [ 'MVAE', 'MVAE_POE', 'scVI_concat' ]

	for sub in sub_dirs:

		NMI_list1 = []
		ARI_list1 = []
		file_list = []

		temp_paths = os.path.join( workPath, sub + '/' )
		file_names = [fn for fn in os.listdir( temp_paths ) if any(fn.endswith(ext) for ext in included_extensions)]

		for files in file_names:

			temp_file = os.path.join( temp_paths, files )
			out_files = temp_file + '-cluster'
			NMI, ARI  = calculate_NMI_ARI( temp_file, cluster_file, 4, 0 )

			NMI_list1.append(NMI)
			ARI_list1.append(ARI)
			file_list.append(files)

		data_three_save  = { "file_list": file_list, "NMI_list1": NMI_list1, "ARI_list1": ARI_list1 }
		data_three_save1 = pd.DataFrame(data_three_save).to_csv(os.path.join( workPath, sub + '_NMI_ARI.csv' ))





if __name__ == "__main__":

	#MOFA_processing()

	#MVAE_procesing_v2()

	temp_paths = "/sibcb1/chenluonanlab6/zuochunman/workPath/Multimodal/MVAE/Datasets/Real/SNARE-seq/P0_BrainCortex/lap_combine/new_one/scVI_concat/"
	included_extensions = ['_latent.csv']
   
	file_names = [fn for fn in os.listdir( temp_paths ) if any(fn.endswith(ext) for ext in included_extensions)]

	for files in file_names:

		temp_file = os.path.join( temp_paths, files )
		out_files = temp_file + '-cluster'

		print(temp_file)
		print(out_files)

		predict_clustering( temp_file, out_files, 19)

	#extract_sequence_by_genomic_location( "/sibcb1/chenluonanlab6/zuochunman/software/data/genomes/mm10/", "/sibcb1/chenluonanlab6/zuochunman/Reference/Ad_chr_region_3000.tsv", "/sibcb1/chenluonanlab6/zuochunman/Reference/Ad_chr_region_3000.fasta" )





