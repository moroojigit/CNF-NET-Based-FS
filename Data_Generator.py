# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:38:02 2022

@author: Reza
"""

import numpy as np 
import os 
import itertools
import copy
from collections import Counter
from bench2cnf import simple_read_bench, tseytin_t, cnf_to_matrix
import pickle as pk
import scipy.stats as ss

#from scipy.sparse import csgraph
#from grakel.graph import floyd_warshall
from scipy.stats import skew
from scipy.stats import kurtosis

def find_target_time(Target_link):
    f = open(Target_link, "r")
    for line in f:
        if "cpu_time" in line:
            Desired_line = line 
            break 
    Desired_List = Desired_line.split(';')
    Target = float(Desired_List[3].split('=')[-1])
    f.close()
    return Target

def ent(data):
    """ input probabilities to get the entropy """
    entropy = ss.entropy(data)
    return entropy

data_dir = os.path.join(os.getcwd(),'Dataset\Part_1')

#data_dir = 'Dataset\Part_1' 

# benchmark = ['c499', 'c2670','c5315']
benchmark = ['c499']
#De_en = ["encrypt","decrypt"]
Obfuscation_methods = ["IOLTS", "RI","RIF","TOC13_XOR"]
#Folders = list(itertools.product(benchmark, De_en))

for k in range(len(benchmark)):
    #Primary_Link = os.path.join(data_dir,Folders[k][0],Folders[k][1])
    Primary_Link = os.path.join(data_dir,benchmark[k])
    Primary_Link_bench = os.path.join(Primary_Link,"encrypt")
    Primary_Link_target = os.path.join(Primary_Link,"decrypt")
    All_data_features = []
    Time_Targets = []
    for each_method in Obfuscation_methods: 
        List_files = os.listdir(os.path.join(Primary_Link_bench,each_method,"1"))
        for file in List_files:
            if file.endswith(".txt"):
                continue
            feat = []
            wires = simple_read_bench(os.path.join(Primary_Link_bench,each_method,"1",file))
            cnf_clause_count, cnf_content = tseytin_t(wires)
            incidence_mat = cnf_to_matrix(cnf_content)
            
             # check if there is zero rows
            zero_row_id = (incidence_mat == 0).all(1)
            incidence_mat = incidence_mat[~zero_row_id]  # remove all zero rows
            cnf_clause_count, cnf_variable_counts = incidence_mat.shape  # refresh var and clause num
            
            # derive literal degree and clause degree
            var_degree = np.sum(np.abs(incidence_mat), axis=0)
            clause_degree = np.sum(np.abs(incidence_mat), axis=1)
            
            # build literal graph and extract literal degree and features
            var_graph = np.zeros((cnf_variable_counts, cnf_variable_counts))
            for i in range(cnf_clause_count):
                non_zero_list = np.where(incidence_mat[i] != 0)
                pair_combo = [pair for pair in itertools.product(non_zero_list[0], repeat=2)]
                for p in pair_combo:
                    var_graph[p] = 1
                var_graph_degree = np.sum(var_graph, axis=0)
                
            #var_graph_floyd = floyd_warshall(var_graph)
            #var_graph_floyd_degree = np.sum(var_graph_floyd, axis=0)   
            
            # extract positive and negative features from literal-clause graph
            positive_mat = copy.deepcopy(incidence_mat)
            positive_mat[positive_mat == -1] = 0
            
            negative_mat = copy.deepcopy(incidence_mat)
            negative_mat[negative_mat == 1] = 0
            negative_mat[negative_mat == -1] = 1
            
            positive_mat_0 = np.sum(positive_mat, axis=0)
            negative_mat_0 = np.sum(negative_mat, axis=0)
            positive_mat_1 = np.sum(positive_mat, axis=1)
            negative_mat_1 = np.sum(negative_mat, axis=1)
            p_ratio_0 = positive_mat_0 / (positive_mat_0 + negative_mat_0)
            n_ratio_0 = negative_mat_0 / (positive_mat_0 + negative_mat_0)
            p_ratio_1 = positive_mat_1 / (negative_mat_1 + positive_mat_1)
            n_ratio_1 = negative_mat_1 / (negative_mat_1 + positive_mat_1)
            
            # count ratio of binary and ternary clause 
            bin_tern_cnt = Counter(np.sum(np.abs(incidence_mat), axis=1))
            bin_ratio = bin_tern_cnt[2] / sum(bin_tern_cnt.values())
            tern_ratio = bin_tern_cnt[3] / sum(bin_tern_cnt.values())
            
            feat.append(tern_ratio)
            feat.append(bin_ratio)
                        
            feat.append(cnf_variable_counts)  # 0
            feat.append(cnf_clause_count)  # 1
            feat.append(float(cnf_clause_count / cnf_variable_counts))  # 2
            
            # extract entropy for each feature
            feat.append(ent(var_degree)) 
            feat.append(ent(clause_degree))
            feat.append(ent(var_graph_degree))
            feat.append(ent(p_ratio_0))
            feat.append(ent(p_ratio_1))
            feat.append(ent(n_ratio_0))
            feat.append(ent(n_ratio_1))
            feat.append(ent(n_ratio_1))
            #feat.append(ent(var_graph_floyd_degree))
            
            
            feat.append(np.mean(var_degree))
            feat.append(np.mean(clause_degree))
            feat.append(np.mean(var_graph_degree))
            feat.append(np.mean(p_ratio_0))
            feat.append(np.mean(p_ratio_1))
            feat.append(np.mean(n_ratio_0))
            feat.append(np.mean(n_ratio_1))
            feat.append(np.mean(n_ratio_1))
            #feat.append(np.mean(var_graph_floyd_degree))
            
            
            feat.append(np.var(var_degree))
            feat.append(np.var(clause_degree))
            feat.append(np.var(var_graph_degree))
            feat.append(np.var(p_ratio_0))
            feat.append(np.var(p_ratio_1))
            feat.append(np.var(n_ratio_0))
            feat.append(np.var(n_ratio_1))
            feat.append(np.var(n_ratio_1))
            #feat.append(np.var(var_graph_floyd_degree))
            
            feat.append(skew(var_degree))
            feat.append(skew(clause_degree))
            feat.append(skew(var_graph_degree))
            feat.append(skew(p_ratio_0))
            feat.append(skew(p_ratio_1))
            feat.append(skew(n_ratio_0))
            feat.append(skew(n_ratio_1))
            feat.append(skew(n_ratio_1))
            #feat.append(skew(var_graph_floyd_degree))
            
            feat.append(kurtosis(var_degree))
            feat.append(kurtosis(clause_degree))
            feat.append(kurtosis(var_graph_degree))
            feat.append(kurtosis(p_ratio_0))
            feat.append(kurtosis(p_ratio_1))
            feat.append(kurtosis(n_ratio_0))
            feat.append(kurtosis(n_ratio_1))
            feat.append(kurtosis(n_ratio_1))
            #feat.append(kurtosis(var_graph_floyd_degree))
            
            feat.append([var_degree])
            feat.append([clause_degree])
            feat.append([var_graph_degree])
            feat.append([p_ratio_0])
            feat.append([p_ratio_1])
            feat.append([n_ratio_0])
            feat.append([n_ratio_1]) 
            #feat.append(var_graph_floyd_degree)

            All_data_features.append(feat)
        List_target_files = os.listdir(os.path.join(Primary_Link_target,each_method,"1"))
        for file in List_target_files:
            time = find_target_time(os.path.join(Primary_Link_target,each_method,"1",file))
            Time_Targets.append(time)
            
    pk.dump(All_data_features, open(os.path.join(os.getcwd(),'{}_X.pk'.format(benchmark[k])), 'wb'))
    pk.dump(Time_Targets, open(os.path.join(os.getcwd(),'{}_Y.pk'.format(benchmark[k])), 'wb'))
        
             

        
        
    
    