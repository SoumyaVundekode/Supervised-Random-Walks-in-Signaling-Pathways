# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from copy import deepcopy
import sys

###########################################################
#Inputs:
# G(V,E) - from tsv file
# feature_vector_matrix - from tsv file
# Restart probability alpha - 0.4 for now.
# source s - assuming V[4] for now.
# Target nodes set T
# Assumptions:
# Convergence threshold = 0.001
###########################################################

def get_graph_n_featureVector(tsv_file):
    V = []
    E = []
    feature_vector = {}
    for index,row in pd.DataFrame.from_csv(tsv_file, sep='\t').iterrows():
        if index not in V:
            V.append(index)
        u = V.index(index)
        if row.uniprot_b not in V:
            V.append(row.uniprot_b)
        v = V.index(row.uniprot_b)
        E.append([u,v])
        if u in feature_vector:
            feature_vector[u][v] = row.tolist()[2:]
        else: 
            feature_vector[u] = {v:row.tolist()[2:]}
        if row.is_directed=="TRUE":
            E.append([v,u])
            if v in feature_vector:
                feature_vector[v][u] = row.tolist()[2:]
            else: 
                feature_vector[v] = {u:row.tolist()[2:]}
    return V,E,feature_vector

def compute_A_n_dfdw(V,E,feature_vector,W):
    A = np.full((len(V),len(V)),0,dtype=float)
    dfdw = np.full((len(V),len(V),len(W)),0,dtype=float)
    for u in feature_vector:
        for v in feature_vector[u]:
            A[u][v] = pow((1+np.exp(0-np.dot(feature_vector[u][v],W))),-1)
            dfdw[u][v] = deepcopy(feature_vector[u][v])
            dfdw[u][v] = (np.exp(0-np.dot(feature_vector[u][v],W))*pow((1+np.exp(0-np.dot(feature_vector[u][v],W))),-2)) * dfdw[u][v]
    return A,dfdw

def compute_Q_n_dQdw(A,dfdw,alpha,src):
    Q = np.full_like(A,0)
    dQdw = np.full_like(dfdw,0)
    Qbar = np.full_like(Q,0)
    for u in feature_vector:
        A_sum = sum(A[u])
        for v in feature_vector[u]:
            Qbar[u][v] = A[u][v]/A_sum
    for u in range(len(A)):
        for v in range(len(A)):
            Q[u][v] = (1-alpha)*Qbar[u][v]
            if v==s:
                Q[u][v] += alpha
    dfdw_sum = np.full_like(dfdw[0][0],0)
    for j in range(len(A)):
        A_sum = sum(A[j])
        for u in range(len(A)):
            dfdw_sum = np.add(dfdw_sum,dfdw[j][u])
        for u in range(len(A)):
            if [j,u] in E:
                dQdw[j][u] = ((1-alpha)/pow(A_sum,2))*np.subtract((A_sum * dfdw[j][u]),(A[j][u] * dfdw_sum))
    return Q,dQdw

def compute_p_n_dpdw(V,Q,dQdw):
    p = [[1/float(len(V))] for u in range(len(V))]
    converged = False     
    conv_thresh = 0.001     # Is 0.001 fine for convergence value?
    t=1
    while not converged:
        converged = True
        for u in range(len(V)):
            p[u].append(0)
            for j in range(len(V)):
                p[u][t] += p[j][t-1]*Q[j][u]
        sum_p = sum(u[t] for u in p) # These 3 lines are for normalizing
        if sum_p>0:
            for u in range(len(V)):
                p[u][t] = p[u][t]/sum_p
        for u in range(len(V)):   # Check convergence
            if p[u][t]-p[u][t-1]>conv_thresh:
                converged= False
        t += 1
    dpdw = [[[0] for k in range(len(dQdw[0][0]))] for i in range(len(V))]
    for k in range(len(dQdw[0][0])):
        converged = False
        dt = 1
        while dt<t and not converged:
            converged = True
            for u in range(len(V)):
                dpdw[u][k].append(0)
                for j in range(len(V)):
                    dpdw[u][k][dt] += Q[j][u]*dpdw[j][k][dt-1] + p[j][dt-1]*dQdw[j][u][k]
            sum_dpdw_k = sum(u[k][dt] for u in dpdw)
            if sum_dpdw_k>0:
                for u in range(len(V)):
                    dpdw[u][k][dt] = dpdw[u][k][dt]/sum_dpdw_k
            for u in range(len(V)):
                if dpdw[u][k][dt]-dpdw[u][k][dt-1]>conv_thresh:
                    converged= False
            dt += 1
    return p,dpdw
 
if __name__ == '__main__':
    
    # Build the graph and feature vectors from the input file
    V,E,feature_vector = get_graph_n_featureVector('C:\Users\soumy\Documents\MS\Fall17\AD\project\Code\pathlinker_split_sample.tsv')
    
    ###############
    # TODO: Quasi-newton gradient descent optimization of F
    # Below are the subs that will be needed for the optimization problem
    ###############
    
    # Given W and feature_vector, compute A and dA matrices
    W = [0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 1L, 0L, 1L]
    A,dfdw = compute_A_n_dfdw(V,E,feature_vector,W)
    
    # Given A(=f),dfdw,alpha and s, compute Q and dQ
    alpha = 0.4
    s = V[4] # Picking a random node now. Needs to specified in input
    if s not in V:
        sys.exit("Given s is not in the list of nodes unless there are no edges from/to it at all. Please check.")
    Q,dQdw = compute_Q_n_dQdw(A,dfdw,alpha,V.index(s))
    
    # Given V,Q and dQdw, compute p and dpdw
    p,dpdw = compute_p_n_dpdw(V,Q,dQdw)
    
    #T = set([V[3],V[10],V[15]])
    