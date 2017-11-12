# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from copy import deepcopy
import sys
from scipy import optimize

###########################################################
#Inputs:
# G(V,E) - from tsv file
# feature_vector_matrix - from tsv file
# Restart probability alpha - 0.4 for now.
# Source nodes set S
# Target nodes set T
###########################################################


# Get nodes, edges, sources, targets and feature vectors from the input files
def get_graph_details(tsv_nodes_file,tsv_psi_file):
    V = []  # list of nodes
    E = []  # list of edges
    feature_vector = {}  # feature vectors
    S = set()
    T = set()
    for index,row in pd.DataFrame.from_csv(tsv_nodes_file, sep='\t').iterrows():
        V.append(index)
        if str(row.Intermediate).lower()=="true":
            T.add(index)
        if str(row.Source).lower()=="true":
            S.add(index)
    for index,row in pd.DataFrame.from_csv(tsv_psi_file, sep='\t').iterrows():
        u = V.index(index)
        v = V.index(row.uniprot_b)
        E.append([u,v])
        if u in feature_vector:
            feature_vector[u][v] = row.tolist()[2:]
        else: 
            feature_vector[u] = {v:row.tolist()[2:]}
        if str(row.is_directed).lower()=="false":   # bidirectional - same psi for both
            E.append([v,u])
            if v in feature_vector:
                feature_vector[v][u] = row.tolist()[2:]
            else: 
                feature_vector[v] = {u:row.tolist()[2:]}
    if len(S)==0:
        sys.exit("There doesn't seem to be a single source among the nodes. Please recheck")
    return V,E,feature_vector,S,T

# Compute edge strengths matrix A and their derivatives dfdw.
# A will be nxn matrix
# A = f(W,ψ) = (1+exp(−ψ.W))^(−1) - calculated for each (u,v)
def compute_A_n_dfdw(W,V,E,feature_vector):
    A = np.full((len(V),len(V)),0,dtype=float)
    dfdw = np.full((len(V),len(V),len(W)),0,dtype=float)
    for u in feature_vector:
        for v in feature_vector[u]:
            A[u][v] = pow((1+np.exp(0-np.dot(feature_vector[u][v],W))),-1)
            dfdw[u][v] = deepcopy(feature_vector[u][v])
            dfdw[u][v] = (np.exp(0-np.dot(feature_vector[u][v],W))*pow((1+np.exp(0-np.dot(feature_vector[u][v],W))),-2)) * dfdw[u][v]
    return A,dfdw

# Compute random walk transition probability matrix Q and it's derivative dQdw
# We compute random walk stochastic transition matrix  Q' (Qbar) is first, then Q, then dQdw values
def compute_Q_n_dQdw(A,dfdw,alpha,src):
    Q = np.full_like(A,0)
    dQdw = np.full_like(dfdw,0)
    Qbar = np.full_like(Q,0)
    for u in feature_vector:   # Qbar computation
        A_sum = sum(A[u])
        for v in feature_vector[u]:
            Qbar[u][v] = A[u][v]/A_sum
    for u in range(len(A)):           # Q_uv values computation
        for v in range(len(A)):
            Q[u][v] = (1-alpha)*Qbar[u][v]
            if v==src:
                Q[u][v] += alpha
    dfdw_sum = np.full_like(dfdw[0][0],0)
    for j in range(len(A)):      # dQdw vectors computation
        A_sum = sum(A[j])
        for u in range(len(A)):
            dfdw_sum = np.add(dfdw_sum,dfdw[j][u])
        for u in range(len(A)):
            if [j,u] in E:
                dQdw[j][u] = ((1-alpha)/pow(A_sum,2))*np.subtract((A_sum * dfdw[j][u]),(A[j][u] * dfdw_sum))
    return Q,dQdw

# Compute random walk scores p and their derivatives dpdw
# p[u][-1] will be the score for u. Taking the last value in the list p[u]. Saving values of scores at all t values till convergence since we need them in dpdw vectors computation
# dpdw[u][-1] will be the derivative vector for u.
def compute_p_n_dpdw(V,Q,dQdw):
    p = [[1/float(len(V))] for u in range(len(V))]
    converged = False     
    conv_thresh = 0.0001
    t=1
    while not converged:
        converged = True
        for u in range(len(V)):
            p[u].append(0)
            for j in range(len(V)):
                p[u][t] += p[j][t-1]*Q[j][u]
        sum_p = sum(u[t] for u in p) # normalizing
        if sum_p>0:
            for u in range(len(V)):
                p[u][t] = p[u][t]/sum_p
        for u in range(len(V)):   # Check convergence
            if abs(p[u][t]-p[u][t-1])>conv_thresh:
                converged= False
        t += 1
    dpdw = [[[0.0]*len(dQdw[0][0]) for i in range(1)] for i in range(len(V))]
    for k in range(len(dQdw[0][0])):
        converged = False
        dt = 1
        while dt<t and not converged:
            converged = True
            for u in range(len(V)):
                dpdw[u].append([0.0]*len(dQdw[0][0]))
                for j in range(len(V)):
                    dpdw[u][dt][k] += Q[j][u]*dpdw[j][dt-1][k] + p[j][dt-1]*dQdw[j][u][k]
            sum_dpdw_k = sum(u[dt][k] for u in dpdw)
            if sum_dpdw_k>0:
                for u in range(len(V)):
                    dpdw[u][dt][k] = dpdw[u][dt][k]/sum_dpdw_k
            for u in range(len(V)):
                if abs(dpdw[u][dt][k]-dpdw[u][dt-1][k])>conv_thresh:
                    converged= False
            dt += 1
    return p,dpdw

# h = {(x+m)^2,0}
def compute_h(x,m):
    if x<0:
        return ((x+m)**2)
    else:
        return (0.0)

# dh/dx = 2(x+m)
def compute_h_der(x,m):
    if x<0:
        return (2*(x+m))
    else:
        return (0.0)

# Compute the objective function value for a given W vector and regularization paramters
def compute_F(W,V,E,feature_vector,S,T,alpha,lambda1,lambda2,h_margin):
    
    s = list(S)[0] # single-source at this point
    
    # Given W and feature_vector, compute A and dA matrices
    A,dfdw = compute_A_n_dfdw(W,V,E,feature_vector)
    # Given A(=f),dfdw,alpha and s, compute Q and dQ
    Q,dQdw = compute_Q_n_dQdw(A,dfdw,alpha,V.index(s))
    # Given V,Q and dQdw, compute p and dpdw
    p,dpdw = compute_p_n_dpdw(V,Q,dQdw)
    
    W_square_sum = sum(w*w for w in W)
    h_sum = 0
    for t in T:
        for u in set(V).difference(T):
            h_sum += compute_h((p[V.index(t)][-1]-p[V.index(u)][-1]),h_margin)
    return ((lambda1*W_square_sum) + (lambda2*h_sum))

# Computing gradient vectors of objective function for given W and regularization parameters
def compute_F_derivative(W,V,E,feature_vector,S,T,alpha,lambda1,lambda2,h_margin):
    
    s = list(S)[0] # single-source at this point
    
    # Given W and feature_vector, compute A and dA matrices
    A,dfdw = compute_A_n_dfdw(W,V,E,feature_vector)    
    # Given A(=f),dfdw,alpha and s, compute Q and dQ
    Q,dQdw = compute_Q_n_dQdw(A,dfdw,alpha,V.index(s))
    # Given V,Q and dQdw, compute p and dpdw
    p,dpdw = compute_p_n_dpdw(V,Q,dQdw)
    
    F_der = np.full_like(W,0)
    for t in T:
        for u in set(V).difference(T):
            F_der = np.add(F_der,(compute_h_der((p[V.index(t)][-1]-p[V.index(u)][-1]),h_margin)*np.subtract(dpdw[V.index(t)][-1],dpdw[V.index(u)][-1])))
    return np.add(np.multiply(2*lambda1,W),np.multiply(lambda2,F_der))
    

if __name__ == '__main__':
    
    # Build the graph and feature vectors from the input file
    V,E,feature_vector,S,T = get_graph_details("C:\\Users\\soumy\\Documents\\MS\\Fall17\\AD\\project\\Code\\egfr-ex-node.tsv","C:\\Users\\soumy\\Documents\\MS\\Fall17\\AD\\project\\Code\\egfr-example-psi.tsv")
    
    # Regularization Paramters
    alpha = 0.4
    lambda1 = 0.5
    lambda2 = 0.5
    h_margin = 0.3
    
    # Quasi-newton gradient descent optimization of F - http://www.scipy-lectures.org/advanced/mathematical_optimization/
    W = [0, 0, 0, 0, 6, 4, 0, 0, 0, 0, 2,2 , 3] # Starting W
    opt_res = optimize.minimize(compute_F, W, args=(V,E,feature_vector,S,T,alpha,lambda1,lambda2,h_margin), method="BFGS", jac=compute_F_derivative)
    print (opt_res) #opt_res.fun and opt_res.x are the optimal objective function value and optimal W respectively