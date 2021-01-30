#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:13:21 2020

@author: qspinat
"""

import numpy as np
from numba import jit,prange
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import functools

#%%#################### cost function ####################

def cost(X,Y,a):
    return np.sum((X-Y[a])**2)

#%%################################

@jit(nopython=True)
def brut_force(X,Y):
    
    m = X.shape[0]
    n = Y.shape[0]
    a = np.zeros(m).astype(np.int64)
    best_a = np.zeros(m).astype(np.int64)

    test = False
    c=0.
    best_c = 0.
    
    print(n**m)
    
    for i in range(n**m):
        for j in range(m):
            a[j] = int(i//(n**j))%n
            if np.unique(a).shape[0]==m:
                c = np.sum((X-Y[a])**2)
                if test == False :
                    best_c = c
                    best_a = a.copy()
                    test = True
                elif c<best_c : 
                    best_c = c
                    best_a = a.copy()
    return best_a


@jit(nopython=True)
def nn_assignment(X,Y):
    """
    Parameters
    ----------
    X : sorted X, size m<n
    Y : sorted Y, size n
    t : array of integer to stock the assignement

    Returns function t wich assign X[i] to Y[t[i]]
    -------
    None.

    """
    m = X.shape[0]
    n = Y.shape[0]
    
    t = np.zeros(m).astype(np.int64)
    
    i=0
    j=0
    while(i<m):
        if X[i]<=Y[j]:
            t[i]=j
            i+=1
        elif j==n-1:
            t[i]=n-1
            i+=1
        elif Y[j+1]<X[i]:
            j+=1
        elif np.abs(X[i]-Y[j])<np.abs(X[i]-Y[j+1]):
            t[i]=j
            i+=1
        else :
            t[i]=j+1
            j+=1
            i+=1
    return t

@jit(nopython=True)
def quad_assignment_preprocessed_jit(X,Y,t):
    """
    Parameters
    ----------
    X : sorted X, size m<n, allready preprocessed
    Y : sorted Y, size n, allready preprocessed
    t : optimal assigment X to Y (allready done)
    a : array of integer to stock the injective assignement

    Returns a injective optimal assignement
    -------
    None doesn't simplify the problem
    s = rightmost current free point
    a[r] = s + 1
    """
    
    m = X.shape[0]
    n = Y.shape[0]
    a = np.zeros(m).astype(np.int64)

    
    #initialization
    a[0] = t[0]
    s = a[0]-1
    r = 0
    S1 = 0.
    S2 = 0.
    
    for i in range(0,m-1):
        if t[i+1]>a[i]:
            a[i+1] = t[i+1]
            if a[i+1]>a[i]+1:
                s = a[i+1]-1
                r = i+1
                S1 = 0.
                S2 = 0.
            else:
                S1 += (X[i]-Y[a[i]-1])**2
                S2 += (X[i]-Y[a[i]])**2
        else:
            #subcases
            if s<0:
                #case 2
                a[i+1] = a[i]+1
                S1 += (X[i]-Y[a[i]-1])**2
                S2 += (X[i]-Y[a[i]])**2
            elif a[i]==n-1:
                S1 += (X[i]-Y[a[i]-1])**2
                #case 1
                a[i+1]=a[i]
                a[r:i+1]=np.arange(s,a[i+1])
                # change s and r and update S2
                S2 = S1
                for j in range(r,-1,-1):
                    if j !=0:
                            S2 += (X[j]-Y[a[j]])**2
                    if j == 0:
                        s = a[0]-1
                        r = 0
                        break
                    elif a[j]>a[j-1]+1:
                        s = a[j]-1
                        r = j
                        break
                #update S1
                S1 = np.sum((X[r:i+1]-Y[a[r:i+1]-1])**2)
            
            else:
            # compute sums
#                print(a[i])
#                print(Y.shape)
#                print(Y[a[i]])
                S1 += (X[i]-Y[a[i]-1])**2
                S2 += (X[i]-Y[a[i]])**2
                w1 = S1 + (X[i+1]-Y[a[i]])**2
                w2 = S2 + (X[i+1]-Y[a[i]+1])**2
                #w1_ = np.sum((X[r:i+1]-Y[a[r:i+1]-1])**2) + (X[i+1]-Y[a[i]])**2
                #w2_ = np.sum((X[r:i+1]-Y[a[r:i+1]])**2) + (X[i+1]-Y[a[i]+1])**2
                
                #case 1
                if w1<w2:
                    a[i+1]=a[i]
                    a[r:i+1]=np.arange(s,a[i+1])
                    
                    # change s and r and update S2
                    S2 = S1
                    for j in range(r,-1,-1):
                        if j !=r:
                            S2 += (X[j]-Y[a[j]])**2
                        if j == 0:
                            s = a[0]-1
                            r = 0
                            break
                        elif a[j]>a[j-1]+1:
                            s = a[j]-1
                            r = j
                            break
                    #update S1
                    S1 = np.sum((X[r:i+1]-Y[a[r:i+1]-1])**2)
                #case 2
                else:
                    a[i+1]=a[i]+1                    

    return a

def quad_assignment_preprocessed(X,Y):
    t = nn_assignment(X, Y)
    a = quad_assignment_preprocessed_jit(X,Y,t)
    return a

@jit(nopython=True)
def quad_assignment_jit(X,Y,t):
    """

    Parameters
    ----------
    X : sorted X, size m<n
    Y : sorted Y, size n
    t : array of integer to stock the assignement(allready done)
    a : array of integer to stock the injective assignement

    Returns a optimal injective assigment t
    -------
    None.

    """
    m = X.shape[0]
    n = Y.shape[0]
    
    a = np.zeros(m).astype(np.int64)

    
    #################  symplifying the problem ###############
    
    # n == m
    if n==m:
        for i in range(m):
            a[i] = i
        return a
    
    # m == n-1
    if m==n-1:
        res_opt = 0
        res = 0
        k_opt=-1
        for k in range(0,m):
            res += (X[k]-Y[k])**2-(X[k]-Y[k+1])**2
            if res<res_opt:
                res_opt=res
                k_opt=k
        a[:k_opt+1] = np.arange(k_opt+1)
        a[k_opt+1:] = np.arange(k_opt+2,n)
        return a
    
#    a = np.zeros(m,dtype=np.int)
  
    # X before Y
    ind_min=0
    while X[ind_min]<=Y[ind_min]:
        a[ind_min]=ind_min
        ind_min+=1
        if ind_min == m:
            return a
    
    
    # X after Y
    ind_max=-1
    while X[m+ind_max]>=Y[n+ind_max]:
        a[m+ind_max]=n+ind_max
        ind_max-=1
        if m+ind_max == ind_min-1:
            return a
        
    #print("OK",ind_min,ind_max)
    
    t = t[ind_min:m+ind_max+1]-ind_min
    
    # number of non-injective values of t
    p = t.shape[0]-np.unique(t).shape[0]
    if p==0:
        a[ind_min:m+ind_max+1]=t+ind_min
        return a
       
    # for numba
    if t[0]-p>0:
        ind_min_Y = ind_min+t[0]-p
    else :
        ind_min_Y = ind_min
    
    if n+ind_max-ind_min<t[m+ind_max-ind_min]+p:
        ind_max_Y = n+ind_max-ind_min+ind_min
    else :
        ind_max_Y = t[m+ind_max-ind_min]+p+ind_min
    #print("OKOK",ind_min_Y,ind_max_Y-n)
    
    # assigment    
    a[ind_min:m+ind_max+1] = quad_assignment_preprocessed_jit(X[ind_min:m+ind_max+1], Y[ind_min_Y:ind_max_Y+1], t-ind_min_Y+ind_min)+ind_min_Y
    
    return a

def quad_assignment(X,Y):
    t = nn_assignment(X, Y)
    a = quad_assignment_jit(X,Y,t)
    return a

#%%#################### Assigment Decomposition ####################

@jit(nopython=True)
def assignment_decomp_jit(X,Y,t,log=False):
    """
    
    Parameters
    ----------
    X : sorted X, size m<n
    Y : sorted Y, size n
    t : array of integer to stock the assignement, size m (allready done)
    f : array of boolean, size m
    A : array of integer, shape (m,4) 

    Returns
    -------
    A[:,0:2] = list of first and last indice included of X in each subproblem
    A[:,3] = last available free spot for each subproblem
    A[:,4] = currently last value considered for each subproblem
    A.shape[0] is the numberof subproblems
    i.e. subproblem i is X[A[i,0]:A[i,1]+1] --> Y[A[i,2]+1:A[i,3]+1]
    
    not memory efficient

    """
        
    m = X.shape[0]
    n = Y.shape[0]
    
    f = np.ones(Y.shape[0])==1
    if X.shape[0]>1000:
        A = np.zeros((X.shape[0]//100,4)).astype(np.int64)
    elif X.shape[0]>100:
        A = np.zeros((X.shape[0]//10,4)).astype(np.int64)
    else:
        A = np.zeros((10,4)).astype(np.int64)
    
    cnt=0
        
    #Initialization
    f[t[0]] = False
    # update s
    s = t[0]-1
    l = t[0]
    #create A_k    
    A[0,:] = 0
    A[0,2] = t[0]-1
    A[0,3] = t[0]
            
    cnt+=1
            
    for i in range(1,m):
        if log and (i%(m//10))==0:
            print(i//(m//100))
        # Nouveau subproblem
        if f[t[i]]:
            f[t[i]] = False
            # update s
            s = t[i]-1
            l = t[i]
            # increase of A if necessary
            if cnt>A.shape[0]:
                lost_vect = A[cnt-1].copy()
                new_dim = min(A.shape[0],X.shape[0]-A.shape[0])
                A = np.concatenate((A,np.zeros((new_dim,4)).astype(np.int64))).copy()
                A[cnt-1]=lost_vect.copy()
            #create A_k            
            A[cnt,0] = i
            A[cnt,1] = i
            A[cnt,2] = s
            A[cnt,3] = l
            
            cnt+=1
        
        # Mise a jour probleme
        else:
            # Premier cas : on extend à droite et à gauche
            k1 = cnt-1 #A_Y[t[i]] ==cnt-1 !!
            if t[i] == t[i-1]:
                # tant que s_k1 pris, on fusionne les problemes
                while A[k1][2]>=0 and not f[A[k1][2]]:
                    k2 = k1-1
                    A[k2][1] = A[k1][1]
                    A[k2][3] = A[k1][3]
                    A[k1][2]=-2
                    A[k1][3]=-2   
                    k1=k2
                    cnt-=1
                    ##
                # tant que l_k1 +1 pris, on fusionne les problemes --> n'arrive jamais
                A[k1][1] = i # np.max([i,AX[k1][1]]) # just i ?
                if A[k1][2] >=0 : 
                    f[A[k1][2]]=False
                    A[k1][2]-=1
                if A[k1][3]<n-1 : 
                    f[A[k1][3]+1]=False
                    A[k1][3]+=1
            # Deuxieme cas : on extend à droite uniquement
            else :
                # tant que l_k1 +1 pris, on fusionne les problemes --> n'arrive jamais
                A[k1][1] = i #np.max([i,AX[k1][1]]) # just i?
                if A[k1][3]<n-1 : 
                    f[A[k1][3]+1]=False
                    A[k1][3]+=1
                    
    A = A[:cnt]
    A_to_keep = np.where(A[:,2]!=-2)
    A = A[A_to_keep]
              
    return A

def assignment_decomp(X, Y,log=False):
    t = nn_assignment(X, Y)
    return assignment_decomp_jit(X, Y, t,log=log)

#%%################### optimalinjective assigment with assignment decomposition #####################################

@jit(nopython=True,parallel=True)
def assignment_jit(X,Y,t,A):
    """   
    Parameters
    ----------
    X : sorted X, size m<n
    Y : sorted Y, size n
    t : array of integer to stock the assignement size m (allready done)
    a : array of integer to stock the injective assignement, size m
    A : subproblem decomposition given by assignment_decomp

    Returns
    -------
    a : injective assignement

    """ 
    
    a = np.zeros(X.shape[0]).astype(np.int64) 
    
    for i in prange(A.shape[0]):
        a[A[i][0]:A[i][1]+1] = quad_assignment_jit( X[A[i][0]:A[i][1]+1] , Y[A[i][2]+1:A[i][3]+1], t[A[i][0]:A[i][1]+1]-A[i][2]-1) + A[i][2]+1
       
    return a

def f_pool(i,X,Y,t,A):
    return quad_assignment_jit(X[A[i][0]:A[i][1]+1] , Y[A[i][2]+1:A[i][3]+1], t[A[i][0]:A[i][1]+1]-A[i][2]-1) + A[i][2]+1


def assignment_pool(X,Y,t,A):
    """   
    Parameters
    ----------
    X : sorted X, size m<n
    Y : sorted Y, size n
    t : array of integer to stock the assignement size m (allready done)
    a : array of integer to stock the injective assignement, size m
    A : subproblem decomposition given by assignment_decomp

    Returns
    -------
    a : injective assignement

    """ 
    
    # definir  fonction
    
    liste = np.arange(A.shape[0],dtype=np.int64)
    
    copier = functools.partial(f_pool, X=X, Y=Y,t=t,A=A)
       
    with Pool(6) as p:
        a = p.map(copier, liste) 
    
    return np.concatenate(a,axis=0)

@jit(nopython=True)
def assignment(X,Y):
    t = nn_assignment(X, Y)
    A = assignment_decomp_jit(X,Y,t)
    a = assignment_jit(X,Y,t,A)

    return a

def assignment_bis(X,Y):
    t = nn_assignment(X, Y)
    A = assignment_decomp_jit(X,Y,t)
    a = assignment_pool(X,Y,t,A)

    return a

#%%################### FIST #######################

@jit(nopython=True)
def FIST_image(X,Y,n_iter,c=None):
    X_match = X.copy().reshape(X.shape[0]*X.shape[1],3)
    Y_match = Y.copy().reshape(Y.shape[0]*Y.shape[1],3)
    alpha = 1
    for i in range(n_iter):
        if ((i%(n_iter//100))==0):
            print((i*100)//n_iter,'%')
        
        # coef SGD
        if c != None:
            alpha = c/(1+i)**0.6
        
        # direction
        theta = np.pi*np.random.uniform(0,1)
        phi = np.pi*np.random.uniform(0,1)
        
        #projection
        X_proj = X_match[:,0]*np.sin(phi)*np.cos(theta)+X_match[:,1]*np.sin(phi)*np.sin(theta)+X_match[:,2]*np.cos(phi)
        Y_proj = Y_match[:,0]*np.sin(phi)*np.cos(theta)+Y_match[:,1]*np.sin(phi)*np.sin(theta)+Y_match[:,2]*np.cos(phi)
        
        #sort
        X_sort_indices = np.argsort(X_proj,kind='mergesort')
        Y_sort_indices = np.argsort(Y_proj,kind='mergesort')
        
        #assigment
        t = nn_assignment(X_proj[X_sort_indices], Y_proj[Y_sort_indices])  
        a = quad_assignment_jit(X_proj[X_sort_indices], Y_proj[Y_sort_indices], t)
        #A = assignment_decomp_jit(X_proj[X_sort_indices],Y_proj[Y_sort_indices],t)
        #a = assignment_jit(X_proj[X_sort_indices], Y_proj[Y_sort_indices], t, A)
        
        #gradient
        grad = X_proj[X_sort_indices]-Y_proj[Y_sort_indices][a]
        if ((i%(n_iter//100))==0):
            print("gradient norm :",np.linalg.norm(grad))
        #grad=grad*m/grad_norm
        #problem ici
        X_match[X_sort_indices,0] -= grad*alpha*np.sin(phi)*np.cos(theta)
        X_match[X_sort_indices,1] -= grad*alpha*np.sin(phi)*np.sin(theta)
        X_match[X_sort_indices,2] -= grad*alpha*np.cos(phi)
        
    return X_match.reshape(X.shape)

### FIST image channels

@jit(nopython=True)
def FIST_features(X,Y,n_iter,dim=3,c=None):
    X_match = X.copy()
    Y_match = Y.copy()
    alpha = 1
    for i in range(n_iter):
        
        # coef SGD
        if c != None:
            alpha = c/(1+i)**0.6
        
        # direction
        direction = np.sign(np.random.randn(dim))*np.random.uniform(0.001,10,size=dim)
        direction = direction/np.linalg.norm(direction)
        
        #projection
        X_proj = X_match.dot(direction)
        Y_proj = Y_match.dot(direction)
        #X_proj = X_match[:,0]*np.sin(phi)*np.cos(theta)+X_match[:,1]*np.sin(phi)*np.sin(theta)+X_match[:,2]*np.cos(phi)
        #Y_proj = Y_match[:,0]*np.sin(phi)*np.cos(theta)+Y_match[:,1]*np.sin(phi)*np.sin(theta)+Y_match[:,2]*np.cos(phi)
        
        #sort
        X_sort_indices = np.argsort(X_proj,kind='mergesort')
        Y_sort_indices = np.argsort(Y_proj,kind='mergesort')
        
        #assigment
        t = nn_assignment(X_proj[X_sort_indices], Y_proj[Y_sort_indices])  
        a = quad_assignment_jit(X_proj[X_sort_indices], Y_proj[Y_sort_indices], t)
        #A = assignment_decomp_jit(X_proj[X_sort_indices],Y_proj[Y_sort_indices],t)
        #a = assignment_jit(X_proj[X_sort_indices], Y_proj[Y_sort_indices], t, A)
        
        #gradient
        grad = X_proj[X_sort_indices]-Y_proj[Y_sort_indices][a]
        if ((i%(n_iter//10))==0):
            print("gradient norm :",np.linalg.norm(grad))
            print((i*100)//n_iter,'%')

        #grad=grad*m/grad_norm
        for k in range(dim):
            X_match[X_sort_indices,k] -= grad*alpha*direction[k]
        #X_match[X_sort_indices,0] -= grad*alpha*np.sin(phi)*np.cos(theta)
        #X_match[X_sort_indices,1] -= grad*alpha*np.sin(phi)*np.sin(theta)
        #X_match[X_sort_indices,2] -= grad*alpha*np.cos(phi)
        
    return X_match


#%%############################### shape matching ##############################

#@jit(nopython=True)
def FIST_2D_similarity(X,Y,n_iter, plot=None):

    X_match = X.copy()

    for i in range(n_iter):
        if ((i%(n_iter//10))==0):
            print((i*100)//n_iter,'%')
        
        # direction
        theta = np.pi*np.random.uniform(0,1)       
        
        #projection
        X_proj = X_match[:,0]*np.cos(theta)+X_match[:,1]*np.sin(theta)
        Y_proj = Y[:,0]*np.cos(theta)+Y[:,1]*np.sin(theta)
        
        #sort
        X_sort_indices = np.argsort(X_proj,kind='mergesort')
        Y_sort_indices = np.argsort(Y_proj,kind='mergesort')
        
        #assigment
        t = nn_assignment(X_proj[X_sort_indices], Y_proj[Y_sort_indices])  
        a = quad_assignment_jit(X_proj[X_sort_indices], Y_proj[Y_sort_indices], t)
        
        cx = np.mean(X_match,axis=0)
        cy = np.mean(Y[Y_sort_indices][a],axis=0)
        #A = assignment_decomp_jit(X_proj[X_sort_indices],Y_proj[Y_sort_indices],t)
        #a = assignment_jit(X_proj[X_sort_indices], Y_proj[Y_sort_indices], t, A)
        
        # update step     
        #rotation A
        M = (Y[Y_sort_indices][a]-cy).T.dot(X_match[X_sort_indices]-cx)
        u,s,vh = np.linalg.svd(M)
        O = u.dot(vh)
        d = np.linalg.det(O)
        S = np.eye(vh.shape[0])
        S[-1,-1] = d
        A = u.dot(S).dot(vh)
        
        #scale
        stdx = np.std(np.linalg.norm(X_match-cx,axis=1))
        stdy = np.std(np.linalg.norm(Y[Y_sort_indices][a]-cy,axis=1))
        #print(stdx,stdy)
        scale = stdy/stdx
        
        #update
        X_match = scale*(X_match-cx).dot(A.T)+cy

        
        if ((i%(n_iter//10))==0):
            print("objective norm :", np.linalg.norm(X_match[X_sort_indices]-Y[Y_sort_indices][a]))
        if plot !=None and i!=0 and ((i%(n_iter//plot))==0):
            plt.scatter(X_match[:,0],X_match[:,1], label = "iteration "+str(i))
        
    return X_match

