####### HMM ##########

import librosa as ls
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

#######wav file read and pre-processing##############

re_mfcc = []
ri_mfcc = []
for i in os.listdir('./re'):
    if i.startswith('re'):
        path = os.path.join('./re',i)
        wav,sr = ls.load(path, duration=0.21, sr=None)
        mfccs = ls.feature.mfcc(y=wav, sr=sr, n_mfcc=13, hop_length=int(0.015*sr), n_fft=int(0.025*sr))
        re_mfcc.append(mfccs.T)

for i in os.listdir('./ri'):
    if i.startswith('ri'):
        path = os.path.join('./ri',i)
        wav,sr = ls.load(path, duration=0.21, sr=None)
        mfccs = ls.feature.mfcc(y=wav, sr=sr, n_mfcc=13, hop_length=int(0.015*sr), n_fft=int(0.025*sr))
        ri_mfcc.append(mfccs.T)
        
######Train-test split on the normalized data#########

re_mfcc = (np.array(re_mfcc)-np.mean(re_mfcc,axis=0))/np.std(re_mfcc,axis=0)
ri_mfcc = (np.array(ri_mfcc)-np.mean(ri_mfcc,axis=0))/np.std(ri_mfcc,axis=0)

re_train,re_test = train_test_split(re_mfcc, test_size=0.2)
ri_train,ri_test = train_test_split(ri_mfcc, test_size=0.2)

#######parameters init############

N,N1 = re_train[0].shape # number of nodes, parameters of each gaussian in the node
k = 7                     # model order
M  = np.random.uniform(-1,1,(k,N1))   # mean initialization
C = []
for i in range(k):
    C.append(np.identity(N1))
C = np.array(C)
A = np.ones((k,k))*1.0/3 # transition probabilities
P = np.random.uniform(1,10,(1,k))
P = P/np.sum(P)

########functions defined according to Bishop book###################

def normal(x, mu, sigma):
    d = len(x)
    px = 1/np.sqrt(np.linalg.det(sigma) * ((2*np.pi)**d))
    px *= np.exp(-0.5*np.matmul(np.matmul(np.transpose(x-mu), np.linalg.inv(sigma)), x-mu))
    return px

def alpha(x,pis,weights,mean,covar):
    alphas = []
    N,_ = x.shape
    # i=0
    for i in range(0,N):
        probs = []
        for j in range(k):
            probs.append(normal(x[i],mean[j],covar[j]))
        probs = np.array(probs)
        probs = probs.reshape(1,k)
        if(i==0):
            alphas.append(pis*probs)
        else:
            alphas.append(probs*np.matmul(alphas[-1],weights))
    alphas = np.array(alphas)
    return alphas

def beta(x,weights,mean,covar):
    betas = []
    N,_ = x.shape
    i = N-1
    while(i>=0):
        probs = []
        for j in range(k):
            probs.append(normal(x[i],mean[j],covar[j]))
        probs = np.array(probs)
        probs = probs.reshape(k,1)
        if(i==N-1):
            temp = np.ones((k,1))
            betas.append(temp)
        else:
            betas.append(np.matmul(weights,betas[-1]*probs))
        i=i-1
    betas = np.array(betas)
    betas = betas[::-1]
    return betas

def gamma(alphas,betas):
    gammas = []
    for i in range(0,len(alphas)):
        gammas.append((alphas[i]*betas[i].reshape(1,k))/np.sum(alphas[-1]))
    gammas = np.array(gammas)
    return gammas

def epsilon(x,alphas,betas,weights,mean,covar):
    epsilons = []
    for i in range(1,len(alphas)):
        j=0
        prob = []
        for j in range(k):
            prob.append(normal(x[i],mean[j],covar[j]))
        prob = np.array(prob)
        prob = prob.reshape(1,k)
        t = np.matmul(np.matmul(weights,alphas[i-1].T),(prob*betas[i].reshape(prob.shape)))/np.sum(alphas[-1])
        epsilons.append(t)
    epsilons = np.array(epsilons)
    return epsilons,prob

#############main hmm training function#############
def hmm_train(train_data,P,A,C,M):
    q_theta=p_temp=A_temp=cov_temp=mu_temp=0
    Len = len(train_data)
    for l in range (Len):
        alphas = alpha(train_data[l],P,A,M,C)
        betas = beta(train_data[l],A,M,C)
        gammas = gamma(alphas,betas)
        epsilons,probs_re = epsilon(train_data[l],alphas,betas,A,M,C)
        P_new = gammas[0]/np.sum(gammas[0])
        gammas=gammas.reshape(N,k)
        M_new = (np.matmul(train_data[l].T,gammas)/np.sum(gammas,axis=0)).T
        A_new = np.sum(epsilons,axis=0)/np.sum(np.sum(epsilons,axis=0),axis=1)
        cov_new=[]
        s = gammas.T
        for i in range(k):
            cov = np.matmul((train_data[l]-M[i]).T,(train_data[l]-M[i]))
            cov1 = np.zeros(cov.shape)
            for j in range(0,len(s[i])):
                cov1 = cov1 + cov*s[i][j]
            cov1 = cov1/np.sum(s[i])
            cov_new.append(cov1)
        cov_new = np.array(cov_new)
        probs_re = np.array(probs_re)
        p_temp += P_new
        A_temp += A_new
        cov_temp += cov_new
        mu_temp += M_new
        q_theta += np.matmul(gammas[0],np.log(P).T) + np.sum(epsilons*np.log(A)) + np.sum(gammas*np.log(probs_re))
    P = p_temp/Len
    A = A_temp/Len
    C = cov_temp/Len
    M = mu_temp/Len
    q_theta = q_theta/Len    
    print(q_theta)
    return P
    return A
    return C
    return M
    count+=1


##########200 iterations for EM to converge#################
count = 0
while(count < 200):
    hmm_train(re_train,P,A,C,M)
    
P_re = P
A_re = A
C_re = C
M_re = M

count = 0
while(count < 200):
    hmm_train(ri_train,P,A,C,M)
   
P_ri = P
A_ri = A
C_ri = C
M_ri = M

#######Test dataset#############
count = 0
rep = []
rip = []
for l in range(len(re_test)):
    alphas = alpha(ri_test[l],P,A_re,M_re,C_re)
    betas = beta(ri_test[l],A_re,M_re,C_re)
    gammas = gamma(alphas,betas)
    epsilons,probs_ri = epsilon(ri_test[l],alphas,betas,A_re,M_re,C_re)
    px = np.sum(np.matmul(alphas,betas))
    rip.append(px)
    alphas = alpha(re_test[l],P,A_re,M_re,C_re)
    betas = beta(re_test[l],A_re,M_re,C_re)
    gammas = gamma(alphas,betas)
    epsilons,probs_re = epsilon(re_test[l],alphas,betas,A_re,M_re,C_re)
    px = np.sum(np.matmul(alphas,betas))
    rep.append(px)

for i in range(len(re_test)):
    if(rep[i]>rip[i]):
        count += 1
print(count/8.0)
