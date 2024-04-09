from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise
import numpy as np
from collections import  Counter 
import torch
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import floyd_warshall

def get_K_NN_Rho(x,k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(x)
    distances, indices = nbrs.kneighbors(x)  
    return -np.sum(distances,axis = 1),distances, indices 

def get_pre_cluster(x,rho,distances, indices):
    def find_mode(x,index,pre_center,higher):
        count = Counter(pre_center[higher])
        mode_list = []
        mode=0
        for i in count.keys():
            if count[i]>mode:
                mode = count[i]
                mode_list = [i]
            elif count[i] == mode:
                mode_list.extend([i])
        if len(mode_list)<2:
            return mode_list[0]
        temp_list = []
        for i in mode_list:
            temp_list.extend(np.where(pre_center[higher] == i)[0])
        temp_list = higher[np.array(temp_list)]
        min_dis = np.inf
        for i in temp_list:
            dist=np.linalg.norm(x[index]-x[i])
            if dist<min_dis:
                min_dis=dist
                mode_index=i
        return pre_center[mode_index]
    n=len(x)
    pre_list=[]
    for i in range(n):
        pre_list.append([])
    pre_center=np.ones(n,dtype=np.int32)*-1
    pre_arrow=np.ones(n,dtype=np.int32)*-1
    sort_rho=np.argsort(-rho)
    for i in sort_rho:
        higher=indices[i][np.where(rho[indices[i]]>rho[i])].copy()
        pre_list[i].extend(list(higher.copy()))
        if len(higher)==0:
            pre_center[i]=i
            continue
        mode=find_mode(x,i,pre_center,higher)
        pre_center[i]=mode
        pre_arrow[i]=higher[np.where(pre_center[higher]==mode)][0]
    for i in np.unique(pre_center):
        if len(np.where(pre_center==i)[0])<2:
            temp=list(indices[i])
            temp.remove(i)
            pre_center[i]=pre_center[temp[0]]
            pre_arrow[i]=pre_arrow[temp[0]]
    return pre_center, pre_arrow #,pre_list

def knn_dist(data, n_neighbors):

    dist = pairwise.euclidean_distances(data)
    dist[dist < 0] = 0
    n = dist.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        index_ = np.argsort(dist[i])[1:1+n_neighbors]
        W[i, index_] = 1
        W[index_, i] = 1

    return W,np.max(dist)

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(),beta=1,alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  
    return dist

def cple(x,y,k = 10,alpha = 5,beta = 5):        
    rho,distances, indices = get_K_NN_Rho(x,k)
    pre_center, pre_arrow = get_pre_cluster(x,rho,distances, indices)
    t = list(np.unique(pre_center))
    target_dim = len(np.unique(y))+1

    dist_matrix = pairwise.euclidean_distances(x)  
    core_index_matrix = np.zeros_like(dist_matrix) 
    core_index_matrix[np.ix_(t,t)] = 1
    core_dist_matrix = torch.tensor(dist_matrix*core_index_matrix).cuda() 
    core_index_matrix = torch.tensor(core_index_matrix).cuda()
    knn_d,max_dist = knn_dist(x,k)  
    knn_matris = torch.tensor(knn_d).cuda() 

    knn_d_02 = np.zeros_like(knn_d) 
    for i in range(len(pre_center)):
        knn_d_02[i][pre_center[i]] = 1 

    W_TT = torch.tensor(pairwise.rbf_kernel(X=x,gamma=1/((max_dist*0.2)**2))*knn_d).cuda() 
    W_TC = torch.tensor(pairwise.rbf_kernel(X=x,gamma=1/((max_dist*0.2)**2))*knn_d_02).cuda() 
    W_Component = W_TT + alpha*W_TC               
   
    D_b = torch.tensor(np.zeros_like(dist_matrix)).cuda()
    for i in range(x.shape[0]):
        D_b[i][i] = torch.sum(W_Component[i])
    D_b = D_b.to(torch.float32)
    
    L_b = (D_b-W_Component).to(torch.float32)

    dist_core = torch.tensor(dist_matrix).cuda()*core_index_matrix
    W_CC1 = (torch.mul(torch.exp(-(dist_core**2)/((max_dist*0.2)**2)),core_index_matrix))
    for i in np.unique(pre_center):
        W_CC1[i][i] = 0
    D_core = torch.tensor(np.zeros_like(dist_matrix)).cuda()
    for i in range(x.shape[0]):
        D_core[i][i] = torch.sum(W_CC1[i])
    D_core = D_core.to(torch.float32)

    dist_neighb = dist_matrix*knn_d
    dis_02 = floyd_warshall(dist_neighb, directed=False)
    pre_W_CC2 = np.exp(-dis_02**2)

    for i in range(len(knn_d)):
        pre_W_CC2[i][i] = 0
    pre_W_CC2 = torch.tensor(pre_W_CC2).cuda()
    W_CC2 = pre_W_CC2*core_index_matrix

    W_core = W_CC1 + beta*W_CC2                                            

    D_c = torch.tensor(np.zeros_like(dist_matrix)).cuda()
    for i in range(W_core.shape[0]):
        D_c[i][i] = torch.sum(W_core[i])   
    L_c = (D_c-W_core).to(torch.float32)

    lamda =  torch.rand(target_dim,requires_grad=True,device='cuda')
    lamda_matrix = torch.diag(lamda)
    y_low = torch.rand([len(x),target_dim],requires_grad=True,device='cuda') 
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam([y_low,lamda], lr = 1e-3)
    loss_new = torch.tensor(0)
    
    i = 0
    while(True):
        low_dist_matrix = euclidean_dist(y_low,y_low).cuda()**2 
        low_knn_dist_matrix = torch.mul(low_dist_matrix,knn_matris) 
        low_matris = torch.mul(low_dist_matrix,W_Component)
        loss_f1 = loss_fn(low_matris,torch.zeros_like(low_knn_dist_matrix))
        low_matris_02 = torch.mul(low_dist_matrix,W_core)    
        loss_f2 = loss_fn(low_matris_02,torch.zeros_like(low_knn_dist_matrix))
        y_low_T = (y_low).transpose(0,1)
        y_low_T_D = torch.mm(y_low_T,D_core)
        low_lagrange = torch.mm(y_low_T_D,y_low)
        I = (torch.tensor(np.eye(target_dim)).cuda()).to(torch.float32)
        f3 = (low_lagrange-I)**2
        loss_f3 = loss_fn(torch.mm(lamda_matrix,f3),torch.zeros_like(I))
        loss = loss_f1*1+1*loss_f2+loss_f3*1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%5000==0:
            print(loss.item())
        if torch.abs(loss-loss_new)<0.0000001:
            break
        loss_new = loss
        i = i+1
    
    re = y_low.detach().cpu().numpy()

    y_last = KMeans(n_clusters=len(np.unique(y))).fit_predict(re)
  
    return y_last
