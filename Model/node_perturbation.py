import numpy as np
import networkx as nx
from tqdm import tqdm
import copy

def new_generate_beta(lambda2):
    # you could set a small gap to aacclerate computing!
    gap = 100
    a = list(np.linspace(0, 5 / lambda2, gap))
    #a = list(np.linspace(0, 1 / lambda2, gap*50))
    b = list(np.linspace(a[-1], 10 / lambda2, gap))
    return a[0:-1] + b

def VertexEnt(G, belta=None, perturb_strategy='default', printLog=False):
    """
    近似计算的方法计算节点纠缠度
    :param G:
    :return:
    """
    # nodelist: list, optional:
    # The rows and columns are ordered according to the nodes in nodelist.
    # If nodelist is None, then the ordering is produced by G.nodes().
    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).todense()
    assert 0 in G, "Node 0 should be in the input graph!"
    assert np.allclose(A, A.T), "adjacency matrix should be symmetric"
    L = np.diag(np.array(sum(A)).flatten()) - A
    N = G.number_of_nodes()

    eigValue,eigVector = np.linalg.eigh(L)
    print("Finish calucating eigen values!")
    eigValue = eigValue.real
    eigVector = eigVector.real

    # sort_idx = np.argsort(eigValue)
    # eigValue=eigValue[sort_idx]
    # eigVector=eigVector[:,sort_idx]

    if printLog:
        print(eigValue)

    if belta is None:
        #belta = generate_Belta(eigValue)
        num_components = nx.number_connected_components(G)
        print(f"Tere are {num_components} components.")
        belta = new_generate_beta(eigValue[num_components])

    # %%
    S = np.zeros(len(belta))
    for i in range(0, len(belta)):
        b = belta[i]
        Z_ = np.sum(np.exp(eigValue * -b))
        t = -b * eigValue
        S[i] = -sum(np.exp(t) * (t / np.log(2) - np.log2(Z_)) / Z_)

    print("Finish calucating spectral entropy!")
    if printLog:
        print(S)

    #%%
    lambda_ral = np.zeros((N,N))
    if perturb_strategy == 'default':
        for v_id in tqdm(range(0, N), desc="Computing eigenValues", unit="node"):
            #dA=np.zeros((N,N))
            neibour = list(G.neighbors(v_id))
            kx = G.degree(v_id)
            A_loc = A[neibour][:,neibour]
            N_loc = kx+np.sum(A_loc)/2
            weight=2*N_loc/kx/(kx+1)
            if weight == 1:
                lambda_ral[v_id] = eigValue
            else:
                neibour.append(v_id)
                neibour = sorted(neibour)
                dA = weight-A[neibour][:,neibour]
                dA = dA - np.diag([weight]*(kx+1))
                dL = np.diag(np.array(sum(dA)).flatten()) - dA
                for j in range(0,N):
                    t__= eigVector[neibour,j].T@dL@eigVector[neibour,j]
                    if isinstance(t__, float):
                        lambda_ral[v_id, j] = eigValue[j] + t__
                    else:
                        lambda_ral[v_id, j] = eigValue[j] + t__[0,0]
    elif perturb_strategy == 'remove':
        for v_id in tqdm(range(0, N), desc="Computing eigenValues for removed networks", unit="node"):
            neibour = list(G.neighbors(v_id))
            pt_A = copy.deepcopy(A)
            pt_A[v_id, :] = 0
            pt_A[:, v_id] = 0
            dA = pt_A - A
            dA = dA[neibour][:,neibour]
            dL = np.diag(np.array(sum(dA)).flatten()) - dA
            for j in range(0, N):
                lambda_ral[v_id, j] = eigValue[j] + (eigVector[neibour, j].T @ dL @ eigVector[neibour, j])[0, 0]

    #%%
    E=np.zeros((len(belta),N))
    for x in tqdm(range(0, N), desc="Searching minium entanglement", unit="node"):
        xl_=lambda_ral[x,:]
        for i in range(0,len(belta)):
            b=belta[i]
            Z_=np.sum(np.exp(-b*xl_))
            t = -b *xl_
            E[i,x] = -sum(np.exp(t) * (t / np.log(2) - np.log2(Z_)) / Z_) - S[i]

    VE = np.min(E, axis=0)
    mean_tau = np.mean(np.array(belta)[np.argmin(E, axis=0)])
    print(f"VE mean_tau={mean_tau}")
    return VE

#calculate node perturbation
#ve = VertexEnt(Graph)
