from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.io import mmread,mminfo
from scipy.sparse import csr_matrix
import sys
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community
import random, math
import numpy as np
from sklearn.manifold import TSNE
from sklearn.manifold._t_sne import trustworthiness
from networkx.algorithms.community import greedy_modularity_communities
from matplotlib.pyplot import cm
from matplotlib import collections as mc
import warnings
warnings.filterwarnings("ignore")

def readEmbeddings(filename, nodes, dim):
    embfile = open(filename, "r")
    firstline = embfile.readline()
    N = int(firstline.strip().split()[0])
    X = [[0]*dim for i in range(nodes)]
    for line in embfile.readlines():
        tokens = line.strip().split()
        nodeid = int(tokens[0])-1
        x = []
        for j in range(1, len(tokens)):
            t = float(tokens[j])
            x.append(t)
        X[nodeid] = x
    embfile.close()
    print("Size of X:", len(X))
    return np.array(X)

def readEmbeddingsFA2(filename, nodes, dim):
    embfile = open(filename, "r")
    X = [[0]*dim for i in range(nodes)]
    i = 0
    for line in embfile.readlines():
        tokens = line.strip().split()
        x = []
        for j in range(0, len(tokens)):
            t = float(tokens[j])
            x.append(t)
        X[i] = x
        i += 1
    embfile.close()
    print("Size of X:", len(X))
    return np.array(X)

def readEmbeddingsHARP(filename, nodes, dim):
    dX = np.load(filename)
    print("Size of X:", len(dX))
    return dX

def readBinEmbeddings(filename, dim):
    embfile = open(filename, "r")
    D = np.fromfile(filename, np.float32)
    length = D.shape[0]
    embd_shape = [int(length / dim), dim]
    D = D.reshape(embd_shape)
    X = []
    for line in D:
        x = []
        for v in line:
            x.append(v)
        X.append(x)
    embfile.close()
    print("Size of X:", len(X))
    return np.array(X)

def readgroundtruth(truthlabelsfile, N):
    Yd = dict()
    distinctlabels = set()
    lfile = open(truthlabelsfile)
    arrY = [-1 for i in range(N)]
    for line in lfile.readlines():
        tokens = line.strip().split()
        node = int(tokens[0])-1
        label = int(tokens[1])
        arrY[node] = label
        if label in Yd:
            tempy = Yd[label]
            tempy.append(node)
            Yd[label] = tempy
        else:
            Yd[label] = [node]
        distinctlabels.add(label)
    lfile.close()
    return Yd, len(distinctlabels), np.array(arrY)

import community as commm
def drawGraphc(G, X, comm, nl, algo1="Graph"):
    gridsize = (1, 1)
    fig = plt.figure(figsize=(8, 5))
    axIN = plt.subplot2grid(gridsize, (0, 0))
    plt.axis('off')
    axIN.set_xlim(min(X[:,0]), max(X[:,0]))
    axIN.set_ylim(min(X[:,1]), max(X[:,1]))
    linesIN = []
    e = 0
    print("Cluster:",len(comm))
    mycolors = cm.rainbow(np.linspace(0,1,nl+2))
    gd = dict()
    for com in comm:
        for node in list(comm[com]):
            gd[node] = com
            plt.scatter(X[node][0], X[node][1], s=2, color = mycolors[com])
    plt.axis('off')
    plt.savefig(algo1+'_vis.pdf')

def drawGraph(G, X, algo1="Graph"):
    g = nx.Graph(G)
    comm = community.greedy_modularity_communities(g)
    gridsize = (1, 1)
    fig = plt.figure(figsize=(8, 5))
    axIN = plt.subplot2grid(gridsize, (0, 0))
    plt.axis('off')
    axIN.set_xlim(min(X[:,0]), max(X[:,0]))
    axIN.set_ylim(min(X[:,1]), max(X[:,1]))
    linesIN = []
    e = 0
    print("Cluster:",len(comm))
    mycolors = cm.rainbow(np.linspace(0,1,len(comm)))
    gd = dict()
    for com in range(len(comm)):
        for node in list(comm[com]):
            gd[node] = com
            plt.scatter(X[node][0], X[node][1], s=2, color = mycolors[com])
    plt.axis('off')
    modularity = commm.community_louvain.modularity(gd, g)
    print("Modularity:", algo1, "=", modularity)
    plt.savefig(algo1+'_vis.pdf')

filename = sys.argv[1]
G = mmread(filename)
graph = nx.Graph(G)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics

if sys.argv[2] == "1":
    print("Running native...")
    X = readEmbeddings(sys.argv[3],  mminfo(filename)[0], int(sys.argv[4]))
elif sys.argv[2] == "5":
    X = readEmbeddingsHARP(sys.argv[3], mminfo(filename)[0], int(sys.argv[4]))
elif sys.argv[2] == "2":
    X = readEmbeddingsFA2(sys.argv[3], mminfo(filename)[0], int(sys.argv[4]))
else:
    X = readBinEmbeddings(sys.argv[3], int(sys.argv[4]))

labs, l, gy = readgroundtruth(sys.argv[5], mminfo(filename)[0])
algoname = sys.argv[6]
print("Running TSNE")
X_f = X
drawGraphc(G, X_f, labs, l, algoname)
shil = metrics.silhouette_score(X_f, gy)
davd = metrics.davies_bouldin_score(X_f, gy)

print("silhouette:", shil, "davies_bouldin:", davd)

print("Visualization complete!")
