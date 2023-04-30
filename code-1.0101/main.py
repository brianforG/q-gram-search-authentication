from dataOwner import *
from cloudServerProvider import *
from user import *
from MHT import *
from tqdm import tqdm
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scienceplots
import sys
sys.setrecursionlimit(100000)
plt.rcParams['text.usetex'] = True

def read_txt(file_path):
    with open(file_path,"r",encoding="utf-8") as f:
        corpus,frequency,rank=[],[],[]
        for i in f.readlines():
            #print(i.strip().split(" "))
            c,f,_,r=[j for j in i.strip().split(" ") if j!=""]
            corpus.append(c.lower())
            frequency.append(f)
            rank.append(r)
    return corpus,frequency,rank
    
def read_author(file_path):
    with open(file_path,"r",encoding="utf-8") as f:
        corpus=[]
        for i in f.readlines():
            #print(i.strip().split(" "))
            c=i.strip()
            corpus.append(c.lower())
    return corpus

def read_querys(file_path):
    with open(file_path,"r",encoding="utf-8") as f:
        querys=[]
        for i in f.readlines():
            querys.append(i.strip().split("\t"))
        querys=[j for i in querys for j in i]
    return querys
dataset2,frequency2,rank2=read_txt("./data/dist.female.first.txt")
query_list=[read_querys("./data/query_strings(FemaleName_dataset).txt")]

# DO=dataOwner(dataset,q=2,optimizer="e_gs_square")
# DO.statistical()
# assert(1==2)

markers=["x","s"]
colors=["black","blue"]
tmpcolors=["k","red","skyblue","grey","lightgreen","yellow"]
labels=["GS$^{2}$","GS$^{2}$-opt"]
dataset_name=["LastName_dataset","FemaleName_dataset","Author_dataset"]
datasets=[dataset2]
fig1_yticks=[[0, 0.2, 0.4, 0.6, 0.8, 1.2],[0, 0.02, 0.03, 0.04, 0.05, 0.06],[0, 10, 20, 30, 40, 50]]
fig2_yticks=[[0, 0.2, 0.4, 0.6, 0.8, 1.0],[0, 4, 8, 12, 16, 20],[0, 10, 20, 30, 40, 50]]
fig3_yticks=[[0, 0.1, 0.2, 0.3, 0.4, 0.5],[0, 0.002, 0.004, 0.006, 0.008, 0.010],[0, 4, 8, 12, 16, 20]]
Q=2
for dataset_idx in range(len(datasets)):
    dataset=datasets[dataset_idx]
    querys=query_list[dataset_idx]
    optimizers=["gs_square","e_gs_square"]
    ks=range(1,3,1)
    xs,VO_construction_times,VO_construction_sections,VO_sizes,VO_size_sections,VO_verification_times,VO_verification_sections,CSTR_nums,RG_FLAGs=[],[],[],[],[],[],[],[],[]
    for optimizer in optimizers:
        x,VO_construction_time,VO_construction_section,VO_size,VO_size_section,VO_verification_time,VO_verification_section,CSTR_num,RG_FLAG=[],[],[],[],[],[],[],[],[]
        DO=dataOwner(dataset,q=Q,optimizer=optimizer)
        D,q,S_dic,S_inv=DO.outsourced()
        CSP=cloudServerProvider(D,q,S_dic,S_inv,optimizer=optimizer)
        
        for k in ks:
            U=user(querys,[k for _ in range(len(querys))],q)
            VO_ct,VO_cs,VO_si,VO_ss,VO_at,VO_as,cstr_num,RG_NUM=0,[0,0,0,0],0,[0,0,0],0,[0,0,0,0,0,0],[],0
            ineff=0
            for cnt in tqdm(range(len(querys))):
                s_q,k=U.get_query()
                if len(s_q)-q+1-k*q<=0:
                    ineff+=1
                    continue
                #print("User query string='%s' with k=%d"%(s_q,k))
                R,S_dic,VO_dic,S_inv,VO_inv,VO_sq,VO_t1,cstring_num,rg_flag=CSP.query(s_q,k)
                cstr_num.append(cstring_num)

            print("Query invalidation num:%d"%(ineff))
