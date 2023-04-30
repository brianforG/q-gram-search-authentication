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


markers=["x","s"]
colors=["black","blue"]
tmpcolors=["k","red","skyblue","grey","lightgreen","yellow"]
labels=["GS$^{2}$","GS$^{2}$-opt"]
dataset_name=["LastName_dataset","FemaleName_dataset","Author_dataset"]
fig1_yticks=[[0, 0.2, 0.4, 0.6, 0.8, 1.0],[0, 0.01, 0.02, 0.03, 0.04, 0.05],[0, 10, 20, 30, 40, 50]]
fig2_yticks=[[0, 0.2, 0.4, 0.6, 0.8, 1.0],[0, 10, 20, 30, 40, 50],[0, 10, 20, 30, 40, 50]]
fig3_yticks=[[0, 0.04, 0.08, 0.12, 0.16, 0.2],[0, 0.004, 0.008, 0.012, 0.016, 0.02],[0, 4, 8, 12, 16, 20]]
Q=2
for dataset_idx in range(len(dataset_name)):
    xs=[range(1,6,1),range(1,6,1)]
    optimizers=["gs_square","e_gs_square"]
    with open("./figure1/VO_Construction_Time(%s).csv"%(dataset_name[dataset_idx]),"r") as f:
        VO_construction_times_1,VO_construction_times_2=[],[]
        VO_construction_sections_1,VO_construction_sections_2=[],[]
        for line in f:
            line=(line.strip()).split(",")
            if "opt" not in line[0] and "GS" in line[0]:
                VO_construction_sections_1.append([float(it) for it in line[2:-1]])
                VO_construction_times_1.append(float(line[-1]))
            elif "opt" in line[0]:
                VO_construction_sections_2.append([float(it) for it in line[2:-1]])
                VO_construction_times_2.append(float(line[-1]))
        VO_construction_times,VO_construction_sections=[VO_construction_times_1,VO_construction_times_2],[VO_construction_sections_1,VO_construction_sections_2]
    with open("./figure1/VO_Size(%s).csv"%(dataset_name[dataset_idx]),"r") as f:
        VO_sizes_1,VO_sizes_2=[],[]
        VO_size_sections_1,VO_size_sections_2=[],[]
        for line in f:
            line=line.strip().split(",")[:6]
            if "opt" not in line[0] and "GS" in line[0]:
                VO_size_sections_1.append([float(it) for it in line[2:-1]])
                VO_sizes_1.append(float(line[-1]))
            elif "opt" in line[0]:
                VO_size_sections_2.append([float(it) for it in line[2:-1]])
                VO_sizes_2.append(float(line[-1]))
        VO_sizes,VO_size_sections=[VO_sizes_1,VO_sizes_2],[VO_size_sections_1,VO_size_sections_2]
    with open("./figure1/VO_Verification_Time(%s).csv"%(dataset_name[dataset_idx]),"r") as f:
        VO_verification_times_1,VO_verification_times_2=[],[]
        VO_verification_sections_1,VO_verification_sections_2=[],[]
        for line in f:
            line=line.strip().split(",")
            if "opt" not in line[0] and "GS" in line[0]:
                VO_verification_sections_1.append([float(it) for it in line[2:-1]])
                VO_verification_times_1.append(float(line[-1]))
            elif "opt" in line[0]:
                VO_verification_sections_2.append([float(it) for it in line[2:-1]])
                VO_verification_times_2.append(float(line[-1]))
        VO_verification_times,VO_verification_sections=[VO_verification_times_1,VO_verification_times_2],[VO_verification_sections_1,VO_verification_sections_2]

    #VO Construction Time
    with plt.style.context(['ieee']):
        for idx in range(len(xs)):
            plt.plot(xs[idx], VO_construction_times[idx], linestyle="-", marker=markers[idx], color=colors[idx], label=labels[idx], linewidth=1, clip_on=False)
        plt.xticks([1, 2, 3, 4, 5])
        plt.yticks(fig1_yticks[dataset_idx])
        plt.tick_params(direction='in',top=True,right=True,labelsize=8)
        plt.xlabel('$k$',fontsize=10)
        plt.ylabel('Time(Second)',fontsize=10)
        plt.xlim(1, 5)
        plt.ylim(0, fig1_yticks[dataset_idx][-1])
        #ax = plt.gca()
        #ax.set_aspect(0.2)
        plt.legend()
        plt.savefig("./figure1/fig1_q=%d_%s.jpg"%(Q,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)

    #VO Size
    plt.clf()
    with plt.style.context(['ieee']):
        for idx in range(len(xs)):
            plt.plot(xs[idx], VO_sizes[idx], linestyle="-", marker=markers[idx], color=colors[idx], label=labels[idx], linewidth=1, clip_on=False)
        plt.xticks([1, 2, 3, 4, 5])
        plt.yticks(fig2_yticks[dataset_idx])
        plt.tick_params(direction='in',top=True,right=True,labelsize=8)
        plt.xlabel('$k$',fontsize=10)
        if dataset_idx==0 or dataset_idx==2:
            plt.ylabel('VO Size(MB)',fontsize=10)
        else:
            plt.ylabel('VO Size(KB)',fontsize=10)
        plt.xlim(1, 5)
        plt.ylim(0, fig2_yticks[dataset_idx][-1])
        #ax = plt.gca()
        #ax.set_aspect(0.15)
        plt.legend()
        plt.savefig("./figure1/fig2_q=%d_%s.jpg"%(Q,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    plt.clf()
    
    #VO Verification Time
    with plt.style.context(['ieee']):
        for idx in range(len(xs)):
            plt.plot(xs[idx], VO_verification_times[idx], linestyle="-", marker=markers[idx], color=colors[idx], label=labels[idx], linewidth=1, clip_on=False)
        plt.xticks([1, 2, 3, 4, 5])
        plt.yticks(fig3_yticks[dataset_idx])
        plt.tick_params(direction='in',top=True,right=True,labelsize=8)
        plt.xlabel('$k$',fontsize=10)
        plt.ylabel('Time(Second)',fontsize=10)
        plt.xlim(1, 5)
        plt.ylim(0, fig3_yticks[dataset_idx][-1])
        #ax = plt.gca()
        #ax.set_aspect(0.5)
        plt.legend()
        plt.savefig("./figure1/fig3_q=%d_%s.jpg"%(Q,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    plt.clf()
    
    #VO Construction Time (section)
    with plt.style.context(['ieee']):
        y=[0,0,0,0,0]
        for idx in range(len(VO_construction_sections[0][0])):
            plt.bar(np.array([it-0.175 for it in [1, 2, 3, 4, 5]]),[it[idx] for it in VO_construction_sections[0]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[0])
            for y_idx in range(len(y)):
                y[y_idx]+=[it[idx] for it in VO_construction_sections[0]][y_idx]
        for idx in range(len(y)):
            plt.text([it-0.175 for it in [1, 2, 3, 4, 5]][idx], y[idx] + 0.0001, labels[0], ha='center', va='bottom', fontsize=3)
        y=[0,0,0,0,0]
        for idx in range(len(VO_construction_sections[1][0])):
            plt.bar(np.array([it+0.175 for it in [1, 2, 3, 4, 5]]),[it[idx] for it in VO_construction_sections[1]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[1])
            for y_idx in range(len(y)):
                y[y_idx]+=[it[idx] for it in VO_construction_sections[1]][y_idx]
        for idx in range(len(y)):
            plt.text([it+0.175 for it in [1, 2, 3, 4, 5]][idx], y[idx] + 0.0001, labels[1], ha='center', va='bottom', fontsize=3)
        plt.xticks([1, 2, 3, 4, 5])
        plt.yticks(fig1_yticks[dataset_idx])
        plt.tick_params(direction='in',top=True,right=True,labelsize=6)
        plt.xlabel('$k$',fontsize=10)
        plt.ylabel('Time(Second)',fontsize=10)
        plt.xlim(0, 6)
        plt.ylim(0, fig1_yticks[dataset_idx][-1])
        plt.legend(['search gram','count sid',"dic MHT","inv MHT"],ncol=4,fontsize=5)
        plt.savefig("./figure1/fig4_q=%d_%s.jpg"%(Q,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    plt.clf()
    
    #VO Size (section)
    with plt.style.context(['ieee']):
        y=[0,0,0,0,0]
        for idx in range(len(VO_size_sections[0][0])):
            plt.bar(np.array([it-0.175 for it in [1, 2, 3, 4, 5]]),[it[idx] for it in VO_size_sections[0]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[0])
            if idx!=len(VO_size_sections[0][0])-1:
                for y_idx in range(len(y)):
                    y[y_idx]+=[it[idx] for it in VO_size_sections[0]][y_idx]
        for idx in range(len(y)):
            plt.text([it-0.175 for it in [1, 2, 3, 4, 5]][idx], y[idx] + 0.001, labels[0], ha='center', va='bottom', fontsize=3)
        y=[0,0,0,0,0]
        for idx in range(len(VO_size_sections[1][0])):
            plt.bar(np.array([it+0.175 for it in [1, 2, 3, 4, 5]]),[it[idx] for it in VO_size_sections[1]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[1])
            if idx!=len(VO_size_sections[1][0])-1:
                for y_idx in range(len(y)):
                    y[y_idx]+=[it[idx] for it in VO_size_sections[1]][y_idx]
        for idx in range(len(y)):
            plt.text([it+0.175 for it in [1, 2, 3, 4, 5]][idx], y[idx] + 0.001, labels[1], ha='center', va='bottom', fontsize=3)
        plt.xticks([1, 2, 3, 4, 5])
        plt.yticks(fig2_yticks[dataset_idx])
        plt.tick_params(direction='in',top=True,right=True,labelsize=6)
        plt.xlabel('$k$',fontsize=10)
        if dataset_idx==0 or dataset_idx==2:
            plt.ylabel('VO Size(MB)',fontsize=10)
        else:
            plt.ylabel('VO Size(KB)',fontsize=10)
        plt.xlim(0, 6)
        plt.ylim(0, fig2_yticks[dataset_idx][-1])
        plt.legend(['dic','inv',"sq"],ncol=3,fontsize=5)
        plt.savefig("./figure1/fig5_q=%d_%s.jpg"%(Q,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    plt.clf()
    
    #VO Verification Time (section)
    with plt.style.context(['ieee']):
        y=[0,0,0,0,0]
        for idx in range(len(VO_verification_sections[0][0])):
            plt.bar(np.array([it-0.175 for it in [1, 2, 3, 4, 5]]),[it[idx] for it in VO_verification_sections[0]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[0])
            if idx!=len(VO_verification_sections[0][0])-1:
                for y_idx in range(len(y)):
                    y[y_idx]+=[it[idx] for it in VO_verification_sections[0]][y_idx]
        for idx in range(len(y)):
            plt.text([it-0.175 for it in [1, 2, 3, 4, 5]][idx], y[idx] + 0.001, labels[0], ha='center', va='bottom', fontsize=3)
        y=[0,0,0,0,0]
        for idx in range(len(VO_verification_sections[1][0])):
            plt.bar(np.array([it+0.175 for it in [1, 2, 3, 4, 5]]),[it[idx] for it in VO_verification_sections[1]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[1])
            if idx!=len(VO_verification_sections[1][0])-1:
                for y_idx in range(len(y)):
                    y[y_idx]+=[it[idx] for it in VO_verification_sections[1]][y_idx]
        for idx in range(len(y)):
            plt.text([it+0.175 for it in [1, 2, 3, 4, 5]][idx], y[idx] + 0.001, labels[1], ha='center', va='bottom', fontsize=3)
        plt.xticks([1, 2, 3, 4, 5])
        plt.yticks(fig3_yticks[dataset_idx])
        plt.tick_params(direction='in',top=True,right=True,labelsize=8)
        plt.xlabel('$k$',fontsize=10)
        plt.ylabel('Time(Second)',fontsize=10)
        plt.xlim(0, 6)
        plt.ylim(0, fig3_yticks[dataset_idx][-1])
        plt.legend(['dic MHT','inv MHT',"Dec S'",'Check gram','Recount sid','Cal edit-dist'],ncol=6,fontsize=3)
        plt.savefig("./figure1/fig6_q=%d_%s.jpg"%(Q,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    plt.clf()

K=1
fig1_yticks=[[0, 0.4, 0.8, 1.2, 1.6, 2.0],[0, 0.02, 0.04, 0.06, 0.08, 0.10],[0, 10, 20, 30, 40, 50]]
fig2_yticks=[[0, 50, 100, 200, 300],[0, 4, 8, 12, 16, 20],[0, 25, 50, 75, 100, 25]]
fig3_yticks=[[0, 0.01, 0.02, 0.03, 0.04, 0.05],[0, 0.001, 0.002, 0.003, 0.004, 0.006],[0, 1, 2, 3, 4, 5]]
for dataset_idx in range(len(dataset_name)):
    xs=[range(2,7,1),range(2,7,1)]
    optimizers=["gs_square","e_gs_square"]
    with open("./figure2/VO_Construction_Time(%s).csv"%(dataset_name[dataset_idx]),"r") as f:
        VO_construction_times_1,VO_construction_times_2=[],[]
        VO_construction_sections_1,VO_construction_sections_2=[],[]
        for line in f:
            line=(line.strip()).split(",")
            if "opt" not in line[0] and "GS" in line[0]:
                VO_construction_sections_1.append([float(it) for it in line[2:-1]])
                VO_construction_times_1.append(float(line[-1]))
            elif "opt" in line[0]:
                VO_construction_sections_2.append([float(it) for it in line[2:-1]])
                VO_construction_times_2.append(float(line[-1]))
        VO_construction_times,VO_construction_sections=[VO_construction_times_1,VO_construction_times_2],[VO_construction_sections_1,VO_construction_sections_2]
    with open("./figure2/VO_Size(%s).csv"%(dataset_name[dataset_idx]),"r") as f:
        VO_sizes_1,VO_sizes_2=[],[]
        VO_size_sections_1,VO_size_sections_2=[],[]
        for line in f:
            line=line.strip().split(",")[:6]
            if "opt" not in line[0] and "GS" in line[0]:
                VO_size_sections_1.append([float(it) for it in line[2:-1]])
                VO_sizes_1.append(float(line[-1]))
            elif "opt" in line[0]:
                VO_size_sections_2.append([float(it) for it in line[2:-1]])
                VO_sizes_2.append(float(line[-1]))
        VO_sizes,VO_size_sections=[VO_sizes_1,VO_sizes_2],[VO_size_sections_1,VO_size_sections_2]
    with open("./figure2/VO_Verification_Time(%s).csv"%(dataset_name[dataset_idx]),"r") as f:
        VO_verification_times_1,VO_verification_times_2=[],[]
        VO_verification_sections_1,VO_verification_sections_2=[],[]
        for line in f:
            line=line.strip().split(",")
            if "opt" not in line[0] and "GS" in line[0]:
                VO_verification_sections_1.append([float(it) for it in line[2:-1]])
                VO_verification_times_1.append(float(line[-1]))
            elif "opt" in line[0]:
                VO_verification_sections_2.append([float(it) for it in line[2:-1]])
                VO_verification_times_2.append(float(line[-1]))
        VO_verification_times,VO_verification_sections=[VO_verification_times_1,VO_verification_times_2],[VO_verification_sections_1,VO_verification_sections_2]
        
    #VO Construction Time
    with plt.style.context(['ieee']):
        for idx in range(len(xs)):
            plt.plot(xs[idx], VO_construction_times[idx], linestyle="-", marker=markers[idx], color=colors[idx], label=labels[idx], linewidth=1, clip_on=False)
        plt.xticks([2, 3, 4, 5, 6])
        plt.yticks(fig1_yticks[dataset_idx])
        plt.tick_params(direction='in',top=True,right=True,labelsize=8)
        plt.xlabel('Length of grams',fontsize=10)
        plt.ylabel('Time(Second)',fontsize=10)
        plt.xlim(2, 6)
        plt.ylim(0, fig1_yticks[dataset_idx][-1])
        plt.legend()
        plt.savefig("./figure2/fig1_k=%d_%s.jpg"%(K,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    plt.clf()
    
    #VO Size
    with plt.style.context(['ieee']):
        for idx in range(len(xs)):
            plt.plot(xs[idx], VO_sizes[idx], linestyle="-", marker=markers[idx], color=colors[idx], label=labels[idx], linewidth=1, clip_on=False)
        plt.xticks([2, 3, 4, 5, 6])
        plt.yticks(fig2_yticks[dataset_idx])
        plt.tick_params(direction='in',top=True,right=True,labelsize=8)
        plt.xlabel('Length of grams',fontsize=10)
        if dataset_idx==2:
            plt.ylabel('VO Size(MB)',fontsize=10)
        else:
            plt.ylabel('VO Size(KB)',fontsize=10)
        plt.xlim(2, 6)
        plt.ylim(0, fig2_yticks[dataset_idx][-1])
        plt.legend()
        plt.savefig("./figure2/fig2_k=%d_%s.jpg"%(K,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    plt.clf()
    
    #VO Verification Time
    with plt.style.context(['ieee']):
        for idx in range(len(xs)):
            plt.plot(xs[idx], VO_verification_times[idx], linestyle="-", marker=markers[idx], color=colors[idx], label=labels[idx], linewidth=1, clip_on=False)
        plt.xticks([2, 3, 4, 5, 6])
        plt.yticks(fig3_yticks[dataset_idx])
        plt.tick_params(direction='in',top=True,right=True,labelsize=8)
        plt.xlabel('Length of grams',fontsize=10)
        plt.ylabel('Time(Second)',fontsize=10)
        plt.xlim(2, 6)
        plt.ylim(0, fig3_yticks[dataset_idx][-1])
        plt.legend()
        plt.savefig("./figure2/fig3_k=%d_%s.jpg"%(K,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    plt.clf()
    
    #VO Construction Time (section)
    with plt.style.context(['ieee']):
        y=[0,0,0,0,0]
        for idx in range(len(VO_construction_sections[0][0])):
            plt.bar(np.array([it-0.175 for it in [2, 3, 4, 5, 6]]),[it[idx] for it in VO_construction_sections[0]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[0])
            if idx!=len(VO_construction_sections[0][0]):
                for y_idx in range(len(y)):
                    y[y_idx]+=[it[idx] for it in VO_construction_sections[0]][y_idx]
        for idx in range(len(y)):
            plt.text([it-0.175 for it in [2, 3, 4, 5, 6]][idx], y[idx] + 0.005, labels[0], ha='center', va='bottom', fontsize=3)
        y=[0,0,0,0,0]
        for idx in range(len(VO_construction_sections[1][0])):
            plt.bar(np.array([it+0.175 for it in [2, 3, 4, 5, 6]]),[it[idx] for it in VO_construction_sections[1]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[1])
            if idx!=len(VO_construction_sections[1][0]):
                for y_idx in range(len(y)):
                    y[y_idx]+=[it[idx] for it in VO_construction_sections[1]][y_idx]
        for idx in range(len(y)):
            plt.text([it+0.175 for it in [2, 3, 4, 5, 6]][idx], y[idx] + 0.005, labels[1], ha='center', va='bottom', fontsize=3)
        plt.xticks([2, 3, 4, 5, 6])
        plt.yticks(fig1_yticks[dataset_idx])
        plt.tick_params(direction='in',top=True,right=True,labelsize=6)
        plt.xlabel('Length of grams',fontsize=10)
        plt.ylabel('Time(Second)',fontsize=10)
        plt.xlim(1, 7)
        plt.ylim(0, fig1_yticks[dataset_idx][-1])
        plt.legend(['search gram','count sid',"dic MHT","inv MHT"],ncol=4,fontsize=5)
        plt.savefig("./figure2/fig4_k=%d_%s.jpg"%(K,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    plt.clf()
    
    #VO Size (section)
    with plt.style.context(['ieee']):
        y=[0,0,0,0,0]
        for idx in range(len(VO_size_sections[0][0])):
            plt.bar(np.array([it-0.175 for it in [2, 3, 4, 5, 6]]),[it[idx] for it in VO_size_sections[0]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[0])
            if idx!=len(VO_size_sections[0][0])-1:
                for y_idx in range(len(y)):
                    y[y_idx]+=[it[idx] for it in VO_size_sections[0]][y_idx]
        for idx in range(len(y)):
            plt.text([it-0.175 for it in [2, 3, 4, 5, 6]][idx], y[idx] + 0.001, labels[0], ha='center', va='bottom', fontsize=3)
        y=[0,0,0,0,0]
        for idx in range(len(VO_size_sections[1][0])):
            plt.bar(np.array([it+0.175 for it in [2, 3, 4, 5, 6]]),[it[idx] for it in VO_size_sections[1]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[1])
            if idx!=len(VO_size_sections[1][0])-1:
                for y_idx in range(len(y)):
                    y[y_idx]+=[it[idx] for it in VO_size_sections[1]][y_idx]
        for idx in range(len(y)):
            plt.text([it+0.175 for it in [2, 3, 4, 5, 6]][idx], y[idx] + 0.001, labels[1], ha='center', va='bottom', fontsize=3)
        plt.xticks([2, 3, 4, 5, 6])
        plt.yticks(fig2_yticks[dataset_idx])
        plt.tick_params(direction='in',top=True,right=True,labelsize=6)
        plt.xlabel('Length of grams',fontsize=10)
        if dataset_idx==2:
            plt.ylabel('VO Size(MB)',fontsize=10)
        else:
            plt.ylabel('VO Size(KB)',fontsize=10)
        plt.xlim(1, 7)
        plt.ylim(0, fig2_yticks[dataset_idx][-1])
        plt.legend(['dic','inv',"sq"],ncol=3,fontsize=5)
        plt.savefig("./figure2/fig5_k=%d_%s.jpg"%(K,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    plt.clf()
    
    #VO Verification Time (section)
    with plt.style.context(['ieee']):
        y=[0,0,0,0,0]
        for idx in range(len(VO_verification_sections[0][0])):
            plt.bar(np.array([it-0.175 for it in [2, 3, 4, 5, 6]]),[it[idx] for it in VO_verification_sections[0]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[0])
            if idx!=len(VO_verification_sections[0][0])-1:
                for y_idx in range(len(y)):
                    y[y_idx]+=[it[idx] for it in VO_verification_sections[0]][y_idx]
        for idx in range(len(y)):
            plt.text([it-0.175 for it in [2, 3, 4, 5, 6]][idx], y[idx] + 0.0001, labels[0], ha='center', va='bottom', fontsize=3)
        y=[0,0,0,0,0]
        for idx in range(len(VO_verification_sections[1][0])):
            plt.bar(np.array([it+0.175 for it in [2, 3, 4, 5, 6]]),[it[idx] for it in VO_verification_sections[1]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[1])
            if idx!=len(VO_verification_sections[1][0])-1:
                for y_idx in range(len(y)):
                    y[y_idx]+=[it[idx] for it in VO_verification_sections[1]][y_idx]
        for idx in range(len(y)):
            plt.text([it+0.175 for it in [2, 3, 4, 5, 6]][idx], y[idx] + 0.0001, labels[1], ha='center', va='bottom', fontsize=3)
        plt.xticks([2, 3, 4, 5, 6])
        plt.yticks(fig3_yticks[dataset_idx])
        plt.tick_params(direction='in',top=True,right=True,labelsize=6)
        plt.xlabel('Length of grams',fontsize=10)
        plt.ylabel('Time(Second)',fontsize=10)
        plt.xlim(1, 7)
        plt.ylim(0, fig3_yticks[dataset_idx][-1])
        plt.legend(['dic MHT','inv MHT',"Dec S'",'Check gram','Recount sid','Cal edit-dist'],ncol=6,fontsize=3)
        plt.savefig("./figure2/fig6_k=%d_%s.jpg"%(K,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    plt.clf()

K=1
Q=2
flags=[["gs_square",False,False,False,False,"snappy"],["gs_square",True,False,False,False,"snappy"],["gs_square",False,True,False,False,"snappy"],["gs_square",False,False,True,False,"snappy"],["gs_square",False,False,False,True,"gzip"],["gs_square",False,False,False,True,"bz2"],["gs_square",False,False,False,True,"lzma"],["gs_square",False,False,False,True,"snappy"]]
labels=["GS$^{2}$","length filtering","representative grams","empty set authentication","gzip compression","bz2 compression","lzma compression","snappy compression"]
for dataset_idx in range(len(dataset_name)):
    with open("./figure3/VO_Size_k=%d_q=%d(%s).csv"%(K,Q,dataset_name[dataset_idx]),"r") as f:
        VO_size_section=[]
        for line in f:
            line=(line.strip()).split(",")
            if "Optimizer" not in line[0]:
                if dataset_idx==2:
                    VO_size_section.append([float(it)/1024 for it in line[1:-1]])
                else:
                    VO_size_section.append([float(it) for it in line[1:-1]])

    with plt.style.context(['ieee']):
        y=[0,0,0,0,0,0,0,0]
        for idx in range(len(VO_size_section[0])):
            plt.bar(np.array(range(1,9,1)),[it[idx] for it in VO_size_section],bottom=y,width=0.4,color=tmpcolors[idx])
            if idx!=len(VO_size_section[0])-1:
                for y_idx in range(len(y)):
                    y[y_idx]+=[it[idx] for it in VO_size_section][y_idx]
        plt.xticks([1,2.8,4,5,6,7,8,9],labels=labels,rotation=-20)
        plt.yticks(fig2_yticks[dataset_idx])
        plt.tick_params(direction='in',top=True,right=True,labelsize=6)
        plt.xlabel('Optimizer',fontsize=10)
        if dataset_idx==2:
            plt.ylabel('VO Size(MB)',fontsize=10)
        else:
            plt.ylabel('VO Size(KB)',fontsize=10)
        plt.xlim(0, 9)
        plt.ylim(0, fig2_yticks[dataset_idx][-1])
        plt.legend(['dic','inv',"sq"],ncol=3,fontsize=5)
        plt.savefig("./figure3/fig7_k=%d_q=%d_%s.jpg"%(K,Q,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    plt.clf()
    

