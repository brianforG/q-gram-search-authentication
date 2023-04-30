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

dataset1,frequency1,rank1=read_txt("./data/dist.all.last.txt")
dataset2,frequency2,rank2=read_txt("./data/dist.female.first.txt")
dataset3=read_author("./data/author-processed.txt")
query_list=[read_querys("./data/query_strings(LastName_dataset).txt"),read_querys("./data/query_strings(FemaleName_dataset).txt"),read_querys("./data/query_strings(Author_dataset).txt")]

# DO=dataOwner(dataset,q=2,optimizer="e_gs_square")
# DO.statistical()
# assert(1==2)

markers=["x","s"]
colors=["black","blue"]
tmpcolors=["k","red","skyblue","grey","lightgreen","yellow"]
labels=["GS$^{2}$","GS$^{2}$-opt"]
dataset_name=["LastName_dataset","FemaleName_dataset","Author_dataset"]
datasets=[dataset1,dataset2,dataset3]
fig1_yticks=[[0, 0.2, 0.4, 0.6, 0.8, 1.2],[0, 0.02, 0.03, 0.04, 0.05, 0.06],[0, 10, 20, 30, 40, 50]]
fig2_yticks=[[0, 0.2, 0.4, 0.6, 0.8, 1.0],[0, 4, 8, 12, 16, 20],[0, 10, 20, 30, 40, 50]]
fig3_yticks=[[0, 0.1, 0.2, 0.3, 0.4, 0.5],[0, 0.002, 0.004, 0.006, 0.008, 0.010],[0, 4, 8, 12, 16, 20]]
Q=2
# for dataset_idx in range(len(datasets)):
    # dataset=datasets[dataset_idx]
    # querys=query_list[dataset_idx]
    # optimizers=["gs_square","e_gs_square"]
    # ks=range(1,6,1)
    # xs,VO_construction_times,VO_construction_sections,VO_sizes,VO_size_sections,VO_verification_times,VO_verification_sections,CSTR_nums,RG_FLAGs=[],[],[],[],[],[],[],[],[]
    # for optimizer in optimizers:
        # x,VO_construction_time,VO_construction_section,VO_size,VO_size_section,VO_verification_time,VO_verification_section,CSTR_num,RG_FLAG=[],[],[],[],[],[],[],[],[]
        # DO=dataOwner(dataset,q=Q,optimizer=optimizer)
        # D,q,S_dic,S_inv=DO.outsourced()
        # CSP=cloudServerProvider(D,q,S_dic,S_inv,optimizer=optimizer)
        
        # for k in ks:

            # #print("\n-----------------data outsource start--------------------")
            
            # # print("CSP get D(%d): %s......"%(len(D),D[:8]))
            # # print("CSP get q:",q)
            # # print("CSP get S_dic: %s......"%(S_dic[:64]))
            # # print("CSP get S_inv: %s......"%(S_inv[:64]))
            # # print("-----------------data outsource end----------------------\n")

            # #print("-------------------User query start----------------------")
            # U=user(querys,[k for _ in range(len(querys))],q)
            # VO_ct,VO_cs,VO_si,VO_ss,VO_at,VO_as,cstr_num,RG_NUM=0,[0,0,0,0],0,[0,0,0],0,[0,0,0,0,0,0],[],0
            # ineff=0
            # for cnt in tqdm(range(len(querys))):
                # s_q,k=U.get_query()
                # if len(s_q)-q+1-k*q<=0:
                    # ineff+=1
                    # continue
                # #print("User query string='%s' with k=%d"%(s_q,k))
                # R,S_dic,VO_dic,S_inv,VO_inv,VO_sq,VO_t1,cstring_num,rg_flag=CSP.query(s_q,k)
                # cstr_num.append(cstring_num)
                # if rg_flag:
                    # RG_NUM+=1
                # VO_s=(len(VO_dic)+len(VO_inv)+len(VO_sq))/1024
                # VO_ct+=sum(VO_t1)
                # for voidx in range(len(VO_t1)):
                    # VO_cs[voidx]+=VO_t1[voidx]
                # VO_si+=VO_s
                # VO_ss[0]+=len(VO_dic)/1024
                # VO_ss[1]+=len(VO_inv)/1024
                # VO_ss[2]+=len(VO_sq)/1024
                # # print("VO Construction Time = %.3fs"%(VO_t1))
                # # print("VO Size = %.2fKB"%(VO_s))
                # # print("-------------------User query end------------------------\n")
        
                # # print("---------------User authentication start-----------------")
                # #authentication phase
                # U.acceptPublicKey(DO.getPublicKey())
                # #print("User get PublicKey from DO")
                # AU_FLAG,VO_t2=U.authenticate(s_q,k,R,S_dic,VO_dic,S_inv,VO_inv,VO_sq,dataset_name[dataset_idx])
                # #print("VO Authentication Result:",AU_FLAG)
                # assert(AU_FLAG==True)
                # VO_at+=sum(VO_t2)
                # for voidx in range(len(VO_t2)):
                    # VO_as[voidx]+=VO_t2[voidx]
                # #print("---------------User authentication end-------------------\n")
            # print("Query invalidation num:%d"%(ineff))
            # assert(ineff<len(querys))
            # VO_construction_time.append(VO_ct/(len(querys)-ineff))
            # VO_construction_section.append([it/(len(querys)-ineff) for it in VO_cs])
            # if dataset_idx==0 or dataset_idx==2:
                # VO_size.append(((VO_si/(len(querys)-ineff)))/1024)
                # VO_size_section.append([(it/(len(querys)-ineff))/1024 for it in VO_ss])
            # else:
                # VO_size.append(((VO_si/(len(querys)-ineff))))
                # VO_size_section.append([it/(len(querys)-ineff) for it in VO_ss])
            # VO_verification_time.append(VO_at/(len(querys)-ineff))
            # VO_verification_section.append([it/(len(querys)-ineff) for it in VO_as])
            # CSTR_num.append(cstr_num)
            # RG_FLAG.append(RG_NUM)
        # xs.append(ks)
        # VO_construction_times.append(VO_construction_time)
        # VO_construction_sections.append(VO_construction_section)
        # VO_sizes.append(VO_size)
        # VO_size_sections.append(VO_size_section)
        # VO_verification_times.append(VO_verification_time)
        # VO_verification_sections.append(VO_verification_section)
        # CSTR_nums.append(CSTR_num)
        # RG_FLAGs.append(RG_FLAG)

    # #VO Construction Time
    # with plt.style.context(['ieee']):
        # for idx in range(len(xs)):
            # plt.plot(xs[idx], VO_construction_times[idx], linestyle="-", marker=markers[idx], color=colors[idx], label=labels[idx], linewidth=1, clip_on=False)
        # plt.xticks([1, 2, 3, 4, 5])
        # plt.yticks(fig1_yticks[dataset_idx])
        # plt.tick_params(direction='in',top=True,right=True,labelsize=8)
        # plt.xlabel('$k$',fontsize=10)
        # plt.ylabel('Time(Second)',fontsize=10)
        # plt.xlim(1, 5)
        # plt.ylim(0, fig1_yticks[dataset_idx][-1])
        # #ax = plt.gca()
        # #ax.set_aspect(0.2)
        # plt.legend()
        # plt.savefig("./figure1/fig1_q=%d_%s.jpg"%(Q,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)

    # #VO Size
    # plt.clf()
    # with plt.style.context(['ieee']):
        # for idx in range(len(xs)):
            # plt.plot(xs[idx], VO_sizes[idx], linestyle="-", marker=markers[idx], color=colors[idx], label=labels[idx], linewidth=1, clip_on=False)
        # plt.xticks([1, 2, 3, 4, 5])
        # plt.yticks(fig2_yticks[dataset_idx])
        # plt.tick_params(direction='in',top=True,right=True,labelsize=8)
        # plt.xlabel('$k$',fontsize=10)
        # if dataset_idx==0 or dataset_idx==2:
            # plt.ylabel('VO Size(MB)',fontsize=10)
        # else:
            # plt.ylabel('VO Size(KB)',fontsize=10)
        # plt.xlim(1, 5)
        # plt.ylim(0, fig2_yticks[dataset_idx][-1])
        # #ax = plt.gca()
        # #ax.set_aspect(0.15)
        # plt.legend()
        # plt.savefig("./figure1/fig2_q=%d_%s.jpg"%(Q,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    # plt.clf()
    
    # #VO Verification Time
    # with plt.style.context(['ieee']):
        # for idx in range(len(xs)):
            # plt.plot(xs[idx], VO_verification_times[idx], linestyle="-", marker=markers[idx], color=colors[idx], label=labels[idx], linewidth=1, clip_on=False)
        # plt.xticks([1, 2, 3, 4, 5])
        # plt.yticks(fig3_yticks[dataset_idx])
        # plt.tick_params(direction='in',top=True,right=True,labelsize=8)
        # plt.xlabel('$k$',fontsize=10)
        # plt.ylabel('Time(Second)',fontsize=10)
        # plt.xlim(1, 5)
        # plt.ylim(0, fig3_yticks[dataset_idx][-1])
        # #ax = plt.gca()
        # #ax.set_aspect(0.5)
        # plt.legend()
        # plt.savefig("./figure1/fig3_q=%d_%s.jpg"%(Q,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    # plt.clf()
    
    # #VO Construction Time (section)
    # df= pd.DataFrame(columns=("method",'k','search gram','count sid',"dic MHT","inv MHT", 'total time (s)'))
    # for K in range(len(VO_construction_sections[0])):
        # col={"method":[labels[0]],'k':[K+1],'search gram':[VO_construction_sections[0][K][0]],'count sid':[VO_construction_sections[0][K][1]],"dic MHT":[VO_construction_sections[0][K][2]],"inv MHT":[VO_construction_sections[0][K][3]],'total time (s)':[sum(VO_construction_sections[0][K])]}
        # df=df.append(pd.DataFrame(col),ignore_index=True)
    # for K in range(len(VO_construction_sections[1])):
        # col={"method":[labels[1]],'k':[K+1],'search gram':[VO_construction_sections[1][K][0]],'count sid':[VO_construction_sections[1][K][1]],"dic MHT":[VO_construction_sections[1][K][2]],"inv MHT":[VO_construction_sections[1][K][3]],'total time (s)':[sum(VO_construction_sections[1][K])]}
        # df=df.append(pd.DataFrame(col),ignore_index=True)
    # df.to_csv("./figure1/VO_Construction_Time(%s).csv"%(dataset_name[dataset_idx]),index=None)
    # with plt.style.context(['ieee']):
        # y=[0,0,0,0,0]
        # for idx in range(len(VO_construction_sections[0][0])):
            # plt.bar(np.array([it-0.175 for it in [1, 2, 3, 4, 5]]),[it[idx] for it in VO_construction_sections[0]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[0])
            # for y_idx in range(len(y)):
                # y[y_idx]+=[it[idx] for it in VO_construction_sections[0]][y_idx]
        # for idx in range(len(y)):
            # plt.text([it-0.175 for it in [1, 2, 3, 4, 5]][idx], y[idx] + 0.0001, labels[0], ha='center', va='bottom', fontsize=3)
        # y=[0,0,0,0,0]
        # for idx in range(len(VO_construction_sections[1][0])):
            # plt.bar(np.array([it+0.175 for it in [1, 2, 3, 4, 5]]),[it[idx] for it in VO_construction_sections[1]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[1])
            # for y_idx in range(len(y)):
                # y[y_idx]+=[it[idx] for it in VO_construction_sections[1]][y_idx]
        # for idx in range(len(y)):
            # plt.text([it+0.175 for it in [1, 2, 3, 4, 5]][idx], y[idx] + 0.0001, labels[1], ha='center', va='bottom', fontsize=3)
        # plt.xticks([1, 2, 3, 4, 5])
        # plt.yticks(fig1_yticks[dataset_idx])
        # plt.tick_params(direction='in',top=True,right=True,labelsize=6)
        # plt.xlabel('$k$',fontsize=10)
        # plt.ylabel('Time(Second)',fontsize=10)
        # plt.xlim(0, 6)
        # plt.ylim(0, fig1_yticks[dataset_idx][-1])
        # plt.legend(['search gram','count sid',"dic MHT","inv MHT"],ncol=4,fontsize=5)
        # plt.savefig("./figure1/fig4_q=%d_%s.jpg"%(Q,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    # plt.clf()
    
    # #VO Size (section)
    # if dataset_idx==0 or dataset_idx==2:
        # size_unit='total size (MB)'
    # else:
        # size_unit='total size (KB)'
    # df= pd.DataFrame(columns=("method",'k','dic','inv',"sq", size_unit, "Cstring num each query"))
    # for K in range(len(VO_size_sections[0])):
        # col={"method":[labels[0]],'k':[K+1],'dic':[VO_size_sections[0][K][0]],'inv':[VO_size_sections[0][K][1]],"sq":[VO_size_sections[0][K][2]],size_unit:[sum(VO_size_sections[0][K])],"Cstring num each query":[CSTR_nums[0][K]]}
        # df=df.append(pd.DataFrame(col),ignore_index=True)
    # for K in range(len(VO_size_sections[1])):
        # col={"method":[labels[1]],'k':[K+1],'dic':[VO_size_sections[1][K][0]],'inv':[VO_size_sections[1][K][1]],"sq":[VO_size_sections[1][K][2]],size_unit:[sum(VO_size_sections[1][K])],"Cstring num each query":[CSTR_nums[1][K]],"representative_grams":[RG_FLAGs[1][K]]}
        # df=df.append(pd.DataFrame(col),ignore_index=True)
    # df.to_csv("./figure1/VO_Size(%s).csv"%(dataset_name[dataset_idx]),index=None)
    # with plt.style.context(['ieee']):
        # y=[0,0,0,0,0]
        # for idx in range(len(VO_size_sections[0][0])):
            # plt.bar(np.array([it-0.175 for it in [1, 2, 3, 4, 5]]),[it[idx] for it in VO_size_sections[0]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[0])
            # if idx!=len(VO_size_sections[0][0])-1:
                # for y_idx in range(len(y)):
                    # y[y_idx]+=[it[idx] for it in VO_size_sections[0]][y_idx]
        # for idx in range(len(y)):
            # plt.text([it-0.175 for it in [1, 2, 3, 4, 5]][idx], y[idx] + 0.001, labels[0], ha='center', va='bottom', fontsize=3)
        # y=[0,0,0,0,0]
        # for idx in range(len(VO_size_sections[1][0])):
            # plt.bar(np.array([it+0.175 for it in [1, 2, 3, 4, 5]]),[it[idx] for it in VO_size_sections[1]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[1])
            # if idx!=len(VO_size_sections[1][0])-1:
                # for y_idx in range(len(y)):
                    # y[y_idx]+=[it[idx] for it in VO_size_sections[1]][y_idx]
        # for idx in range(len(y)):
            # plt.text([it+0.175 for it in [1, 2, 3, 4, 5]][idx], y[idx] + 0.001, labels[1], ha='center', va='bottom', fontsize=3)
        # plt.xticks([1, 2, 3, 4, 5])
        # plt.yticks(fig2_yticks[dataset_idx])
        # plt.tick_params(direction='in',top=True,right=True,labelsize=6)
        # plt.xlabel('$k$',fontsize=10)
        # if dataset_idx==0 or dataset_idx==2:
            # plt.ylabel('VO Size(MB)',fontsize=10)
        # else:
            # plt.ylabel('VO Size(KB)',fontsize=10)
        # plt.xlim(0, 6)
        # plt.ylim(0, fig2_yticks[dataset_idx][-1])
        # plt.legend(['dic','inv',"sq"],ncol=3,fontsize=5)
        # plt.savefig("./figure1/fig5_q=%d_%s.jpg"%(Q,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    # plt.clf()
    
    # #VO Verification Time (section)
    # df= pd.DataFrame(columns=("method",'k','dic MHT','inv MHT',"Dec S'",'Check gram','Recount sid','Cal edit-dist', 'total time (s)'))
    # for K in range(len(VO_verification_sections[0])):
        # col={"method":[labels[0]],'k':[K+1],'dic MHT':[VO_verification_sections[0][K][0]],'inv MHT':[VO_verification_sections[0][K][1]],"Dec S'":[VO_verification_sections[0][K][2]],'Check gram':[VO_verification_sections[0][K][3]],'Recount sid':[VO_verification_sections[0][K][4]],'Cal edit-dist':[VO_verification_sections[0][K][5]],'total time (s)':[sum(VO_verification_sections[0][K])]}
        # df=df.append(pd.DataFrame(col),ignore_index=True)
    # for K in range(len(VO_verification_sections[1])):
        # col={"method":[labels[1]],'k':[K+1],'dic MHT':[VO_verification_sections[1][K][0]],'inv MHT':[VO_verification_sections[1][K][1]],"Dec S'":[VO_verification_sections[1][K][2]],'Check gram':[VO_verification_sections[1][K][3]],'Recount sid':[VO_verification_sections[1][K][4]],'Cal edit-dist':[VO_verification_sections[1][K][5]],'total time (s)':[sum(VO_verification_sections[1][K])]}
        # df=df.append(pd.DataFrame(col),ignore_index=True)
    # df.to_csv("./figure1/VO_Verification_Time(%s).csv"%(dataset_name[dataset_idx]),index=None)
    # with plt.style.context(['ieee']):
        # y=[0,0,0,0,0]
        # for idx in range(len(VO_verification_sections[0][0])):
            # plt.bar(np.array([it-0.175 for it in [1, 2, 3, 4, 5]]),[it[idx] for it in VO_verification_sections[0]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[0])
            # if idx!=len(VO_verification_sections[0][0])-1:
                # for y_idx in range(len(y)):
                    # y[y_idx]+=[it[idx] for it in VO_verification_sections[0]][y_idx]
        # for idx in range(len(y)):
            # plt.text([it-0.175 for it in [1, 2, 3, 4, 5]][idx], y[idx] + 0.001, labels[0], ha='center', va='bottom', fontsize=3)
        # y=[0,0,0,0,0]
        # for idx in range(len(VO_verification_sections[1][0])):
            # plt.bar(np.array([it+0.175 for it in [1, 2, 3, 4, 5]]),[it[idx] for it in VO_verification_sections[1]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[1])
            # if idx!=len(VO_verification_sections[1][0])-1:
                # for y_idx in range(len(y)):
                    # y[y_idx]+=[it[idx] for it in VO_verification_sections[1]][y_idx]
        # for idx in range(len(y)):
            # plt.text([it+0.175 for it in [1, 2, 3, 4, 5]][idx], y[idx] + 0.001, labels[1], ha='center', va='bottom', fontsize=3)
        # plt.xticks([1, 2, 3, 4, 5])
        # plt.yticks(fig3_yticks[dataset_idx])
        # plt.tick_params(direction='in',top=True,right=True,labelsize=8)
        # plt.xlabel('$k$',fontsize=10)
        # plt.ylabel('Time(Second)',fontsize=10)
        # plt.xlim(0, 6)
        # plt.ylim(0, fig3_yticks[dataset_idx][-1])
        # plt.legend(['dic MHT','inv MHT',"Dec S'",'Check gram','Recount sid','Cal edit-dist'],ncol=6,fontsize=3)
        # plt.savefig("./figure1/fig6_q=%d_%s.jpg"%(Q,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    # plt.clf()

K=1
fig1_yticks=[[0, 0.4, 0.8, 1.2, 1.6, 2.0],[0, 0.02, 0.04, 0.06, 0.08, 0.10],[0, 10, 20, 30, 40, 50]]
fig2_yticks=[[0, 30, 60, 90, 120, 150],[0, 2, 4, 6, 8, 10],[0, 20, 40, 60, 80, 100]]
fig3_yticks=[[0, 0.01, 0.02, 0.03, 0.04, 0.05],[0, 0.001, 0.002, 0.003, 0.004, 0.005],[0, 1, 2, 3, 4, 5]]
# for dataset_idx in range(len(datasets)):
    # dataset=datasets[dataset_idx]
    # querys=query_list[dataset_idx]
    # optimizers=["gs_square","e_gs_square"]
    # qs=range(2,7,1)
    # xs,VO_construction_times,VO_construction_sections,VO_sizes,VO_size_sections,VO_verification_times,VO_verification_sections,CSTR_nums,RG_FLAGs=[],[],[],[],[],[],[],[],[]
    # for optimizer in optimizers:
        # x,VO_construction_time,VO_construction_section,VO_size,VO_size_section,VO_verification_time,VO_verification_section,CSTR_num,RG_FLAG=[],[],[],[],[],[],[],[],[]
        # for Q in qs:
            # DO=dataOwner(dataset,q=Q,optimizer=optimizer)

            # D,q,S_dic,S_inv=DO.outsourced()

            # #print("\n-----------------data outsource start--------------------")
            # CSP=cloudServerProvider(D,q,S_dic,S_inv,optimizer=optimizer)
            # # print("CSP get D(%d): %s......"%(len(D),D[:8]))
            # # print("CSP get q:",q)
            # # print("CSP get S_dic: %s......"%(S_dic[:64]))
            # # print("CSP get S_inv: %s......"%(S_inv[:64]))
            # # print("-----------------data outsource end----------------------\n")

            # #print("-------------------User query start----------------------")
            # U=user(querys,[K for _ in range(len(querys))],q)
            # VO_ct,VO_cs,VO_si,VO_ss,VO_at,VO_as,cstr_num,RG_NUM=0,[0,0,0,0],0,[0,0,0],0,[0,0,0,0,0,0],[],0
            # ineff=0
            # for cnt in tqdm(range(len(querys))):
                # s_q,k=U.get_query()
                # if len(s_q)-q+1-k*q<=0:
                    # ineff+=1
                    # continue
                # #print("User query string='%s' with k=%d"%(s_q,k))
                # R,S_dic,VO_dic,S_inv,VO_inv,VO_sq,VO_t1,cstring_num,rg_flag=CSP.query(s_q,k)
                # cstr_num.append(cstring_num)
                # if rg_flag:
                    # RG_NUM+=1
                # VO_s=(len(VO_dic)+len(VO_inv)+len(VO_sq))/1024
                # VO_ct+=sum(VO_t1)
                # for voidx in range(len(VO_t1)):
                    # VO_cs[voidx]+=VO_t1[voidx]
                # VO_si+=VO_s
                # VO_ss[0]+=len(VO_dic)/1024
                # VO_ss[1]+=len(VO_inv)/1024
                # VO_ss[2]+=len(VO_sq)/1024
                # # print("VO Construction Time = %.3fs"%(VO_t1))
                # # print("VO Size = %.2fKB"%(VO_s))
                # # print("-------------------User query end------------------------\n")
        
                # # print("---------------User authentication start-----------------")
                # #authentication phase
                # U.acceptPublicKey(DO.getPublicKey())
                # #print("User get PublicKey from DO")
                # AU_FLAG,VO_t2=U.authenticate(s_q,k,R,S_dic,VO_dic,S_inv,VO_inv,VO_sq,dataset_name[dataset_idx])
                # #print("VO Authentication Result:",AU_FLAG)
                # assert(AU_FLAG==True)
                # VO_at+=sum(VO_t2)
                # for voidx in range(len(VO_t2)):
                    # VO_as[voidx]+=VO_t2[voidx]
                # #print("---------------User authentication end-------------------\n")
            # print("Query invalidation num:%d"%(ineff))
            # if ineff==len(querys):
                # VO_construction_time.append(0)
                # VO_size.append(0)
                # VO_verification_time.append(0)
                # continue
            # VO_construction_time.append(VO_ct/(len(querys)-ineff))
            # VO_construction_section.append([it/(len(querys)-ineff) for it in VO_cs])
            # if dataset_idx<2:
                # VO_size.append(((VO_si/(len(querys)-ineff))))
                # VO_size_section.append([it/(len(querys)-ineff) for it in VO_ss])
            # else:
                # VO_size.append(((VO_si/(len(querys)-ineff)))/1024)
                # VO_size_section.append([(it/(len(querys)-ineff))/1024 for it in VO_ss])
            # VO_verification_time.append(VO_at/(len(querys)-ineff))
            # VO_verification_section.append([it/(len(querys)-ineff) for it in VO_as])
            # CSTR_num.append(cstr_num)
            # RG_FLAG.append(RG_NUM)
        # xs.append(qs)
        # VO_construction_times.append(VO_construction_time)
        # VO_construction_sections.append(VO_construction_section)
        # VO_sizes.append(VO_size)
        # VO_size_sections.append(VO_size_section)
        # VO_verification_times.append(VO_verification_time)
        # VO_verification_sections.append(VO_verification_section)
        # CSTR_nums.append(CSTR_num)
        # RG_FLAGs.append(RG_FLAG)
        
    # #VO Construction Time
    # with plt.style.context(['ieee']):
        # for idx in range(len(xs)):
            # plt.plot(xs[idx], VO_construction_times[idx], linestyle="-", marker=markers[idx], color=colors[idx], label=labels[idx], linewidth=1, clip_on=False)
        # plt.xticks([2, 3, 4, 5, 6])
        # plt.yticks(fig1_yticks[dataset_idx])
        # plt.tick_params(direction='in',top=True,right=True,labelsize=8)
        # plt.xlabel('Length of grams',fontsize=10)
        # plt.ylabel('Time(Second)',fontsize=10)
        # plt.xlim(2, 6)
        # plt.ylim(0, fig1_yticks[dataset_idx][-1])
        # plt.legend()
        # plt.savefig("./figure2/fig1_k=%d_%s.jpg"%(K,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    # plt.clf()
    
    # #VO Size
    # with plt.style.context(['ieee']):
        # for idx in range(len(xs)):
            # plt.plot(xs[idx], VO_sizes[idx], linestyle="-", marker=markers[idx], color=colors[idx], label=labels[idx], linewidth=1, clip_on=False)
        # plt.xticks([2, 3, 4, 5, 6])
        # plt.yticks(fig2_yticks[dataset_idx])
        # plt.tick_params(direction='in',top=True,right=True,labelsize=8)
        # plt.xlabel('Length of grams',fontsize=10)
        # if dataset_idx==2:
            # plt.ylabel('VO Size(MB)',fontsize=10)
        # else:
            # plt.ylabel('VO Size(KB)',fontsize=10)
        # plt.xlim(2, 6)
        # plt.ylim(0, fig2_yticks[dataset_idx][-1])
        # plt.legend()
        # plt.savefig("./figure2/fig2_k=%d_%s.jpg"%(K,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    # plt.clf()
    
    # #VO Verification Time
    # with plt.style.context(['ieee']):
        # for idx in range(len(xs)):
            # plt.plot(xs[idx], VO_verification_times[idx], linestyle="-", marker=markers[idx], color=colors[idx], label=labels[idx], linewidth=1, clip_on=False)
        # plt.xticks([2, 3, 4, 5, 6])
        # plt.yticks(fig3_yticks[dataset_idx])
        # plt.tick_params(direction='in',top=True,right=True,labelsize=8)
        # plt.xlabel('Length of grams',fontsize=10)
        # plt.ylabel('Time(Second)',fontsize=10)
        # plt.xlim(2, 6)
        # plt.ylim(0, fig3_yticks[dataset_idx][-1])
        # plt.legend()
        # plt.savefig("./figure2/fig3_k=%d_%s.jpg"%(K,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    # plt.clf()
    
    # #VO Construction Time (section)
    # df= pd.DataFrame(columns=("method",'Length of grams','search gram','count sid',"dic MHT","inv MHT", 'total time (s)'))
    # for Q in range(len(VO_construction_sections[0])):
        # col={"method":[labels[0]],'Length of grams':[Q+2],'search gram':[VO_construction_sections[0][Q][0]],'count sid':[VO_construction_sections[0][Q][1]],"dic MHT":[VO_construction_sections[0][Q][2]],"inv MHT":[VO_construction_sections[0][Q][3]],'total time (s)':[sum(VO_construction_sections[0][Q])]}
        # df=df.append(pd.DataFrame(col),ignore_index=True)
    # for Q in range(len(VO_construction_sections[1])):
        # col={"method":[labels[1]],'Length of grams':[Q+2],'search gram':[VO_construction_sections[1][Q][0]],'count sid':[VO_construction_sections[1][Q][1]],"dic MHT":[VO_construction_sections[1][Q][2]],"inv MHT":[VO_construction_sections[1][Q][3]],'total time (s)':[sum(VO_construction_sections[1][Q])]}
        # df=df.append(pd.DataFrame(col),ignore_index=True)
    # df.to_csv("./figure2/VO_Construction_Time(%s).csv"%(dataset_name[dataset_idx]),index=None)
    # with plt.style.context(['ieee']):
        # y=[0,0,0,0,0]
        # for idx in range(len(VO_construction_sections[0][0])):
            # plt.bar(np.array([it-0.175 for it in [2, 3, 4, 5, 6]]),[it[idx] for it in VO_construction_sections[0]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[0])
            # if idx!=len(VO_construction_sections[0][0])-1:
                # for y_idx in range(len(y)):
                    # y[y_idx]+=[it[idx] for it in VO_construction_sections[0]][y_idx]
        # for idx in range(len(y)):
            # plt.text([it-0.175 for it in [2, 3, 4, 5, 6]][idx], y[idx] + 0.005, labels[0], ha='center', va='bottom', fontsize=3)
        # y=[0,0,0,0,0]
        # for idx in range(len(VO_construction_sections[1][0])):
            # plt.bar(np.array([it+0.175 for it in [2, 3, 4, 5, 6]]),[it[idx] for it in VO_construction_sections[1]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[1])
            # if idx!=len(VO_construction_sections[1][0])-1:
                # for y_idx in range(len(y)):
                    # y[y_idx]+=[it[idx] for it in VO_construction_sections[1]][y_idx]
        # for idx in range(len(y)):
            # plt.text([it+0.175 for it in [2, 3, 4, 5, 6]][idx], y[idx] + 0.005, labels[1], ha='center', va='bottom', fontsize=3)
        # plt.xticks([2, 3, 4, 5, 6])
        # plt.yticks(fig1_yticks[dataset_idx])
        # plt.tick_params(direction='in',top=True,right=True,labelsize=6)
        # plt.xlabel('Length of grams',fontsize=10)
        # plt.ylabel('Time(Second)',fontsize=10)
        # plt.xlim(1, 7)
        # plt.ylim(0, fig1_yticks[dataset_idx][-1])
        # plt.legend(['search gram','count sid',"dic MHT","inv MHT"],ncol=4,fontsize=5)
        # plt.savefig("./figure2/fig4_k=%d_%s.jpg"%(K,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    # plt.clf()
    
    # #VO Size (section)
    # df= pd.DataFrame(columns=("method",'Length of grams','dic','inv',"sq", 'total size (KB)',"Cstring num each query"))
    # for Q in range(len(VO_size_sections[0])):
        # col={"method":[labels[0]],'Length of grams':[Q+2],'dic':[VO_size_sections[0][Q][0]],'inv':[VO_size_sections[0][Q][1]],"sq":[VO_size_sections[0][Q][2]],'total size (KB)':[sum(VO_size_sections[0][Q])],"Cstring num each query":[CSTR_nums[0][Q]]}
        # df=df.append(pd.DataFrame(col),ignore_index=True)
    # for Q in range(len(VO_size_sections[1])):
        # col={"method":[labels[1]],'Length of grams':[Q+2],'dic':[VO_size_sections[1][Q][0]],'inv':[VO_size_sections[1][Q][1]],"sq":[VO_size_sections[1][Q][2]],'total size (KB)':[sum(VO_size_sections[1][Q])],"Cstring num each query":[CSTR_nums[1][Q]],"representative_grams":[RG_FLAGs[1][Q]]}
        # df=df.append(pd.DataFrame(col),ignore_index=True)
    # df.to_csv("./figure2/VO_Size(%s).csv"%(dataset_name[dataset_idx]),index=None)
    # with plt.style.context(['ieee']):
        # y=[0,0,0,0,0]
        # for idx in range(len(VO_size_sections[0][0])):
            # plt.bar(np.array([it-0.175 for it in [2, 3, 4, 5, 6]]),[it[idx] for it in VO_size_sections[0]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[0])
            # if idx!=len(VO_size_sections[0][0])-1:
                # for y_idx in range(len(y)):
                    # y[y_idx]+=[it[idx] for it in VO_size_sections[0]][y_idx]
        # for idx in range(len(y)):
            # plt.text([it-0.175 for it in [2, 3, 4, 5, 6]][idx], y[idx] + 0.001, labels[0], ha='center', va='bottom', fontsize=3)
        # y=[0,0,0,0,0]
        # for idx in range(len(VO_size_sections[1][0])):
            # plt.bar(np.array([it+0.175 for it in [2, 3, 4, 5, 6]]),[it[idx] for it in VO_size_sections[1]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[1])
            # if idx!=len(VO_size_sections[1][0])-1:
                # for y_idx in range(len(y)):
                    # y[y_idx]+=[it[idx] for it in VO_size_sections[1]][y_idx]
        # for idx in range(len(y)):
            # plt.text([it+0.175 for it in [2, 3, 4, 5, 6]][idx], y[idx] + 0.001, labels[1], ha='center', va='bottom', fontsize=3)
        # plt.xticks([2, 3, 4, 5, 6])
        # plt.yticks(fig2_yticks[dataset_idx])
        # plt.tick_params(direction='in',top=True,right=True,labelsize=6)
        # plt.xlabel('Length of grams',fontsize=10)
        # if dataset_idx==2:
            # plt.ylabel('VO Size(MB)',fontsize=10)
        # else:
            # plt.ylabel('VO Size(KB)',fontsize=10)
        # plt.xlim(1, 7)
        # plt.ylim(0, fig2_yticks[dataset_idx][-1])
        # plt.legend(['dic','inv',"sq"],ncol=3,fontsize=5)
        # plt.savefig("./figure2/fig5_k=%d_%s.jpg"%(K,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    # plt.clf()
    
    # #VO Verification Time (section)
    # df= pd.DataFrame(columns=("method",'Length of grams','dic MHT','inv MHT',"Dec S'",'Check gram','Recount sid','Cal edit-dist', 'total time (s)'))
    # for Q in range(len(VO_verification_sections[0])):
        # col={"method":[labels[0]],'Length of grams':[Q+2],'dic MHT':[VO_verification_sections[0][Q][0]],'inv MHT':[VO_verification_sections[0][Q][1]],"Dec S'":[VO_verification_sections[0][Q][2]],'Check gram':[VO_verification_sections[0][Q][3]],'Recount sid':[VO_verification_sections[0][Q][4]],'Cal edit-dist':[VO_verification_sections[0][Q][5]],'total time (s)':[sum(VO_verification_sections[0][Q])]}
        # df=df.append(pd.DataFrame(col),ignore_index=True)
    # for Q in range(len(VO_verification_sections[1])):
        # col={"method":[labels[1]],'Length of grams':[Q+2],'dic MHT':[VO_verification_sections[1][Q][0]],'inv MHT':[VO_verification_sections[1][Q][1]],"Dec S'":[VO_verification_sections[1][Q][2]],'Check gram':[VO_verification_sections[1][Q][3]],'Recount sid':[VO_verification_sections[1][Q][4]],'Cal edit-dist':[VO_verification_sections[1][Q][5]],'total time (s)':[sum(VO_verification_sections[1][Q])]}
        # df=df.append(pd.DataFrame(col),ignore_index=True)
    # df.to_csv("./figure2/VO_Verification_Time(%s).csv"%(dataset_name[dataset_idx]),index=None)
    # with plt.style.context(['ieee']):
        # y=[0,0,0,0,0]
        # for idx in range(len(VO_verification_sections[0][0])):
            # plt.bar(np.array([it-0.175 for it in [2, 3, 4, 5, 6]]),[it[idx] for it in VO_verification_sections[0]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[0])
            # if idx!=len(VO_verification_sections[0][0])-1:
                # for y_idx in range(len(y)):
                    # y[y_idx]+=[it[idx] for it in VO_verification_sections[0]][y_idx]
        # for idx in range(len(y)):
            # plt.text([it-0.175 for it in [2, 3, 4, 5, 6]][idx], y[idx] + 0.0001, labels[0], ha='center', va='bottom', fontsize=3)
        # y=[0,0,0,0,0]
        # for idx in range(len(VO_verification_sections[1][0])):
            # plt.bar(np.array([it+0.175 for it in [2, 3, 4, 5, 6]]),[it[idx] for it in VO_verification_sections[1]],bottom=y,width=0.3,color=tmpcolors[idx],label=labels[1])
            # if idx!=len(VO_verification_sections[1][0])-1:
                # for y_idx in range(len(y)):
                    # y[y_idx]+=[it[idx] for it in VO_verification_sections[1]][y_idx]
        # for idx in range(len(y)):
            # plt.text([it+0.175 for it in [2, 3, 4, 5, 6]][idx], y[idx] + 0.0001, labels[1], ha='center', va='bottom', fontsize=3)
        # plt.xticks([2, 3, 4, 5, 6])
        # plt.yticks(fig3_yticks[dataset_idx])
        # plt.tick_params(direction='in',top=True,right=True,labelsize=6)
        # plt.xlabel('Length of grams',fontsize=10)
        # plt.ylabel('Time(Second)',fontsize=10)
        # plt.xlim(1, 7)
        # plt.ylim(0, fig3_yticks[dataset_idx][-1])
        # plt.legend(['dic MHT','inv MHT',"Dec S'",'Check gram','Recount sid','Cal edit-dist'],ncol=6,fontsize=3)
        # plt.savefig("./figure2/fig6_k=%d_%s.jpg"%(K,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    # plt.clf()

K=1
Q=2
fig2_yticks=[[0, 40, 80, 120, 160, 200],[0, 2, 4, 6, 8, 10],[0, 4, 8, 12, 16, 20]]
flags=[["gs_square",False,False,False,False,"snappy"],["gs_square",True,False,False,False,"snappy"],["gs_square",False,True,False,False,"snappy"],["gs_square",False,False,True,False,"snappy"],["gs_square",False,False,False,True,"gzip"],["gs_square",False,False,False,True,"bz2"],["gs_square",False,False,False,True,"lzma"],["gs_square",False,False,False,True,"snappy"],["e_gs_square",True,True,True,True,"lzma"]]
labels=["GS$^{2}$","length filtering","representative grams","empty set authentication","gzip compression","bz2 compression","lzma compression","snappy compression","GS$^{2}$-opt"]
for dataset_idx in range(len(datasets)):
    dataset=datasets[dataset_idx]
    querys=query_list[dataset_idx]
    VO_size_section=[]
    for flag in flags:
        DO=dataOwner(dataset,q=Q,optimizer=flag[0],length_filtering=flag[1],representative_grams=flag[2],empty_set_authentication=flag[3],compress=flag[4],compressor=flag[5])
        D,q,S_dic,S_inv=DO.outsourced()
        CSP=cloudServerProvider(D,q,S_dic,S_inv,optimizer=flag[0],length_filtering=flag[1],representative_grams=flag[2],empty_set_authentication=flag[3],compress=flag[4],compressor=flag[5])
        U=user(querys,[K for _ in range(len(querys))],q)
        ineff=0
        VO_ss=[0,0,0]
        for cnt in tqdm(range(len(querys))):
            s_q,k=U.get_query()
            if len(s_q)-q+1-k*q<=0:
                ineff+=1
                continue
            R,S_dic,VO_dic,S_inv,VO_inv,VO_sq,VO_t1,cstring_num,rg_flag=CSP.query(s_q,k)
            VO_s=(len(VO_dic)+len(VO_inv)+len(VO_sq))/1024
            VO_ss[0]+=len(VO_dic)/1024
            VO_ss[1]+=len(VO_inv)/1024
            VO_ss[2]+=len(VO_sq)/1024
        if dataset_idx!=2:
            VO_size_section.append([it/(len(querys)-ineff) for it in VO_ss])
        else:
            VO_size_section.append([(it/(len(querys)-ineff))/1024 for it in VO_ss])
        print("Query invalidation num:%d"%(ineff))
    if dataset_idx==2:
        size_unit="(MB)"
    else:
        size_unit="(KB)"
    df= pd.DataFrame(columns=("Optimizer",'dic','inv',"sq", 'total size '+size_unit))
    for flag_idx in range(len(flags)):
        col={"Optimizer":[labels[flag_idx]],'dic':[VO_size_section[flag_idx][0]],'inv':[VO_size_section[flag_idx][1]],"sq":[VO_size_section[flag_idx][2]],'total size '+size_unit:[sum(VO_size_section[flag_idx])]}
        df=df.append(pd.DataFrame(col),ignore_index=True)
    df.to_csv("./figure3/VO_Size_k=%d_q=%d(%s).csv"%(K,Q,dataset_name[dataset_idx]),index=None)
    with plt.style.context(['ieee']):
        y=[0,0,0,0,0,0,0,0,0]
        for idx in range(len(VO_size_section[0])):
            plt.bar(np.array(range(1,10,1)),[it[idx] for it in VO_size_section],bottom=y,width=0.4,color=tmpcolors[idx])
            if idx!=len(VO_size_section[0])-1:
                for y_idx in range(len(y)):
                    y[y_idx]+=[it[idx] for it in VO_size_section][y_idx]
        plt.xticks([1,2.8,4,5,6,7,8,9,10],labels=labels,rotation=-20)
        plt.yticks(fig2_yticks[dataset_idx])
        plt.tick_params(direction='in',top=True,right=True,labelsize=6)
        plt.xlabel('Optimizer',fontsize=10)
        if dataset_idx==2:
            plt.ylabel('VO Size(MB)',fontsize=10)
        else:
            plt.ylabel('VO Size(KB)',fontsize=10)
        plt.xlim(0, 10)
        plt.ylim(0, fig2_yticks[dataset_idx][-1])
        plt.legend(['dic','inv',"sq"],ncol=3,fontsize=5)
        plt.savefig("./figure3/fig7_k=%d_q=%d_%s.jpg"%(K,Q,dataset_name[dataset_idx]), bbox_inches='tight', dpi=600)
    plt.clf()
    

