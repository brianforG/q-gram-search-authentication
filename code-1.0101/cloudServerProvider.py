import Levenshtein
import hashlib
import base64
import time
import math
import os
from collections import Counter
from collections import defaultdict
from nltk import ngrams
from Crypto.PublicKey import RSA
from Crypto.Hash import SHA256
from Crypto.Cipher import PKCS1_v1_5 as PKCS1_cipher
from Crypto.Signature import PKCS1_v1_5 as PKCS1_signature
from compress import Compressor
from MHT import *

def getHash(v):
    v = v.encode('utf-8')
    v = getattr(hashlib, "sha256")(v).hexdigest()
    return v

def h_t(t):
    #compute hash value h_t of the tuple t(sid,string) ∈ D
    a=bytearray.fromhex(t[0])
    b=bytearray.fromhex(t[1])
    return getattr(hashlib, "sha256")(a+b).hexdigest()

class cloudServerProvider(object):
    def __init__(self, D, q, S_dic, S_inv, optimizer="gs_square",length_filtering=False,representative_grams=False,empty_set_authentication=False,compress=False,compressor="lzma"):
        time_init1 = time.clock()
        self.D = D
        self.q = q
        self.S_dic = S_dic
        self.S_inv = S_inv
        self.optimizer = optimizer
        if optimizer=="gs_square":
            self.length_filtering = False
            self.representative_grams = False
            self.empty_set_authentication = False
            self.compress = False
        elif optimizer=="e_gs_square":
            self.length_filtering = True
            self.representative_grams = True
            self.empty_set_authentication = True
            self.compress = False
        if length_filtering:
            self.length_filtering=True
        if representative_grams:
            self.representative_grams = True
        if empty_set_authentication:
            self.empty_set_authentication = True
        if compress:
            self.compress = False
        if self.compress:
            self.compressor = Compressor()
            if compressor=="gzip":
                self.compressor.use_gzip()
            elif compressor=="bz2":
                self.compressor.use_bz2()
            elif compressor=="lzma":
                self.compressor.use_lzma()
            elif compressor=="lz4":
                self.compressor.use_lz4()
            elif compressor=="snappy":
                self.compressor.use_snappy()
        self.string2sid = defaultdict(int)
        self.sid2string = defaultdict(str)
        self.mht1 = MHT(hash_type="SHA256")
        self.gram2inverted = defaultdict(list)
        self.mht2 = MHT(hash_type="SHA256")
        self.construct()
        time_init2 = time.clock()
        #print("CSP init time: %.3fs"%(time_init2-time_init1))
        
    def construct(self):
        #build a dictionary authentication structure
        
        ##step1: sort the strings in dataset D with alphabetical order,and assign each of string an unique number id,Sid
        for i in range(len(self.D)):
            self.string2sid[self.D[i]]=i+1
            self.sid2string[str(i+1)]=self.D[i]
                
        ##step2: compute the tuple hash h_t for each tuple t(sid,string) ∈ D
        ##step3: compute the Merkle hash tree for dataset D
        for i in range(len(self.D)):
            sid,string=str(i+1),self.D[i]
            self.mht1.add_leaf({sid:string})
        self.mht1.make_tree()
        
        if self.length_filtering==True:
            len2sid=[[1,len(self.D)] for i in range(100)]
            for i in range(len(self.D)-1):
                if len(self.D[i])<len(self.D[i+1]):
                    len2sid[len(self.D[i+1])][0]=i+1
                    len2sid[len(self.D[i])][1]=i+2
            for i in range(len(self.D[len(self.D)-1]),100,1):
                len2sid[i][0]=len2sid[len(self.D[len(self.D)-1])][0]
                
            self.len2sid=len2sid
        
        #build an inverted list authentication structure

        ##step1: build a q-gram based inverted list table
        for i in range(len(self.D)):
            sid=str(i+1)
            for gram in ngrams(self.D[i], self.q):
                gram="".join(gram)
                self.gram2inverted[gram].append(sid)

        ##step2: compute the tuple hash ht for each tuple(gram,inverted list) t ∈ inverted lists
        ##step3: compute the Merkle hash tree for the inverted lists
        for gram in sorted(self.gram2inverted.keys()):
            if self.length_filtering==True:
                mht3=MHT(hash_type="SHA256")
                for idx,leaf in enumerate(self.gram2inverted[gram]):
                    mht3.add_leaf(str(leaf))
                mht3.make_tree()
                self.mht2.add_leaf((gram,mht3))
            else:
                self.mht2.add_leaf((gram,"_".join(self.gram2inverted[gram])))
        self.mht2.make_tree()

    def query(self,s_q,k):
        s_q=s_q.lower()
        R=set()
        
        #search gram
        time_vo1 = time.clock()
        τ=len(s_q)-self.q+1-k*self.q
        s_q_grams=["".join(it) for it in ngrams(s_q,self.q)]
        e_gram,ne_gram=set(),set()
        for gram in s_q_grams:
            if len(self.gram2inverted[gram])==0:
                ne_gram.add(gram)
            else:
                e_gram.add(gram)
        if self.empty_set_authentication==False or len(ne_gram)<=k*self.q:
            if self.representative_grams==True:
                e_gram_num=len(e_gram)-τ+1
                e_gram_candidate=sorted([(gram,self.gram2inverted[gram]) for gram in e_gram],key=lambda x:(len(x[1]),x[0]))[:e_gram_num]
                e_gram=set([i[0] for i in e_gram_candidate])
                time_vo2 = time.clock()
                sid_cstr=set([j for i in e_gram_candidate for j in i[1]])
            else:
                time_vo2 = time.clock()
                sid_list=[]
                for gram in e_gram:
                    sid_list=sid_list+self.gram2inverted[gram]
                sid_cnt=Counter(sid_list)
                sid_cstr=set()
                for sid,cnt in sid_cnt.items():
                    if cnt>=τ:
                        sid_cstr.add(sid)
            if self.length_filtering==True:
                for sid in list(sid_cstr):
                    if int(sid)>self.len2sid[len(s_q)+k][1] or int(sid)<self.len2sid[max(len(s_q)-k,0)][0]:
                        sid_cstr.remove(sid)
        # print("optimizer=%s"%(self.optimizer))
        # print(s_q,k)
        # print("CSP generate e_gram(%d): %s......"%(len(e_gram),list(e_gram)[:]))
        # print("CSP generate ne_gram(%d): %s......"%(len(ne_gram),list(ne_gram)[:]))
        # print("CSP generate sid_cstring(%d) =%s......"%(len(sid_cstr),list(sid_cstr)[:5]))
        # print("CSP generate C_string(%d) =%s......"%(len(sid_cstr),list(self.sid2string[sid] for sid in sid_cstr)[:5]))
            time_vo3 = time.clock()
            for sid in sid_cstr:
                cstr=self.sid2string[sid]
                if Levenshtein.distance(s_q,cstr)<=k:
                    R.add(cstr)
            #print("CSP generate R =",R)
        else:
            print("enpty set true")
            time_vo2 = time.clock()
            e_gram,sid_cstr=set(),set()
            time_vo3 = time.clock()
        
        time_vo4 = time.clock()
        #construct VO_dic
        index_state=[]
        for i in range(len(self.D)):
            sid,string=str(i+1),self.D[i]
            if sid in sid_cstr:
                index_state.append(1)
            else:
                index_state.append(0)
        ##buddy system
        if self.length_filtering==True:
            if len(self.D)>1000000:
                buddy_system_limit=4
            else:
                buddy_system_limit=8
            for i in range(0,len(index_state),buddy_system_limit):
                if sum(index_state[i:i+buddy_system_limit])>0:
                    for j in range(i,i+buddy_system_limit,1):
                        if j<len(index_state):
                            index_state[j]=1
                        else:
                            break
        time_vo5 = time.clock()
        VO_dic=self.mht1.get_proof(index_state)
        if self.compress:
            VO_dic=self.compressor.compress(VO_dic.encode('utf-8'))
                
        #construct VO_inv
        time_vo6 = time.clock()
        ne_gram_point,ne_gram_list=0,sorted(list(ne_gram))
        gram_list=[gram for gram in sorted(self.gram2inverted.keys()) if len(self.gram2inverted[gram])!=0]
        leaf_state=[0 for i in range(len(gram_list))]
        #print(start,end)
        for i in range(len(gram_list)):
            if gram_list[i] in e_gram:
                if self.length_filtering==True:
                    start,end=self.len2sid[max(len(s_q)-k,0)][0],self.len2sid[len(s_q)+k][1]
                    gram,inv_list=gram_list[i],self.gram2inverted[gram_list[i]]
                    tmp_state=[]
                    for j in range(len(inv_list)):
                        if int(inv_list[j])>start and int(inv_list[j])<end:
                            tmp_state.append(1)
                        else:
                            tmp_state.append(0)
                    for j in range(len(tmp_state)):
                        if tmp_state[j]==1:
                            if j>0:
                               tmp_state[j-1]=1
                            break
                    for j in range(len(tmp_state)-1,-1,-1):
                        if tmp_state[j]==1:
                            if j<len(tmp_state)-1:
                               tmp_state[j+1]=1
                            break
                    #print(tmp_state)
                    leaf_state[i]=tmp_state
                else:
                    leaf_state[i]=1
            elif ne_gram_point<len(ne_gram_list):
                if i<len(gram_list) and ne_gram_list[ne_gram_point]>gram_list[i]:
                    if i+1<len(self.gram2inverted.keys()):
                        if i+1<len(gram_list) and ne_gram_list[ne_gram_point]<gram_list[i+1]:
                            leaf_state[i]=leaf_state[i+1]=2
                            ne_gram_point+=1
                            continue
                    else:
                        leaf_state[i]=1
                        ne_gram_point+=1
        #print(leaf_state)
        time_vo7 = time.clock()
        
        VO_inv=self.mht2.get_proof(leaf_state)
        #print(VO_inv)
        if self.compress:
            VO_inv=self.compressor.compress(VO_inv.encode('utf-8'))

        # for level in range(len(self.mht2.levels) - 2, -1, -1):
            # print("level=",level)
            # for t in self.mht2.levels[level]:
                # print(self.mht2._to_hex(t))
        
        if self.length_filtering==True:
            VO_sq=[(self.len2sid[max(0,len(s_q)-k)][0],self.D[self.len2sid[max(0,len(s_q)-k)][0]-1]),(self.len2sid[len(s_q)+k][1],self.D[self.len2sid[len(s_q)+k][1]-1])]
            VO_sq=",".join([str(j) for i in VO_sq for j in i])
        else:
            VO_sq=""
        time_vo8 = time.clock()
        VO_construction_time=[time_vo2-time_vo1+time_vo5-time_vo4+time_vo7-time_vo6,time_vo3-time_vo2,time_vo6-time_vo5,time_vo8-time_vo7]
        print("ssss",self.representative_grams)
        print("k=%d,q=%d,R=%s,VOdic=%s,VOinv=%s,isRepreG=%d" % (k, self.q, R, VO_dic, VO_inv, self.representative_grams))
        if self.representative_grams==True:
            self.representative_grams=False
            R2,_,VO_dic2,_,VO_inv2,_,VO_construction_time2,sid_cstr_len,_=self.query(s_q,k)
            self.representative_grams=True
            if len(VO_dic2)+len(VO_inv2)<len(VO_dic)+len(VO_inv):
                print("k=%d,q=%d,R=%s,VOdic=%s,VOinv=%s,isRepreG=%d" % (k, self.q, R2, VO_dic2, VO_inv2, False))
                return (R2,self.S_dic,VO_dic2,self.S_inv,VO_inv2,VO_sq,VO_construction_time2,sid_cstr_len,False)
        return (R,self.S_dic,VO_dic,self.S_inv,VO_inv,VO_sq,VO_construction_time,len(sid_cstr),self.representative_grams)
            
        # print(self.mht2.get_proof(3))
        # print(self.mht2.get_leaf(0))
        # print(self.mht2.get_leaf(1))
        # print(self.mht2.get_leaf(2))
        # print(self.mht2.get_leaf(3))
    
        
