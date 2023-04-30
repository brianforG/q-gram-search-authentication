import Levenshtein
import hashlib
import time
import os
from nltk import ngrams
from collections import Counter
from Crypto.PublicKey import RSA
from Crypto.Hash import SHA256
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

class user(object):
    def __init__(self, s_q, k, q, compressor="lzma"):
        self.s_q = s_q       #query string
        self.k = k
        self.q = q
        self.point=0
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
    def get_query(self):
        if self.point<len(self.s_q) and self.point<len(self.k):
            res=(self.s_q[self.point],self.k[self.point])
            self.point+=1
            return res
        else:
            return None
    def acceptPublicKey(self,file_name):
        if not os.path.exists(file_name):
            print("public_key not found!")
        else:
            with open(file_name, 'rb') as x:
                self.public_key = RSA.importKey(x.read())
        
    def authenticate(self,s_q,k,R,S_dic,VO_dic,S_inv,VO_inv,VO_sq,dateset_name):
        time_au=[]
        
        time_au1 = time.clock()
        τ=len(s_q)-self.q+1-k*self.q
        if type(VO_dic)==type(b'\xe5\x8c\x97\xe4\xba\xac'):
            VO_dic=self.compressor.decompress(VO_dic).decode('utf-8')
        
        mht=MHT()
        #Step 1:  re-constructs the MHT from VO_dic
        dic_message,dic,_,_=mht.String2VO(VO_dic)
        #print("dic_root:",dic_message)
        time_au2 = time.clock()
        time_au.append(time_au2-time_au1)
        dic_message_hash = SHA256.new()
        dic_message_hash.update(dic_message.encode())
        if not PKCS1_signature.new(self.public_key).verify(dic_message_hash, S_dic):
            print("check S_dic == S_dic' : False")
            return False,time.clock()-time_au1
        #else:
            #print("check S_dic == S_dic' : True")
        time_au3 = time.clock()
        
        #Step 2:  re-constructs the MHT from VO_inv
        if type(VO_inv)==type(b'\xe5\x8c\x97\xe4\xba\xac'):
            VO_inv=self.compressor.decompress(VO_inv).decode('utf-8')
        inv_message,_,gram2inverted,vo_gram=mht.String2VO(VO_inv)
        #print("inv_root:",inv_message)
        time_au4 = time.clock()
        time_au.append(time_au4-time_au3)
        inv_message_hash = SHA256.new()
        inv_message_hash.update(inv_message.encode())
        if not PKCS1_signature.new(self.public_key).verify(inv_message_hash, S_inv):
            print("check S_inv == S_inv' : False")
            return False,time.clock()-time_au1
        #else:
            #print("check S_inv == S_inv' : True")
        time_au5 = time.clock()
        time_au.append(time_au5-time_au4+time_au3-time_au2)
        
        #Step 3:  check e-gram and ne-gram, and then re-count the Sid occurrence times on inverted list and check the C_string
        ne_gram=set(["".join(it) for it in ngrams(s_q.lower(),self.q)])-set(gram2inverted.keys())
        #print("ne_gram: %s......"%(list(ne_gram)[:]))
        point=0
        #print("vo_gram: %s......"%(sorted(list(vo_gram))[:]))
        VO_sq_flag=False
        if len(VO_sq)>0:
            VO_sq=VO_sq.split(",")
            VO_sq=[(int(VO_sq[0]),VO_sq[1]),(int(VO_sq[2]),VO_sq[3])]
            VO_sq_flag=True
            vo_gram=sorted(list(vo_gram))
            ne_gram=sorted(ne_gram)
            real_ne_gram=[]
            ne_gram_num=0
            for idx in range(len(vo_gram)-1):
                gram1,gram2=vo_gram[idx],vo_gram[idx+1]
                while point<len(ne_gram) and ne_gram[point]<gram1:
                    point+=1
                while point<len(ne_gram) and ne_gram[point]>gram1 and ne_gram[point]<gram2:
                    real_ne_gram.append(ne_gram[point])
                    point+=1
                    ne_gram_num+=1
            if len(vo_gram)>0:
                for idx in range(point,len(ne_gram)):
                    if ne_gram[idx]>vo_gram[-1]:
                        real_ne_gram.append(ne_gram[idx])
                        ne_gram_num+=1
                if len(R)>0 and ne_gram_num+τ-1<len(ne_gram):
                    print(VO_sq)
                    print(vo_gram)
                    print("R:",R)
                    print("ne_gram_num:",ne_gram_num)
                    print("real_ne_gram:",real_ne_gram)
                    print("ne_gram:",ne_gram)
                    print("τ:",τ)
                    print("ne_gram error")
                    return False,time.clock()-time_au1
                else:
                    ne_gram=real_ne_gram
        elif len(vo_gram)>0:
            vo_gram=sorted(list(vo_gram))
            for i,gram in enumerate(sorted(ne_gram)):
                while point<len(vo_gram)-1 and gram>vo_gram[point+1]:
                    point+=1
                if not (gram>vo_gram[point] and ((point<len(vo_gram)-1 and gram<vo_gram[point+1]) or point==len(vo_gram)-1)):
                    print(dic)
                    print(gram2inverted)
                    print()
                    print(VO_dic)
                    print()
                    print(VO_inv)
                    return False,time.clock()-time_au1
        #print("check ne_gram == %s : True"%(ne_gram))
        time_au6 = time.clock()
        time_au.append(time_au6-time_au5)
        
        sid_list=[str(j) for i in gram2inverted.values() for j in i]
        sid_cnt=Counter(sid_list)
        sid_cstr=set()
        for sid,cnt in sid_cnt.items():
            if cnt>=τ:
                sid_cstr.add(sid)
        #print("C_string:",set(self.sid2string[sid] for sid in sid_cstr))
        if VO_sq_flag==True:
            if dateset_name=="LastName_dataset":
                max_len=13
            elif dateset_name=="Author_dataset":
                max_len=67
            else:
                max_len=11
            if len(s_q)-k-1>0 and len(s_q)-k-1<max_len and (len(s_q)-len(VO_sq[0][1])!=k+1 or (len(VO_sq[1][1])-len(s_q)!=k+1 and len(VO_sq[1][1])!=max_len)):
                print("s_q=%s k=%d"%(s_q,k))
                print(VO_sq[0][1],VO_sq[1][1],max_len)
                return False,time.clock()-time_au1
            for sid in list(sid_cstr):
                if int(sid)<=VO_sq[0][0] or int(sid)>=VO_sq[1][0]:
                    sid_cstr.remove(sid)
        #print("check sid occurrence times >= %d == %s...... : True"%(τ,list(sid_cstr)[:5]))
        time_au7 = time.clock()
        time_au.append(time_au7-time_au6)
        
        if len(R)>0:
            r=set()
            sids=set(dic.keys())- sid_cstr
            for sid in dic.keys():
                string=dic[sid]
                if Levenshtein.distance(s_q,string)<=k:
                    r.add(string)
            if len(R-r)!=0:
                print("R error")
                print(R)
                print(dic)
                print(sid_cstr)
                return False,time.clock()-time_au1
        #print("check C_string %s...... : True"%(list(dic.keys())[:5]))
        time_au8 = time.clock()
        time_au.append(time_au8-time_au7)
        
        #print("VO Authentication Time: %.3fs"%(time_au2-time_au1))
        return True,time_au
                

        
