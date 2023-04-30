import Levenshtein
import hashlib
import base64
import time
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

def e_gs_square(x):
    return (len(x),x)

class dataOwner(object):
    def __init__(self, D, q, optimizer="gs_square",length_filtering=False,representative_grams=False,empty_set_authentication=False,compress=False,compressor="lzma"):
        #print("--------------------DO init start------------------------")
        time_init1 = time.clock()
        self.optimizer=optimizer
        self.q = q
        self.string2sid = defaultdict(int)
        self.sid2string = defaultdict(str)
        self.mht1 = MHT(hash_type="SHA256")
        self.gram2inverted = defaultdict(list)
        self.mht2 = MHT(hash_type="SHA256")
        self.compressor = Compressor()
        if optimizer=="gs_square":
            self.D = sorted(list(set([d.lower() for d in D])))
            self.length_filtering = False
            self.representative_grams = False
            self.empty_set_authentication = False
            self.compress = False
        elif optimizer=="e_gs_square":
            self.length_filtering = True
            self.representative_grams = True
            self.empty_set_authentication = True
            self.compress = True
        if length_filtering:
            self.length_filtering=True
        if representative_grams:
            self.representative_grams = True
        if empty_set_authentication:
            self.empty_set_authentication = True
        if compress:
            self.compress = True
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
        if optimizer=="e_gs_square" or self.length_filtering:
            self.D = sorted(list(set([d.lower() for d in D])),key=e_gs_square)
        #print("D(%d): %s......"%(len(self.D),self.D[:8]))
        if not (os.path.exists("private_key.pem") and os.path.exists("public_key.pem")):
            self.getKeys()
        with open("private_key.pem", 'rb') as x:
            self.private_key = RSA.importKey(x.read())
        with open("public_key.pem", 'rb') as x:
            self.public_key = RSA.importKey(x.read())
        self.signer = PKCS1_signature.new(self.private_key)
        self.construct()
        time_init2 = time.clock()
    
    def statistical(self):
        import matplotlib.pyplot as plt
        len2gram=defaultdict(list)
        for gram in self.gram2inverted.keys():
            len2gram[len(self.gram2inverted[gram])].append(gram)
        x,y=[0,0,0],[]
        for l in sorted(list(len2gram.keys())):
            #x.append(l)
            #y.append(len(len2gram[l]))
            if l<=1000:
                x[0]+=len(len2gram[l])
            elif l<=10000:
                x[1]+=len(len2gram[l])
            else:
                x[2]+=len(len2gram[l])
        #print(sorted(list(len2gram.keys())))
        print(len(self.gram2inverted.keys()))
        plt.pie(x,labels=["x<=1000","1000<x<=10000","x>10000"])
        plt.show()
    
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
        ##step4: sign on the root hash H_t with DO’s private key
        self.S_dic=""
        #print("S_dic:",self.S_dic)
        #print("dic_root:",message)
        
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

        ##step4: signs the root hash value

        self.S_inv=""
        #print("inv_root:",message)
        #print("S_inv:",self.S_inv)
        
    def getKeys():
        rsa = RSA.generate(2048)
        #generate private key
        private_key = rsa.export_key("PEM")
        with open("private_key.pem", "wb") as x:
            x.write(private_key)
        print(private_key.decode('utf-8'))
        print("-" * 30 + "分割线" + "-" * 30)
        #generate public key
        public_key = rsa.publickey().export_key()
        with open("public_key.pem", "wb") as x:
            x.write(public_key)
        print(public_key.decode('utf-8'))
        
    def outsourced(self):
        return (self.D,self.q,self.S_dic,self.S_inv)
        
    def getPublicKey(self):
        return "public_key.pem"
