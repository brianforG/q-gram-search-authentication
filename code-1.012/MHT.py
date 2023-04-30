import time
from merkletools import *

def getHash(v):
    v = v.encode('utf-8')
    v = getattr(hashlib, "sha256")(v).hexdigest()
    return v

def h_t(t):
    #compute hash value h_t of the tuple t(sid,string) ∈ D
    a=bytearray.fromhex(t[0])
    b=bytearray.fromhex(t[1])
    return getattr(hashlib, "sha256")(a+b).hexdigest()

def isHash(v):
    if len(v)==64:
        c=set("0123456789abcdef")
        for i in range(len(v)):
            if v[i] not in c:
                return False
        return True
    else:
        return False
    

class MHT(MerkleTools):
    def reset_tree(self):
        self.values = list()    #leaf value
        self.leaves = list()    #leaf hash value 
        self.levels = None
        self.is_ready = False
    
    def reset_tree(self):
        self.leaves = list()
        self.values = list()
        self.levels = None
        self.is_ready = False
        
    def add_leaf(self, values):
        self.is_ready = False
        # check if single leaf
        if not isinstance(values, list):
            values = [values]
        for v in values:
            self.values.append(v)
            if isinstance(v,dict):
                v=h_t((getHash(list(v.keys())[0]),getHash(list(v.values())[0])))
            elif isinstance(v,tuple):
                if isinstance(v[1],MHT):
                    v=h_t((getHash(v[0]),v[1].get_merkle_root()))
                else:
                    v=h_t((getHash(v[0]),getHash(v[1])))
            else:
                v=getHash(v)
            v = bytearray.fromhex(v)
            self.leaves.append(v)
            
    def get_value(self, index):
        if isinstance(self.values[index], dict):
            return list(self.values[index].keys())[0]+":"+list(self.values[index].values())[0]
        return self.values[index]
        
    def get_proof(self, index_state):
        #state( 0:hash 1:value 2:(gram,hash(inv_list)) )
        if self.levels is None:
            return None
        elif not self.is_ready or not isinstance(index_state, list) or len(index_state)!=self.get_leaf_count():
            print("index_state size !=",self.get_leaf_count())
            return None
        else:
            VO=[]
            for idx in range(self.get_leaf_count()):
                if isinstance(index_state[idx],list):
                    gram,mht=self.get_value(idx)
                    VO.append((gram,mht.get_proof(index_state[idx])))
                elif index_state[idx]==0:
                    VO.append(self.get_leaf(idx))
                elif index_state[idx]==1:
                    VO.append(self.get_value(idx))
                elif index_state[idx]==2:
                    gram,inv=self.get_value(idx)
                    if isinstance(inv,MHT):
                        VO.append((gram,inv.get_merkle_root()))
                    else:
                        if isinstance(inv,list):
                            inv="_".join(inv)
                        VO.append((gram,getHash(inv)))
                else:
                    print("index_state error")
            VO=tuple(VO)
            for level in range(len(self.levels) - 2, -1, -1):
                VO_tmp=[]
                for idx in range(0,len(VO),2):
                    if idx+1==len(VO):
                        VO_tmp.append(VO[idx])
                    elif isinstance(VO[idx],str) and isinstance(VO[idx+1],str) and isHash(VO[idx]) and isHash(VO[idx+1]):
                        VO_tmp.append(self._to_hex(self.levels[level][idx//2]))
                    else:
                        VO_tmp.append((VO[idx],VO[idx+1]))
                VO=tuple(VO_tmp)
            if len(VO)==1:
                VO=VO[0]
            return self.VO2String(VO)
            
    def VO2String(self,VO):
        if isinstance(VO,tuple) and len(VO)==2:
            return "("+self.VO2String(VO[0])+","+self.VO2String(VO[1])+")"
        elif isinstance(VO,dict):
            return str(list(VO.keys())[0])+":"+str(list(VO.values())[0])
        elif isinstance(VO,str):
            return VO
        elif isinstance(VO,list):
            return "_".join(VO)
        else:
            print(VO,"error")
            assert(1==2)
    
    def String2VO(self,String):
        i,stack,dic,gram2inverted,vo_gram=0,[],{},{},[]
        if isHash(String):
            return String,dic,gram2inverted,sorted(vo_gram)
        gram,inv_list=[],[]
        while i<len(String):
            j=i
            if String[j]=="(":
                stack.append("(")
                i+=1
            elif String[j]==")":
                if isHash(stack[-3]) and isHash(stack[-1]):
                    s=h_t((stack[-3],stack[-1]))
                elif (not isHash(stack[-3])) and isHash(stack[-1]):
                    if stack[-3].isdigit():
                        inv_list.append(int(stack[-3]))
                    s=h_t((getHash(stack[-3]),stack[-1]))
                else:
                    print("error",stack)
                    assert(1==2)
                stack=stack[:-4]
                stack.append(s)
                i+=1
            elif String[j]==",":
                stack.append(",")
                i=j+1
            elif stack[-1]==",":
                while j<len(String) and String[j]!=")":
                    j+=1
                s=str(String[i:j])
                if ":" in s:
                    sid,string=s.split(":")
                    dic[sid]=string
                    s=h_t((getHash(sid),getHash(string)))
                elif not isHash(s):
                    if s.isdigit():
                        inv_list.append(int(s))
                    elif "_" in s:
                        inv_list=inv_list+s.split("_")
                    s=getHash(s)
                stack.append(s)
                i=j
            elif stack[-1]=="(":
                while j<len(String) and String[j]!=",":
                    j+=1
                s=str(String[i:j])
                if ":" in s:
                    sid,string=s.split(":")
                    stack.append(h_t((getHash(sid),getHash(string))))
                    dic[sid]=string
                else:
                    if not isHash(s):
                        if not s.isdigit():
                            gram.append(s)
                            if len(gram)>1:
                                if len(inv_list)!=0:
                                    gram2inverted[gram[-2]]=inv_list
                                    inv_list=[]
                                    gram=gram[:-2]+gram[-1:]
                        else:
                            inv_list.append(int(s))
                    stack.append(s)
                i=j
            else:
                print(stack,"error")
                assert(1==2)
        if len(gram)>0:
            if len(inv_list)!=0:
                gram2inverted[gram[-1]]=inv_list
                vo_gram=vo_gram+gram[:-1]
            else:
                vo_gram=vo_gram+gram
        if len(stack)>0 and len(stack[-1])==2 and isHash(stack[-1][0]) and isHash(stack[-1][1]):
            stack[-1]=h_t(stack[-1])
        return stack[0],dic,gram2inverted,sorted(vo_gram)
