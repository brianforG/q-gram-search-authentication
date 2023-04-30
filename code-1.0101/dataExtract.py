import unicodedata
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

def is_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

authors=[]
# with open("./data/dblp-2015-03-02.xml",encoding="utf-8") as f:
    # s=f.read()
    # #s=str(unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode())
    # #print(s[100000:120000])
    # for i in range(0,len(s),1000):
        # if "erhard Tr&#246;ster" in s[i:i+1000]:
            # print(s[i:i+1000])
    # res=re.findall(r'<author>[^<>]*</author>',s)
    # for i in tqdm(res):
        # author=i[8:-9]
        # #if len(author)>96:
            # #print(i,author,len(author))
        # if "erhard Tr&#246;ster" in author:
            # print(author)
        # authors.append(author)
# train=pd.read_csv("./data/data.tsv", sep='\t')
# #print(train["primaryName"])
# for index,row in tqdm(train.iterrows()):
    # author,profession=row["primaryName"],row["primaryProfession"]
    # if not isinstance(author,float) and not isinstance(profession,float) and ("actor" in profession.lower() or "actress" in profession.lower()):
        # #print(author,">>",author.split(" ")[-1])
        # if len(author)>2 and not is_contain_chinese(author):
            # authors.append(author)

# print(authors[:1])
# print(len(authors),"->",len(set(authors)))
# authors=sorted(list(set(authors)),key=lambda x:len(x))
# print(authors[0])
# print(authors[-1])
# print("min:",len(authors[0]),"max:",len(authors[-1]),"avg:",sum([len(x) for x in authors])/len(authors))
# with open("./data/data.txt","w",encoding="utf-8") as f:
    # f.write("\n".join(authors))
    
with open("./data/data.txt","r",encoding="utf-8") as f:
    authors=f.read().split("\n")
authors=[i.lower() for i in authors if len(i)>5]
authors=sorted(list(set(authors)),key=lambda x:(len(x),x))
print(len(authors))
authors=authors[:2600000]
print("min:",len(authors[0]),"max:",len(authors[-1]),"avg:",sum([len(x) for x in authors])/len(authors))
    
