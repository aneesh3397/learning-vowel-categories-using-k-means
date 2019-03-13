#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:48:39 2017

@author: aneeshnaik
"""
import numpy as np
from sklearn.cluster import KMeans
import itertools
import sys


if(len(sys.argv)==4):
    textname = sys.argv[1]
    n = int(sys.argv[2])
    init_file = sys.argv[3]

else:
    textname = sys.argv[1]
    n = int(sys.argv[2])

#reading in vowel data:
f = open(textname,'r')

v_list = f.read()
s = v_list.split('\n')

f_list = []
identity = []

#extracting f1 and f2 values:
for i in range(0,len(s)):
    split = s[i].split('\t')
    f1 = int(split[4]) 
    f2 = int(split[5])   
    f_list.append((f1,f2))
    identity.append(int(split[2]))


#reading in initialization file:
f = open(init_file,'r')
init = f.read()
s2 = init.split('\n')

init_list = []

for i in range(0,10):
    split = s2[i].split('\t')
    init_list.append((split[0],split[1]))
    

init_final = np.asarray(init_list)

print(init_final)
    

means = []

for i in range(0,len(init_list)):
  means.append((int(init_list[i][0]),int(init_list[i][1])))
  
  
#compiling data
final_data = []
    
for i in range(0,len(s)):
    final_data.append((f_list[i][0],f_list[i][1]))
     

#running k-means
kmeans = KMeans(n_clusters=n, init = init_final, max_iter=3000).fit(final_data)
print(kmeans.cluster_centers_,'\n' )


labels = kmeans.labels_

f = open("points.txt", "w")

for i in range(0,len(final_data)):
    f.write(str(final_data[i][0])+'   '+ str(final_data[i][1])+'   '+str(labels[i])+'\n')

means_list = kmeans.labels_.tolist()

pairs = []

for i in range(0,len(s)):
    pairs.append((identity[i], means_list[i]))
        

combos = list(itertools.combinations(pairs, 2))

#calculating precision recall and F-score:

TP = TN = FP = FN = 0

for i in range(0,len(combos)):
    if(combos[i][0][1]==combos[i][1][1] and combos[i][0][0]==combos[i][1][0]):
        TP+=1
    
    if(combos[i][0][1]==combos[i][1][1] and combos[i][0][0]!=combos[i][1][0]):
        FP+=1
        
    if(combos[i][0][1]!=combos[i][1][1] and combos[i][0][0]!=combos[i][1][0]):
        TN+=1
    
    if(combos[i][0][1]!=combos[i][1][1] and combos[i][0][0]==combos[i][1][0]):
        FN+=1

precision = TP/(TP+FP)
recall = TP/(TP+FN)
f_score = (2*precision*recall)/(precision+recall)

print("Precision: ", precision)
print("Recall: ", recall)
print("F-score: ", f_score)
    


    



    




