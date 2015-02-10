
# coding: utf-8

# Problem 3: WINKLR
# ==========
# Winklr is a curiously popular social network for fans of the sitcom Happy Days. Users can post photos, write messages, and most importantly, follow each other’s posts and content. This helps users keep up with new content from their favorite users on the site.
# 
# To help its users discover new people to follow on the site, Winklr is building a new machine learning system, called the Fonz, to predict who a given user might like to follow.
# 
# **WINKLR Data set**
# 
# Phase one of the Fonz project is underway. The engineers can export the entire user graph as tuples. For example, the tuple “user1,user2” means that user1 follows user2. (user2 does not necessarily follow user1 in this case.)
# 
# Furthermore, an engineer has examined users who click frequently on other users who they do not already follow. She has created a data set with a large number of “user1,user2” tuples, where user1 has clicked frequently on user2’s content but does not yet follow user2.
# 
# You have joined the Fonz project to implement Phase two, which improves on this result. Given the user graph and the list of frequent-click tuples here, you will select a subset of those frequent-click tuples that look most promising. These tuples must be the “user1,user2” tuples where you believe user1 is mostly likely to want to follow user2, given the information in the user graph.
# 
# You have been asked to select 70,000 tuples. These tuples will be used in an email campaign, inviting the targeted users to follow the users you recommend.
# 
# **WINKLR Deliverables**
# 
# The 70,000 most liking pairs must be placed in a CSV file called problem3.csv, with one “user1,user2” pair on each line (without quotes). The pairs must be in order of most likely to want to connect to least likely to want to connect, i.e. the first line of the file should contain the pair of IDs that are most likely to want to connect, and the last line should contain the pair of IDs that are least likely (of the 70,00 selected) to want to connect.

import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx
import pylab as pl
import pickle
import os
os.chdir('yourdatafolderhere')
#get_ipython().magic(u'matplotlib inline')


# ---------------------------------------------------------
# **Read the whole graph**
# ---------------------------------------------------------

d = {}
with open('Winklr-network.csv','r') as openfile:
    for k,line in enumerate(openfile):
        #if k<1000:
            if line.split(',')[0] not in d:
                d[line.split(',')[0]] = [line.split(',')[1][:-1]]
            else:
                d[line.split(',')[0]].append(line.split(',')[1][:-1])
            if k % 100000 == 0:
                print k,


#use networkx to create a graph                
G=nx.DiGraph(d)


# create a file with shortest path for each tuple in the test set
dtest = []
with open('Winklr-topClickPairs.csv','r') as openfile:
    with open('topClickPairs_shortest_path.csv','w') as outfile:
        for k,line in enumerate(openfile):
            #if k<1000:
                try:
                    sp = nx.shortest_path_length(G,line.strip().split(',')[0],
                                            line.strip().split(',')[1])
                except:
                    sp = 999
                outfile.write('%s,%i\n'%(line.strip(),sp))
                if k % 100000 == 0:
                    print k,
                    

#create a file with shortest path for each neighbor of each follower in the test set
with open('Winklr-topClickPairs.csv','r') as openfile:
    with open('topClickPairs_shortest_path_all2.csv','w') as outfile:
        for k,line in enumerate(openfile):
            neighbors = G.neighbors(line.strip().split(',')[0])
            #neighbors.remove(line.strip().split(',')[0])
            for neighbor in neighbors:
                try:
                    sp = nx.shortest_path_length(G,neighbor,
                                            line.strip().split(',')[1])
                except:
                    sp = 999
                outfile.write('%s,%s,%i\n'%(line.strip(),neighbor,sp))
            if k % 10000 == 0:
                print k,


# import the neighbors shortest path
neighbors = pd.read_csv('topClickPairs_shortest_path_all2.csv',header=None,names=['follower','followed','neighbor','sp'])

#distribution of shortest paths amongst neighbors
neighbors.groupby(['sp']).follower.count().plot(kind='bar')

#remove followers as neighbors
neighbors = neighbors[neighbors.follower!=neighbors.neighbor]

# calculate mean shortest path and support
f = {'sp': ['mean','count']}
n_neigh_connected = neighbors[(neighbors.sp < 999) & (neighbors.sp >1)].groupby(['follower','followed']).agg(f).reset_index()

n_neigh_connected.columns = ['follower','followed','mean_sp','count']

n_neigh_connected.head()

shortest_path = pd.read_csv('topClickPairs_shortest_path.csv',header=None,names=['follower','followed','sp'])
shortest_path.groupby(['sp']).follower.count().plot(kind='bar')

# merge shortest path and neighbors shortest path
final = pd.merge(left=shortest_path,right=n_neigh_connected,how='left',on=['follower','followed'])

#remove some granularity since this is essentially going to be a sort I don't want more than first digit to be a factor
final['mean_sp'] =final['mean_sp'].apply(round,ndigits=1)

final_exp = final[final.sp>0].sort(columns=['sp','mean_sp','count'],ascending=[True, True, False], inplace=False)

final_exp.head()

final_exp.to_csv('solution.csv',index=False)

final_exp[:70000][['follower','followed']].to_csv('problem3.csv',index=False,header=False)








