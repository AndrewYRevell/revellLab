#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:25:00 2022

@author: arevell
"""



import os
import sys
import copy
import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy import interpolate


def preprocess_adj(adj, threshold = 0.4, log_normalize = False, binarize = False):
    """
    adj : 
    threshold :
    log_normalize : 
    binarize : 
    """
    #log normalizing
    if log_normalize:
        adj[np.where(adj == 0)] = 1
        adj = np.log10(adj)
    adj = adj/np.max(adj)
    
    threshold = 0.4 #bottom X percent of edge weights are eliminated
    C_thresh = copy.deepcopy(adj)

    number_positive_edges = len(np.where(C_thresh > 0)[0])
    cutoff = int(np.round(number_positive_edges*threshold))

    positive_edges = C_thresh[np.where(C_thresh > 0)]
    cutoff_threshold = np.sort(positive_edges)[cutoff]
    C_thresh[np.where(C_thresh < cutoff_threshold)] = 0
    if binarize:
        C_thresh[np.where(C_thresh >= cutoff_threshold)] = 1
    return C_thresh


def LTM(adj, seed = 0, time_steps = 50, threshold = 0.2):
    
    N = len(adj) #number of nodes
    node_state = np.zeros(shape = (time_steps, N))
    node_state[0, seed] = 1 #make seed active

    #neighbor_sum_distribution = np.zeros(shape = (time_steps, N))
    for t in range(1, time_steps):
        #print(t)
        for i in range(N): #loop thru all nodes
            #find neighbors of node i
            previous_state = node_state[t-1,i]
            neighbors = np.where(adj[i,:] > 0)
            neighbors_weight = adj[i,neighbors] 
            neighbors_state = node_state[t-1, neighbors]
            neighbors_sum = np.sum(neighbors_weight * neighbors_state)
            strength = np.sum(neighbors_weight)
            if neighbors_sum >= threshold*strength: #if sum is greater than threshold, make that node active
                node_state[t, i] = 1
            if neighbors_sum < threshold*strength:
                node_state[t, i] = 0
            if previous_state == 1:
                node_state[t, i] = 1
    node_state = node_state.astype(int)
    return node_state

def gradient_diffusion(adj, seed = 0, time_steps = 50, gradient = 0.1):
    
    N = len(adj) #number of nodes
    node_state = np.zeros(shape = (time_steps, N))
    node_state[0, seed] = 1 #make seed active

    for t in range(1, time_steps):
        #print(t)
        for i in range(N): #loop thru all nodes
            #find neighbors of node i
            previous_state = node_state[t-1,i]
            neighbors = np.where(adj[i,:] > 0)
            neighbors_weight = adj[i,neighbors] 
            neighbors_state = node_state[t-1, neighbors]
            neighbors_sum = np.sum(neighbors_weight * neighbors_state * gradient)
            
            # Add the cumulative sum from the neighbors to the node's current (i.e. previous) state. If greater than 1, make it 1
            node_state[t, i] = previous_state + neighbors_sum
            if node_state[t, i] > 1: node_state[t, i] = 1
    return node_state

def gradient_LTM(adj, seed = 0, time_steps = 50, gradient = 0.1, threshold = 0.2):
    
    N = len(adj) #number of nodes
    node_state = np.zeros(shape = (time_steps, N))
    node_state[0, seed] = 1 #make seed active

    for t in range(1, time_steps):
        #print(t)
        for i in range(N): #loop thru all nodes
            #find neighbors of node i
            previous_state = node_state[t-1,i]
            neighbors = np.where(adj[i,:] > 0)
            neighbors_weight = adj[i,neighbors] 
            neighbors_state = node_state[t-1, neighbors]
            neighbors_sum = np.sum(neighbors_weight * neighbors_state * gradient)
        
            # Add the cumulative sum from the neighbors to the node's current (i.e. previous) state. If greater than 1, make it 1
            node_state[t, i] = previous_state + neighbors_sum
            if node_state[t, i] > 1: 
                node_state[t, i] = 1
            if node_state[t, i] >= threshold: #if sum is greater than threshold, make that node active
                node_state[t, i] = 1
    return node_state


#%%

def add_edge_weights(G, distribution = "random_uniform_cont", min_weight = 1, max_weight = 10, mu = 0, sigma = 1,alpha = 1, beta = 1, lambd = 1, kappa = 1):
    """ Add edge weights to G based on a specified distribution
    G:
    distribution: """
    edges = G.edges(data=True)
    #create edge weights
    if distribution == "random_uniform_int":
        for (node1,node2, ew) in edges:
            ew = int(np.round(random.uniform(min_weight, max_weight)))
            G.add_edge(node1,node2,weight=ew)
            
    if distribution == "random_uniform_cont":
        for (node1,node2, ew) in edges:
            ew = random.uniform(min_weight, max_weight)
            G.add_edge(node1,node2,weight=ew)
            
    if distribution == "gauss":
        for (node1,node2, ew) in edges:
            ew = random.gauss(mu, sigma)
            G.add_edge(node1,node2,weight=ew)
            
    if distribution == "gauss_positive":
        for (node1,node2, ew) in edges:
            ew = abs(random.gauss(mu, sigma))
            G.add_edge(node1,node2,weight=ew)    
            
    if distribution == "betavariate":
        for (node1,node2, ew) in edges:
            ew = random.betavariate(alpha, beta)
            G.add_edge(node1,node2,weight=ew)

    if distribution == "expovariate":
        for (node1,node2, ew) in edges:
            ew = random.expovariate(lambd)
            G.add_edge(node1,node2,weight=ew)

    if distribution == "gammavariate":
        for (node1,node2, ew) in edges:
            ew = random.gammavariate(alpha, beta)
            G.add_edge(node1,node2,weight=ew)
            
    if distribution == "lognormvariate":
        for (node1,node2, ew) in edges:
            ew = random.lognormvariate(mu, sigma)
            G.add_edge(node1,node2,weight=ew)
            
            
    if distribution == "vonmisesvariate":
        for (node1,node2, ew) in edges:
            ew = random.vonmisesvariate(mu, kappa)
            G.add_edge(node1,node2,weight=ew)
                      
    if distribution == "paretovariate":
        for (node1,node2, ew) in edges:
            ew = random.paretovariate(alpha)
            G.add_edge(node1,node2,weight=ew)
            
    if distribution == "weibullvariate":
        for (node1,node2, ew) in edges:
            ew = random.weibullvariate(alpha, beta)
            G.add_edge(node1,node2,weight=ew)
             
    return G
        

#%% plotting

def get_colors_from_activation_map_nodes(am, color_dict = {"0": "#1f78b4", "1": "#9e2626"}):
    nrow, ncol = am.shape
    colors= [[0]*ncol for i in range(nrow)]
    for r in range(nrow):
        row = am[r]
        for c in range(ncol):
            activity = row[c]
            colors[r][c] = color_dict[str(activity)]
    return colors
        
def get_colors_from_activation_map_edsges(am, edge_list, color_dict = {"0": "#1f78b4", "1": "#9e2626"}):
    nrow, ncol = am.shape
    nedges = len(edge_list)
    colors= [[color_dict["0"]]*nedges for i in range(nrow)]
    
    #turn colors between active nodes to different color 
    for r in range(nrow):
        row = am[r]
        for i, e in enumerate(edge_list):
            n1, n2 = e
            if row[n1] == 1 and row[n2] == 1:
                colors[r][i] = color_dict["1"]
    return colors


def draw_networkx_weighted(G, pos = None, node_names = None, node_size = 300, node_color = '#1f78b4', edge_color = "black", node_alpha = 1, linewidths = 0, edge_alpha = 1, ax = None, draw_networkx_labels = True):
    
    widths = nx.get_edge_attributes(G, 'weight')
    nodelist = G.nodes()
    if pos == None:
        pos = nx.spring_layout(G)
    if node_names == None:
        node_names = nodelist
    
    nx.draw_networkx_nodes(G, pos,
                           nodelist=nodelist,
                           node_size=node_size,
                           node_color=node_color,
                           alpha=node_alpha,
                           linewidths = linewidths, ax = ax)
    nx.draw_networkx_edges(G, pos,
                           edgelist = widths.keys(),
                           width=list(widths.values()),
                           edge_color=edge_color,
                           alpha=edge_alpha, ax = ax)
    if draw_networkx_labels:
        nx.draw_networkx_labels(G, pos=pos,
                                labels=dict(zip(nodelist,node_names)),
                                font_color='black', ax = ax)



"""

widths = nx.get_edge_attributes(G, 'weight')
nodelist = G.nodes()
pos = nx.spring_layout(G)

"""