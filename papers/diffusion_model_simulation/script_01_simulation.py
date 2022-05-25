#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:56:25 2022

@author: arevell
"""

import sys
import os
import json
import copy
import time
import bct
import glob
import math
import random
import pickle
import scipy
import re
import pkg_resources
import pandas as pd
import numpy as np
import seaborn as sns
import nibabel as nib
import multiprocessing
import networkx as nx
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

from scipy import signal, stats
from scipy.io import loadmat
from itertools import repeat
from matplotlib import pyplot as plt
from os.path import join, splitext, basename

from packages.utilities import utils
from packages.diffusionModels import functions as dmf 


sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False, "axes.spines.bottom": False, "axes.spines.left": False, 'figure.dpi': 300, "legend.frameon": False, "savefig.transparent": True},  palette="pastel")
sns.set_context("talk")

#%%
#Paramters
N = 20 #number of nodes
time_steps_ltm = 8 # timesteps of LTM
theta_ltm = 0.5
#%%


#create network


G = nx.newman_watts_strogatz_graph(N, 2, p=0.5)
pos = nx.spring_layout(G, seed=1)


nx.draw_networkx(G) #draw network

G = dmf.add_edge_weights(G, max_weight = 5) #add edges
dmf.draw_networkx_weighted(G, pos = pos, edge_alpha = 0.6, draw_networkx_labels = False, node_size= 100)


# do diffusion model dimulation
adj = nx.to_numpy_array(G)
am = dmf.LTM(adj, seed = 16, time_steps = time_steps_ltm, threshold = 0.2)
am_colors_nodes = dmf.get_colors_from_activation_map_nodes(am)  
am_colors_edges = dmf.get_colors_from_activation_map_edsges(am, list(G.edges))



fig, ax = utils.plot_make(r= 2, c = 4)
ax = ax.flatten()
for t in range(len(ax)):
    if t < time_steps_ltm:
        dmf.draw_networkx_weighted(G, pos = pos, node_color = am_colors_nodes[t], edge_color = am_colors_edges[t], ax = ax[t], edge_alpha = 0.6, draw_networkx_labels = False, node_size= 100)
    else:
        ax[t].axis('off')

#%%
node_xyz = np.array([pos[v] for v in sorted(G)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(*node_xyz.T, s=100, ec="w")

# Plot the edges
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:gray")

dmf.format_axes(ax)


#%% 3D layout
spring_3D =nx.spring_layout(G,dim=3, seed=0)
#we need to seperate the X,Y,Z coordinates for Plotly
x_nodes = [spring_3D[i][0] for i in range(N)]# x-coordinates of nodes
y_nodes = [spring_3D[i][1] for i in range(N)]# y-coordinates
z_nodes = [spring_3D[i][2] for i in range(N)]# z-coordinates



edge_list = G.edges()
#we  need to create lists that contain the starting and ending coordinates of each edge.
x_edges=[]
y_edges=[]
z_edges=[]

#need to fill these with all of the coordiates
for edge in edge_list:
    #format: [beginning,ending,None]
    x_coords = [spring_3D[edge[0]][0], spring_3D[edge[1]][0], None]
    x_edges += x_coords

    y_coords = [spring_3D[edge[0]][1], spring_3D[edge[1]][1], None]
    y_edges += y_coords

    z_coords = [spring_3D[edge[0]][2], spring_3D[edge[1]][2], None]
    z_edges += z_coords

#create a trace for the edges
trace_edges = go.Scatter3d(x=x_edges,
                        y=y_edges,
                        z=z_edges,
                        mode='lines',
                        line=dict(color='black', width=2),
                        hoverinfo='none')
t=2
trace_nodes = go.Scatter3d(x=x_nodes,
                         y=y_nodes,
                        z=z_nodes,
                        mode='markers',
                        marker=dict(symbol='circle',
                                    size=10,
                                    color=am[t], #color the nodes according to their community
                                    colorscale=['#1f78b4','#9e2626'], #either green or mageneta
                                    line=dict(color='black', width=0)),
                        text=list(G.nodes()),
                        hoverinfo='text')

axis = dict(showbackground=True,
            showline=True,
            zeroline=False,
            showgrid=True,
            showticklabels=False,
            title='')

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1, y=1.25, z=0)
)

layout = go.Layout(title="",
                width=1920,
                height=1280,
                showlegend=False,
                scene=dict(xaxis=dict(axis),
                        yaxis=dict(axis),
                        zaxis=dict(axis),
                        camera = camera
                        ),
                margin=dict(t=100),
                hovermode='closest')


data = [trace_edges, trace_nodes]

pio.renderers.default='browser'
pio.renderers.default='svg'

fig = go.Figure(data=data, layout=layout)

#fig.update_scenes(camera_projection_type='orthographic')
fig.show()

def rotate_z(x, y, z, theta):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

x_eye = 1
y_eye = 1.25
z_eye = 1.25
for t in np.linspace(0,2*np.pi,20):
    xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=xe, y=ye, z=ze)
    )
    
    fig.update_layout(scene_camera=camera)
    fig.show()




frames=[]
for t in np.arange(0, 6.26, 0.1):
    xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
    frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
fig.frames=frames












