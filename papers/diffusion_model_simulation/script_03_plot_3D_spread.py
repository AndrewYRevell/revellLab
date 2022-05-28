#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 23:35:18 2022

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
import imageio

from scipy import signal, stats
from scipy.io import loadmat
from itertools import repeat
from matplotlib import pyplot as plt
from os.path import join, splitext, basename

from packages.utilities import utils
from packages.diffusionModels import functions as dmf 
from paths import constants_paths as paths

sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False, "axes.spines.bottom": False, "axes.spines.left": False, 'figure.dpi': 300, "legend.frameon": False, "savefig.transparent": True},  palette="pastel")
sns.set_context("talk")


#%%
#Get data




centroids = pd.read_csv(paths.AAL2_CENTROIDS )
xyz = np.array(centroids[["x","y","z"]])


adj = utils.read_DSI_studio_Txt_files_SC(paths.SUB01_AAL2)
region_names = utils.get_DSIstudio_TXT_file_ROI_names_for_spheres(paths.SUB01_AAL2)
adj_hat = adj/adj.max()
#adj_hat = utils.log_normalize_adj(adj)
#%
N= len(adj_hat)
adj_hat_select = 0.06 < adj_hat
adj_hat[~adj_hat_select]=0

#%%Get brain template

b = np.load(paths.BRAIN_TEMPLATE_B3)
list(b.keys())

template = [ b["vertices"],b["faces"], b["normals"] ]
template_vertices = template[0]
template_faces = template[1]

vert_x, vert_y, vert_z = template_vertices[:,0], template_vertices[:,1], template_vertices[:,2]
face_i, face_j, face_k = template_faces[:,0], template_faces[:,1], template_faces[:,2]


fig = go.Figure()
fig.add_trace(go.Mesh3d(x=vert_x, y=vert_y, z=vert_z, i=face_i, j=face_j, k=face_k,
                        color='gray', opacity=0.5, name='', showscale=False, hoverinfo='none'))



#%%
G = nx.from_numpy_array(adj_hat)
pos = nx.spring_layout(G, seed=1, dim=3)

for p in range(N):
    pos[p] = xyz[p]



node_xyz = np.array([pos[v] for v in sorted(G)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

#fig = plt.figure()
#ax = fig.add_subplot(111, projection="3d")
#ax.scatter(*node_xyz.T, s=100, ec="w")

# Plot the edges
#for vizedge in edge_xyz:
#    ax.plot(*vizedge.T, color="tab:gray")

#dmf.format_axes(ax)

#%%
# do diffusion model dimulation

am = dmf.gradient_LTM(adj_hat, seed = 40, time_steps = 30, threshold = 0.5)

am_select = 0.5 < am
am[~am_select]=0
am = am.astype(int)
am_colors_nodes = dmf.get_colors_from_activation_map_nodes(am)  
am_colors_edges = dmf.get_colors_from_activation_map_edges(am, list(G.edges))
am_colors_edges_binary = dmf.get_colors_from_activation_map_edges_binary(am, list(G.edges))



#%%


x_nodes = [pos[i][0] for i in range(N)]# x-coordinates of nodes
y_nodes = [pos[i][1] for i in range(N)]# y-coordinates
z_nodes = [pos[i][2] for i in range(N)]# z-coordinates



edge_list = G.edges()
edge_list_weights = np.array(list(nx.get_edge_attributes(G, 'weight').values()))*40
#we  need to create lists that contain the starting and ending coordinates of each edge.
x_edges=[]
y_edges=[]
z_edges=[]

#need to fill these with all of the coordiates
for edge in edge_list:
    #format: [beginning,ending,None]
    x_coords = [pos[edge[0]][0], pos[edge[1]][0], None]
    x_edges += x_coords

    y_coords = [pos[edge[0]][1], pos[edge[1]][1], None]
    y_edges += y_coords

    z_coords = [pos[edge[0]][2], pos[edge[1]][2], None]
    z_edges += z_coords

#%%

axis = dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='')



#%%
frame_count=0
rotation_per_frame = 2*np.pi/(3*30*6)

frames = []
#how many repeats the simulation is shown
for repeat in range(1):
    #t is the number of time steps gone through the activation map (am)
    for t in range(20):
    
        #make X number of frames per time_step
        for frame in range(1):
        
            traces={}
            
            count = 0
            for i in np.arange(0, len(x_edges)-1, 3):
                traces['trace_' + str(i)]= go.Scatter3d(x = [x_edges[i], x_edges[i+1]], 
                                                       y = [y_edges[i], y_edges[i+1]],
                                                       z = [z_edges[i], z_edges[i+1]],
                                                       line=dict(
                                                            color=am_colors_edges[t][count],
                                                            width=edge_list_weights[count]), 
                                                       hoverinfo='none')
                count =  count +1
                
                
            #create a trace for the edges without colors
            trace_edges = list(traces.values())
            """
            trace_edges = [go.Scatter3d(x=x_edges,
                                    y=y_edges,
                                    z=z_edges,
                                    mode='lines',
                                    line=dict(color=np.asarray(am_colors_edges_binary[t]),colorscale=[ '#1f78b4', '#9e2626'], cmin = 0, cmax = 1, cmid = 0.9,width=5),
                                    hoverinfo='none')]
            
            
            
            """
            
            
            
            
            if not len(np.unique(am[t])) == 1:
                trace_nodes = go.Scatter3d(x=x_nodes,
                                         y=y_nodes,
                                        z=z_nodes,
                                        mode='markers',
                                        marker=dict(symbol='circle',
                                                    size=15,
                                                    color=am[t], #color the nodes according to their community
                                                    colorscale=['#185b88','#751c1c'], #either green or mageneta
                                                    line=dict(color='black', width=0)),
                                        text=list(G.nodes()),
                                        hoverinfo='text')
            else:
                trace_nodes = go.Scatter3d(x=x_nodes,
                                         y=y_nodes,
                                        z=z_nodes,
                                        mode='markers',
                                        marker=dict(symbol='circle',
                                                    size=15,
                                                    color=am[t], #color the nodes according to their community
                                                    colorscale=['#751c1c','#751c1c'], #either green or mageneta
                                                    line=dict(color='black', width=0)),
                                        text=list(G.nodes()),
                                        hoverinfo='text')
            
            
            
            
            xeye, yeye, zeye = dmf.rotate_z(1.25, 1.25, -0.1, np.pi*.6 - frame_count*rotation_per_frame)
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=xeye, y=yeye, z=zeye)
            )
            
            layout = go.Layout(title="",
                           # width=800,
                           # height=800,
                            showlegend=False,
                            scene=dict(xaxis=dict(axis),
                                    yaxis=dict(axis),
                                    zaxis=dict(axis),
                            #        camera = camera
                                    ) ,
                            #margin=go.layout.Margin(
                            #        l=0, #left margin
                            #        r=0, #right margin
                            #        b=0, #bottom margin
                            #        t=0, #top margin
                            #        ),
                            #hovermode='closest'
                            )
            
            
            
            data=  trace_edges + [trace_nodes]
            if frame_count == 0:
                first_frame = copy.deepcopy(data)
            frames.append(go.Frame(data=data, layout= layout) )
            
            frame_count = frame_count + 1
            print(frame_count)
            #fig = go.Figure(data=data, layout=layout)
            #%%
xeye, yeye, zeye = dmf.rotate_z(1.25, 1.25, -0.1, np.pi*.6  )   
fig = go.Figure(
    data=first_frame,
    layout=go.Layout(
    scene=dict(xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=xeye, y=yeye, z=zeye))
            ) ,
    showlegend=False,
    title="Linear Threshold Model",
    updatemenus=[dict(
    type="buttons",
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None, {"frame": {"duration": 10},
                                "transition": {"duration": 10 }}]),
                 ],
        )]
        ),
    frames=frames
    )

fig.add_trace(go.Mesh3d(x=vert_x, y=vert_y, z=vert_z, i=face_i, j=face_j, k=face_k,
                        color='gray', opacity=0.1, name='', showscale=False, hoverinfo='none'))

#%%
#fig.update_scenes(camera_projection_type='orthographic')
#fig.show()
pio.renderers.default='browser'
pio.renderers.default='svg'

#fig.write_html("pics/animation/a_web.html", default_width="100%", default_height="100%")


for k in range(len(fig["frames"])):
    xeye, yeye, zeye = dmf.rotate_z(1.25, 1.25, -0.1, np.pi*0.6 - k*rotation_per_frame)   
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=xeye, y=yeye, z=zeye)
    )
    layout = go.Layout(title="",
                       width=800,
                       height=800,
                    showlegend=False,
                    scene=dict(xaxis=dict(axis),
                            yaxis=dict(axis),
                            zaxis=dict(axis),
                           camera = camera
                            ) ,
                    margin=go.layout.Margin(
                        l=0, #left margin
                            r=0, #right margin
                            b=0, #bottom margin
                            t=0, #top margin
                            ),
                    hovermode='closest'
                    )
    fig2 = go.Figure(data = frames[k]["data"], layout= layout)
   

    fig2.add_trace(go.Mesh3d(x=vert_x, y=vert_y, z=vert_z, i=face_i, j=face_j, k=face_k,
                            color='gray', opacity=0.1, name='', showscale=False, hoverinfo='none'))
    fig2.write_image(f"pics/animation/tmp/fig_{k:03d}.png", scale=1)
    print(k)
#%%

images_data = []
#load X images
for frame_count in range(540):
    if utils.checkIfFileExists(f"pics/animation/tmp/fig_{frame_count:03}.png", printBOOL=False):
        data = imageio.imread(f"pics/animation/tmp/fig_{frame_count:03}.png")
        images_data.append(data)

imageio.mimwrite(f"pics/animation/tmp/fig.gif", images_data, format= '.gif', fps = 20)



#%%







































