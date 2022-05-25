#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:37:18 2022

@author: arevell
"""

import numpy as np
import pandas as pd
from visbrain.objects import (BrainObj, SceneObj, SourceObj, ConnectObj)
from visbrain.io import download_file

from packages.utilities import utils

from paths import constants_paths as paths

import seaborn as sns
from visbrain.gui import Brain
sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False, "axes.spines.bottom": False, "axes.spines.left": False, 'figure.dpi': 300, "legend.frameon": False, "savefig.transparent": True},  palette="pastel")
sns.set_context("talk")
#%
#Get data

centroids = pd.read_csv(paths.AAL2_CENTROIDS )
xyz = np.array(centroids[["x","y","z"]])


adj = utils.read_DSI_studio_Txt_files_SC(paths.SUB01_AAL2)
region_names = utils.get_DSIstudio_TXT_file_ROI_names_for_spheres(paths.SUB01_AAL2)
adj_hat = utils.log_normalize_adj(adj)

#%
n_sources = len(adj_hat)
conn = np.triu(adj/adj.max())
conn_select = 0.1 < conn
data=np.ones(n_sources)


#%%
sc = SceneObj(bgcolor="white",verbose = 2)

s_obj_1 = SourceObj('s1', xyz, data=data, radius_min=10, radius_max=20)
b_obj_1 = BrainObj('B3', translucent=True)

s_obj_1.color_sources(data=data, cmap='inferno')


sc.add_to_subplot(s_obj_1,  title='Animate multiple objects')
sc.add_to_subplot(b_obj_1,  rotate='right', use_this_cam=True, zoom=.7)




c_obj_1 = ConnectObj('c1', xyz, conn, select=conn_select, dynamic=(0., 1.),
                     dynamic_orientation='center', dynamic_order=3, cmap='bwr',
                     antialias=True)

sc.add_to_subplot(c_obj_1)

b_obj_1.animate()
s_obj_1.animate()


sc.preview()

sc.screenshot()

sc.record_animation('pics/animate_example1.png', n_pic=1)
#%%


vb = Brain(brain_obj= b_obj_1, source_obj=s_obj_1, connect_obj=c_obj_1 )

"""Render the scene and save the jpg picture with a 300dpi
"""
save_as =  'pics/0_main_brain.jpg'
vb.screenshot(save_as, dpi=300, print_size=(10, 10), autocrop=True)


