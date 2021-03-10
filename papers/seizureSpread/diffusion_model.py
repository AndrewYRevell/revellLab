#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:25:28 2021

@author: arevell
"""

fname = ospj(fpath_data,   f"sub-{subID}_{iEEG_filename}_{start_time_usec}_{stop_time_usec}_data.pickle" )
if (os.path.exists(fname)):
    with open(fname, 'rb') as f: all_data = pickle.load(f)

sfc_data = sfc( **all_data  ) 