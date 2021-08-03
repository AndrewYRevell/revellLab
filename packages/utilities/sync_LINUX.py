import sys
import os
import time
import copy
from os import listdir
from os.path import join, isfile, splitext, basename
import glob
import dirsync



def checkPathError(path):
    """
    Check if path exists
    """
    if not os.path.exists(path):
        raise IOError(f"\n\n\n\nPath or file does not exist:\n{path}\n\n\n\n" )



def sync_BIDS(root_directory_server, root_directory_local, subdirectory, from_to = "server_to_local"):
    checkPathError(root_directory_server)
    checkPathError(root_directory_local)

    directory_server = join(root_directory_server, subdirectory)
    directory_local = join(root_directory_local, subdirectory)

    if from_to == "server_to_local":
        dirsync.sync(directory_server, directory_local, 'sync', verbose=True, ctime=True, create=True) #sync SERVER to LOCAL. Source is the SERVER
    elif from_to == "local_to_server":
        dirsync.sync(directory_local, directory_server, 'sync', verbose=True, ctime=True, create=True) #sync LOCAL to SERVER. Source is the LOCAL directory



#%%

root_directory_server = "/home/arevell/borel/DATA/Human_Data/BIDS"
root_directory_local = "/media/arevell/sharedSSD/linux/data/BIDS"


subdirectory = "PIER"
subdirectory = "derivatives/freesurferReconAll"
subdirectory = "derivatives/implantRenders"
subdirectory = "derivatives/atlasLocalization"
subdirectory = "derivatives/qsiprep"
subdirectory = "derivatives/tractography"

#SERVER to LOCAL
sync_BIDS(root_directory_server, root_directory_local, subdirectory, from_to = "server_to_local")

#LOCAL to SERVER
sync_BIDS(root_directory_server, root_directory_local, subdirectory, from_to = "local_to_server")

#%%



syncFolder = "PIER"
syncFolder = "PIER"


LINUX_BIDSserver = join("/home/arevell/borel/DATA/Human_Data/BIDS", syncFolder)
LINUX_BIDSlocal = join("/media/arevell/sharedSSD/linux/data/BIDS", syncFolder)


BIDSserver = LINUX_BIDSserver
BIDSlocal = LINUX_BIDSlocal

# %%


dirsync.sync(BIDSserver, BIDSlocal, 'sync', verbose=True, ctime=True) #sync BIDSserver TO BIDSlocal. Source: BIDSserver
dirsync.sync(BIDSlocal, BIDSserver, 'sync', verbose=True, ctime=True) #sync local to server
#%%


