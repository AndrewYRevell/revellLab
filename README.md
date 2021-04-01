# revellLab

Add this repository to python path:

1. Locate where this GitHub repository is stored on you computer. Example: /media/arevell/sharedSSD/linux/revellLab
2. Add this to your ~/.bashrc file (make sure to change the path above): export PYTHONPATH=$PYTHONPATH:/media/arevell/sharedSSD/linux
3. Now you can import any packages in this repository. Example: from revellLab.packages.eeg.echobase import echobase; import numpy as np; echobase.plot_adj(np.random.rand(5,5))
