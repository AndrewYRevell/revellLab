#COMPUTER
HOME = "/Users/arevell"
FREESURFER_LICENSE = "$FREESURFER_HOME/license.txt"
IEEG_USERNAME_PASSWORD =  "/media/arevell/data/linux/ieegorg.json"
DSI_STUDIO_SINGULARITY = "/home/arevell/singularity/dsistudio/dsistudio_latest.sif"

#DATA
DATA = "/media/arevell/data/linux/data"

#METADATA
METADATA = "/media/arevell/data/linux/data/metadata"


FIGURES = "/media/arevell/data/linux/figures"

#GITHUB REVELLLAB
REVELLLAB = "/home/arevell/Documents/revellLab"
TOOLS = "/home/arevell/Documents/revellLab/tools"

#ATLASES and TEMPLATES
ATLASES = "/home/arevell/Documents/revellLab/tools/atlases/atlases"
ATLAS_LABELS = "/home/arevell/Documents/revellLab/tools/atlases/atlasLabels"
ATLAS_FILES_PATH =  "/home/arevell/Documents/revellLab/tools/atlases/atlasMetadata.json"
MNI_TEMPLATE = "/home/arevell/Documents/revellLab/tools/templates/MNI/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182.nii.gz"
MNI_TEMPLATE_BRAIN = "/home/arevell/Documents/revellLab/tools/templates/MNI/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182_brain.nii.gz"


#BIDS and DERIVATIVES
BIDS = "/media/arevell/data/linux/data/BIDS"
BIDS_DERIVATIVES_ATLAS_LOCALIZATION = "/media/arevell/data/linux/data/BIDS/derivatives/atlasLocalization"
BIDS_DERIVATIVES_QSIPREP = "/media/arevell/data/linux/data/BIDS/derivatives/qsiprep_v0.12.2"
BIDS_DERIVATIVES_RECONALL = "/media/arevell/data/linux/data/BIDS/derivatives/freesurferReconAll"

#General Data
#Tractography and structural connectivity
BIDS_DERIVATIVES_TRACTOGRAPHY = "/media/arevell/data/linux/data/BIDS/derivatives/structural_connectivity/tractography"
BIDS_DERIVATIVES_STRUCTURAL_MATRICES = "/media/arevell/data/linux/data/BIDS/derivatives/structural_connectivity/structural_matrices"


#Papers

#White matter iEEG paper
BIDS_DERIVATIVES_WM_IEEG = "/media/arevell/data/linux/data/BIDS/derivatives/white_matter_iEEG"
BIDS_DERIVATIVES_WM_IEEG_METADATA = "/media/arevell/data/linux/data/BIDS/derivatives/white_matter_iEEG/iEEGdata_WM_iEEG.json"
BIDS_DERIVATIVES_WM_IEEG_IEEG = "/media/arevell/data/linux/data/BIDS/derivatives/white_matter_iEEG/iEEG"
BIDS_DERIVATIVES_WM_IEEG_FUNCTIONAL_CONNECTIVITY = "/media/arevell/data/linux/data/BIDS/derivatives/white_matter_iEEG/functionalConnectivity"
BIDS_DERIVATIVES_WM_IEEG_FUNCTIONAL_CONNECTIVITY_IEEG = "/media/arevell/data/linux/data/BIDS/derivatives/white_matter_iEEG/functionalConnectivity/iEEG"
BIDS_DERIVATIVES_WM_IEEG_TRACTOGRAPHY = "/media/arevell/data/linux/data/BIDS/derivatives/white_matter_iEEG/tractography"

#Seizure Spread paper
BIDS_DERIVATIVES_WM_IEEG_METADATA = "/media/arevell/data/linux/data/BIDS/derivatives/white_matter_iEEG/iEEGdata_WM_iEEG.json"
DEEP_LEARNING_MODELS = "/media/arevell/data/linux/data/deepLearningModels/seizureSpread"
SEIZURE_SPREAD_ATLASES_PROBABILITIES = "/media/arevell/data/linux/data/SOZ_atlas_probabilities"



"""
#COMPUTER
HOME = "/Users/arevell"
FREESURFER_LICENSE = "$FREESURFER_HOME/license.txt"
IEEG_USERNAME_PASSWORD =  "/media/arevell/sharedSSD/linux/ieegorg.json"
DSI_STUDIO_SINGULARITY = "/home/arevell/singularity/dsistudio/dsistudio_latest.sif"

#DATA
DATA = "/media/arevell/sharedSSD/linux/data"

#METADATA
METADATA = "/media/arevell/sharedSSD/linux/data/metadata"


FIGURES = "/media/arevell/sharedSSD/linux/figures"

#GITHUB REVELLLAB
REVELLLAB = "/media/arevell/sharedSSD/linux/revellLab/"
TOOLS = "/media/arevell/sharedSSD/linux/revellLab//tools"

#ATLASES and TEMPLATES
ATLASES = "/media/arevell/sharedSSD/linux/revellLab//tools/atlases/atlases"
ATLAS_LABELS = "/media/arevell/sharedSSD/linux/revellLab//tools/atlases/atlasLabels"
ATLAS_FILES_PATH =  "/media/arevell/sharedSSD/linux/revellLab//tools/atlases/atlasMetadata.json"
MNI_TEMPLATE = "/media/arevell/sharedSSD/linux/revellLab//tools/templates/MNI/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182.nii.gz"
MNI_TEMPLATE_BRAIN = "/media/arevell/sharedSSD/linux/revellLab//tools/templates/MNI/mni_icbm152_t1_tal_nlin_asym_09c_182x218x182_brain.nii.gz"


#BIDS and DERIVATIVES
BIDS = "/media/arevell/sharedSSD/linux/data/BIDS"
BIDS_DERIVATIVES_ATLAS_LOCALIZATION = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/atlasLocalization"
BIDS_DERIVATIVES_QSIPREP = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/qsiprep_v0.12.2"
BIDS_DERIVATIVES_RECONALL = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/freesurferReconAll"


#White matter iEEG paper
BIDS_DERIVATIVES_WM_IEEG = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/white_matter_iEEG"
BIDS_DERIVATIVES_WM_IEEG_METADATA = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/white_matter_iEEG/iEEGdata_WM_iEEG.json"
BIDS_DERIVATIVES_WM_IEEG_IEEG = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/white_matter_iEEG/iEEG"
BIDS_DERIVATIVES_WM_IEEG_FUNCTIONAL_CONNECTIVITY = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/white_matter_iEEG/functionalConnectivity"
BIDS_DERIVATIVES_WM_IEEG_FUNCTIONAL_CONNECTIVITY_IEEG = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/white_matter_iEEG/functionalConnectivity/iEEG"
BIDS_DERIVATIVES_WM_IEEG_TRACTOGRAPHY = "/media/arevell/sharedSSD/linux/data/BIDS/derivatives/white_matter_iEEG/tractography"
"""