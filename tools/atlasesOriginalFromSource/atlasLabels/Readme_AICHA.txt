E-mail: aicha.gin.brainatlas@gmail.com
Version: V1
Date: December 7th, 2015

We present AICHA (for Atlas of Intrinsic Connectivity of Homotopic Areas), a functional brain ROIs atlas based on resting-state fMRI data acquired in 281 individuals. AICHA ROIs cover the whole cerebrum, each having 1- homogeneity of its constituting voxels intrinsic activity, and 2- a unique homotopic contralateral counterpart with which it has maximal intrinsic connectivity.

The atlas is fully described in the following publication: 
Joliot M, Jobard G, Naveau M, Delcroix N, Petit L, Zago L, Crivello F, Mellet E, Mazoyer B, Tzourio-Mazoyer N (2015) AICHA: An atlas of intrinsic connectivity of homotopic areas. J Neurosci Methods 254:46-59.

This atlas is protected by copyright; you can freely use it for none profit research purposes, providing the above reference is cited. For other use please contact us through aicha.gin.brainatlas@gmail.com

The AICHA atlas includes 192 couples of homotopic regions for a total of 384 regions. AICHA is provided in the MNI stereotaxic space (MNI ICBM 152, Template sampling size of 2x2x2 mm3 voxels; bounding box, x = -90 to 90 mm, y = -126 to 91 mm, z = -72 to 109 mm). 
Each region get a pseudo-color with odd number for region belong to the left hemisphere and even for the right. Each homotopic pair is labeled with an odd number (Left) and the following even number (Right). For example: “1 and 2” code for G_Frontal_Sup-1-L and G_Frontal_Sup-1-R respectively, “3 and 4” code for G_Frontal_Sup-2-L and G_Frontal_Sup-2-R, …

AICHA atlas includes both regions located in the crown of the gyri (named Gyrus, region name beginning by “G_”) and regions located in the depth of the sulci (named Suclus, region name beginning by “S_”). In addition the subcortical nuclei were labeled separately (name Nucleus, region name beginning by “N_”). 
Different parcels belonging to the same anatomical region were labeled with numbers (starting to 1). For example the precuneus show as 9 subparts labeled from G_Precuneus-1-L to G_Precuneus-9-L.

Format of AICHA
AICHA is provided as a nifti+ file with associated files for AAL(*) toolbox (Tzourio-Mazoyer 2002), SPM12 and mricron. 

Files of AICHA distribution
- Readme_AICHA.txt: This file.
- AICHA.nii: Volumetric atlas nifti+ format
- AICHA_vol1.txt: text files describing the features of each regions: 
1.	nom_c: short name
2.	nom_l: long name (identical to nom_c for compatibility with AAL toolbox)
3.	color: pseudocolor_index used in the “AICHA.nii” file
4.	vol_vox: volume of the region (voxel)	
5.	vol_mm3: volume of the region (mm3)	
6.	xv: x MNI coordinate of the mass center of the region (voxel)
7.	yv: y MNI coordinate of the mass center of the region (voxel)
8.	zv: z MNI coordinate of the mass center of the region (voxel)
9.	xmm: x MNI coordinate of the mass center of the region (mm)
10.	ymm: y MNI coordinate of the mass center of the region (mm)
11.	zmm: z MNI coordinate of the mass center of the region (mm)
- AICHA.xml: file to be used with SPM12. 
- AICHA_ROI_MNI_V1.txt is to be used with the AAL software for SPM12 (provided through http://www.gin.cnrs.fr/AAL2)
- AICHA_Border.mat, AICHA_List.mat, AICHA_vol.mat: file to be used with the AAL(*) toolbox (Tzourio-Mazoyer 2002). 
- AICHAmc.nii.gz, AICHAmc.nii.txt, AICHAmc.nii.lut: Files to be used with mricron visualizer. To accommodate the limits of 255 labeled regions both homotopic pair were affected the same pseudo-color and same name.
- embeddedAICHAmc.nii.gz: file to be used with MRIcroGL visualizer (https://www.nitrc.org/projects/mricrogl/)

The authors: GIN, UMR5296, 2015

Bibliography:
Tzourio-Mazoyer N, Landeau B, Papathanassiou D, Crivello F, Etard O, Delcroix N, Mazoyer B, Joliot M (2002) Automated anatomical labeling of activations in SPM using a macroscopic anatomical parcellation of the MNI MRI single-subject brain. Neuroimage 15:273-289.

(*) AAL toolbox can be downloaded at: http://www.gin.cnrs.fr/AAL and http://www.gin.cnrs.fr/AAL2

Acknowledgements
The authors are grateful to Dr Chris Rorden for his helpful contribution of the render in MRIcroGL.