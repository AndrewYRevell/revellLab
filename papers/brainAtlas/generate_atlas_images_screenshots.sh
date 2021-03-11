#!/bind/bash

ATLAS_DIRECTORY=/Users/andyrevell/deepLearner/home/arevell/Documents/01_papers/paper001/data_raw/atlases/standard_atlases
FIGURE_DIRECTORY=/Users/andyrevell/deepLearner/home/arevell/Documents/01_papers/paper001/paper001/figures/atlas_screenshots
ATLAS_DIR_RANDOM=/Users/andyrevell/deepLearner/home/arevell/Documents/01_papers/paper001/data_raw/atlases/random_atlases
ATLAS_DIR_overlap=/Users/andyrevell/deepLearner/home/arevell/Documents/01_papers/paper001/data_raw/AAL-JHU_combination
Other_Atlases_directory=/Users/andyrevell/deepLearner/home/arevell/Documents/01_papers/paper001/data_raw/atlases_for_figures
RAS_directory=/Users/andyrevell/deepLearner/home/arevell/Documents/01_papers/paper001/data_raw/atlases/original_atlases_in_RAS_orientation
#coordinates
x=8.953007
y=-10.28403
z=10.2997

#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/AAL.png $ATLAS_DIRECTORY/AAL3v1_1mm.nii.gz -cm brain_colours_flow
#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/AICHA.png $ATLAS_DIRECTORY/AICHA.nii.gz -cm brain_colours_nih_fire
#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/Hammersmith.png $ATLAS_DIRECTORY/Hammersmith_atlas_n30r83_SPM5.nii.gz -cm brain_colours_nih_ice_iso
#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/JHU.png $ATLAS_DIRECTORY/JHU-ICBM-labels-1mm.nii.gz -cm brain_colours_nih_ice_iso
#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/Schaefer7_0100.png $ATLAS_DIRECTORY/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz -cm subcortical
#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/Juelich.png $ATLAS_DIRECTORY/Juelich-maxprob-thr25-1mm.nii.gz -cm subcortical
#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/MMP.png $ATLAS_DIRECTORY/MMP_in_MNI_corr.nii.gz -cm subcortical
#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/DKT.png $ATLAS_DIRECTORY/OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_v2.nii.gz -cm subcortical

#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/Talairach_coronal.png $ATLAS_DIRECTORY/Talairach-labels-1mm.nii.gz -cm brain_colours_nih
#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -yh -zc 0 0 -zz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/Talairach_axial.png $ATLAS_DIRECTORY/Talairach-labels-1mm.nii.gz -cm brain_colours_nih

#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/HO_cort.png $ATLAS_DIRECTORY/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz -cm brain_colours_nih



#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/brodmann_res.png $Other_Atlases_directory/brodmann_res-1x1x1.nii.gz -cm subcortical

fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/BN_atlas.png $Other_Atlases_directory/BN_Atlas_246_1mm.nii.gz -cm brain_colours_flow
fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/Yeo2011_7.png $RAS_directory/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz -cm brain_colours_flow
fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/Yeo2011_7_liberal.png $RAS_directory/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz -cm brain_colours_flow
fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/MMP.png $RAS_directory/MMP_in_MNI_corr.nii -cm brain_colours_flow



#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/RandomAtlas0000100_v0001.png $ATLAS_DIR_RANDOM/RandomAtlas0000100/RandomAtlas0000100_v0001.nii.gz -cm color_map71 #brain_colours_cardiac_iso
#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/RandomAtlas0001000_v0001.png $ATLAS_DIR_RANDOM/RandomAtlas0001000/RandomAtlas0001000_v0001.nii.gz -cm color_map71 #brain_colours_cardiac_iso
#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/RandomAtlas0000010_v0001.png $ATLAS_DIR_RANDOM/RandomAtlas0000010/RandomAtlas0000010_v0001.nii.gz -cm brain_colours_cardiac_iso
#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/RandomAtlas0010000_v0001.png $ATLAS_DIR_RANDOM/RandomAtlas0010000/RandomAtlas0010000_v0001.nii.gz -cm brain_colours_cardiac_iso

#note Yeo atlases have different x y z orientation (-yx --> -zh option):  
#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -yh -zc 0 0 -zz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/Yeo7.png $Other_Atlases_directory/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz -cm brain_colours_diverging_bwr #Yeo needs to be flipped
#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/glasser.png $Other_Atlases_directory/glasser_res-1x1x1.nii.gz -cm brain_colours_flow_iso #brain_colours_cardiac_iso

#Overlap Image
#x=18.58511
#y=-13.35112
#z=4.946755

zoom=870
size_x=3000
size_y=1000

#fsleyes render --scene ortho --worldLoc $x $y $z --outfile $FIGURE_DIRECTORY/overlap_AAL_JHU.png --displaySpace $ATLAS_DIRECTORY/AAL.nii.gz -sz $size_x $size_y -xc 0 0 -yc 0 0 -zc 0 0 -xz $zoom -yz $zoom -zz $zoom -hl --layout horizontal -hc --bgColour 1 1 1 $ATLAS_DIRECTORY/AAL.nii.gz --name "AAL" --overlayType volume --alpha 100.0 --brightness 50.0 --contrast 50.0 --cmap blue-lightblue --negativeCmap greyscale --displayRange 0.0 116.0 --clippingRange 0.0 117.16 --gamma 0.0 --cmapResolution 256 --interpolation none --numSteps 100 --blendFactor 0.1 --smoothing 0 --resolution 100 --numInnerSteps 10 --clipMode intersection --volume 0 $ATLAS_DIRECTORY/JHU-ICBM-labels-1mm.nii.gz --name "JHU" --overlayType volume --alpha 100.0 --brightness 50.0 --contrast 50.0 --cmap red --negativeCmap greyscale --displayRange 0.0 48.0 --clippingRange 0.0 48.48 --gamma 0.0 --cmapResolution 256 --interpolation none --numSteps 100 --blendFactor 0.1 --smoothing 0 --resolution 100 --numInnerSteps 10 --clipMode intersection --volume 0 $ATLAS_DIR_overlap/niiEmptyEdit.nii --name "NoCoverage" --overlayType volume --alpha 100.0 --brightness 50.0 --contrast 50.0 --cmap brain_colours_nih --negativeCmap greyscale --useNegativeCmap --displayRange 0.0 7000.0 --clippingRange 0.0 7070.0 --gamma 0.0 --cmapResolution 256 --interpolation none --numSteps 100 --blendFactor 0.1 --smoothing 0 --resolution 100 --numInnerSteps 10 --clipMode intersection --volume 0 $ATLAS_DIR_overlap/niiOverlapEdit.nii --name "Overlap" --overlayType volume --alpha 100.0 --brightness 50.0 --contrast 50.0 --cmap yellow --negativeCmap greyscale --displayRange 0.0 7000.0 --clippingRange 0.0 7070.0 --gamma 0.0 --cmapResolution 256 --interpolation none --numSteps 100 --blendFactor 0.1 --smoothing 0 --resolution 100 --numInnerSteps 10 --clipMode intersection --volume 0



#Population Specific Atlas/Study Specific Atlas: Melbourne Children's Regional Infant Brain Atlas
x=11.12859
y=63.9754
z=0.002661852

#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/Melbourne_Childrens_Regional_Infant_Brain_atlas.png $Other_Atlases_directory/Melbourne_Childrens_Regional_Infant_Brain_atlas/M-CRIB_orig_P01_parc_edited.nii.gz -cm Render3 #brain_colours_cardiac_iso

x=-2.335776
y=-55.59925
z=-46.08542
#fsleyes render -wl $x $y $z -hc -bg 1 1 1 -xh -zh -yc 0 0 -yz 900 -hl -sz 400 500 --outfile $FIGURE_DIRECTORY/Cerebellum.png $Other_Atlases_directory/Cerebellum-MNIflirt-maxprob-thr25-1mm.nii.gz -cm brain_colours_cardiac_iso
