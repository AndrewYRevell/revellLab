

#---------------------------------
# New invocation of recon-all Wed Mar 24 16:39:33 EDT 2021 
#--------------------------------------------
#@# MotionCor Wed Mar 24 16:39:34 EDT 2021

 cp /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/orig/001.mgz /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/rawavg.mgz 


 mri_convert /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/rawavg.mgz /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/orig.mgz --conform 


 mri_add_xform_to_header -c /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/transforms/talairach.xfm /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/orig.mgz /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/orig.mgz 

#--------------------------------------------
#@# Talairach Wed Mar 24 16:39:43 EDT 2021

 mri_nu_correct.mni --no-rescale --i orig.mgz --o orig_nu.mgz --ants-n4 --n 1 --proto-iters 1000 --distance 50 


 talairach_avi --i orig_nu.mgz --xfm transforms/talairach.auto.xfm 

talairach_avi log file is transforms/talairach_avi.log...

 cp transforms/talairach.auto.xfm transforms/talairach.xfm 

lta_convert --src orig.mgz --trg /home/arevell/freesurfer/average/mni305.cor.mgz --inxfm transforms/talairach.xfm --outlta transforms/talairach.xfm.lta --subject fsaverage --ltavox2vox
#--------------------------------------------
#@# Talairach Failure Detection Wed Mar 24 16:43:25 EDT 2021

 talairach_afd -T 0.005 -xfm transforms/talairach.xfm 


 awk -f /home/arevell/freesurfer/bin/extract_talairach_avi_QA.awk /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/transforms/talairach_avi.log 


 tal_QC_AZS /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/transforms/talairach_avi.log 

#--------------------------------------------
#@# Nu Intensity Correction Wed Mar 24 16:43:25 EDT 2021

 mri_nu_correct.mni --i orig.mgz --o nu.mgz --uchar transforms/talairach.xfm --proto-iters 1000 --distance 50 --n 1 --ants-n4 


 mri_add_xform_to_header -c /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/transforms/talairach.xfm nu.mgz nu.mgz 

#--------------------------------------------
#@# Intensity Normalization Wed Mar 24 16:47:02 EDT 2021

 mri_normalize -g 1 -seed 1234 -mprage nu.mgz T1.mgz 

#--------------------------------------------
#@# Skull Stripping Wed Mar 24 16:48:18 EDT 2021

 mri_em_register -skull nu.mgz /home/arevell/freesurfer/average/RB_all_withskull_2020_01_02.gca transforms/talairach_with_skull.lta 


 mri_watershed -T1 -brain_atlas /home/arevell/freesurfer/average/RB_all_withskull_2020_01_02.gca transforms/talairach_with_skull.lta T1.mgz brainmask.auto.mgz 


 cp brainmask.auto.mgz brainmask.mgz 

#-------------------------------------
#@# EM Registration Wed Mar 24 16:53:00 EDT 2021

 mri_em_register -uns 3 -mask brainmask.mgz nu.mgz /home/arevell/freesurfer/average/RB_all_2020-01-02.gca transforms/talairach.lta 

#--------------------------------------
#@# CA Normalize Wed Mar 24 16:55:41 EDT 2021

 mri_ca_normalize -c ctrl_pts.mgz -mask brainmask.mgz nu.mgz /home/arevell/freesurfer/average/RB_all_2020-01-02.gca transforms/talairach.lta norm.mgz 

#--------------------------------------
#@# CA Reg Wed Mar 24 16:56:35 EDT 2021

 mri_ca_register -nobigventricles -T transforms/talairach.lta -align-after -mask brainmask.mgz norm.mgz /home/arevell/freesurfer/average/RB_all_2020-01-02.gca transforms/talairach.m3z 

#--------------------------------------
#@# SubCort Seg Wed Mar 24 17:57:17 EDT 2021

 mri_ca_label -relabel_unlikely 9 .3 -prior 0.5 -align norm.mgz transforms/talairach.m3z /home/arevell/freesurfer/average/RB_all_2020-01-02.gca aseg.auto_noCCseg.mgz 

#--------------------------------------
#@# CC Seg Wed Mar 24 19:35:02 EDT 2021

 mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz -lta /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/transforms/cc_up.lta freesurfer 

#--------------------------------------
#@# Merge ASeg Wed Mar 24 19:35:39 EDT 2021

 cp aseg.auto.mgz aseg.presurf.mgz 

#--------------------------------------------
#@# Intensity Normalization2 Wed Mar 24 19:35:39 EDT 2021

 mri_normalize -seed 1234 -mprage -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz 

#--------------------------------------------
#@# Mask BFS Wed Mar 24 19:37:53 EDT 2021

 mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz 

#--------------------------------------------
#@# WM Segmentation Wed Mar 24 19:37:54 EDT 2021

 AntsDenoiseImageFs -i brain.mgz -o antsdn.brain.mgz 


 mri_segment -wsizemm 13 -mprage antsdn.brain.mgz wm.seg.mgz 


 mri_edit_wm_with_aseg -keep-in wm.seg.mgz brain.mgz aseg.presurf.mgz wm.asegedit.mgz 


 mri_pretess wm.asegedit.mgz wm norm.mgz wm.mgz 

#--------------------------------------------
#@# Fill Wed Mar 24 19:40:45 EDT 2021

 mri_fill -a ../scripts/ponscc.cut.log -xform transforms/talairach.lta -segmentation aseg.presurf.mgz wm.mgz filled.mgz 

#--------------------------------------------
#@# Tessellate lh Wed Mar 24 19:42:09 EDT 2021

 mri_pretess ../mri/filled.mgz 255 ../mri/norm.mgz ../mri/filled-pretess255.mgz 


 mri_tessellate ../mri/filled-pretess255.mgz 255 ../surf/lh.orig.nofix 


 rm -f ../mri/filled-pretess255.mgz 


 mris_extract_main_component ../surf/lh.orig.nofix ../surf/lh.orig.nofix 

#--------------------------------------------
#@# Tessellate rh Wed Mar 24 19:42:14 EDT 2021

 mri_pretess ../mri/filled.mgz 127 ../mri/norm.mgz ../mri/filled-pretess127.mgz 


 mri_tessellate ../mri/filled-pretess127.mgz 127 ../surf/rh.orig.nofix 


 rm -f ../mri/filled-pretess127.mgz 


 mris_extract_main_component ../surf/rh.orig.nofix ../surf/rh.orig.nofix 

#--------------------------------------------
#@# Smooth1 lh Wed Mar 24 19:42:20 EDT 2021

 mris_smooth -nw -seed 1234 ../surf/lh.orig.nofix ../surf/lh.smoothwm.nofix 

#--------------------------------------------
#@# Smooth1 rh Wed Mar 24 19:42:20 EDT 2021

 mris_smooth -nw -seed 1234 ../surf/rh.orig.nofix ../surf/rh.smoothwm.nofix 

#--------------------------------------------
#@# Inflation1 lh Wed Mar 24 19:42:25 EDT 2021

 mris_inflate -no-save-sulc ../surf/lh.smoothwm.nofix ../surf/lh.inflated.nofix 

#--------------------------------------------
#@# Inflation1 rh Wed Mar 24 19:42:25 EDT 2021

 mris_inflate -no-save-sulc ../surf/rh.smoothwm.nofix ../surf/rh.inflated.nofix 

#--------------------------------------------
#@# QSphere lh Wed Mar 24 19:43:00 EDT 2021

 mris_sphere -q -p 6 -a 128 -seed 1234 ../surf/lh.inflated.nofix ../surf/lh.qsphere.nofix 

#--------------------------------------------
#@# QSphere rh Wed Mar 24 19:43:00 EDT 2021

 mris_sphere -q -p 6 -a 128 -seed 1234 ../surf/rh.inflated.nofix ../surf/rh.qsphere.nofix 

#@# Fix Topology lh Wed Mar 24 19:48:08 EDT 2021

 mris_fix_topology -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh -ga -seed 1234 freesurfer lh 

#@# Fix Topology rh Wed Mar 24 19:48:09 EDT 2021

 mris_fix_topology -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh -ga -seed 1234 freesurfer rh 



#---------------------------------
# New invocation of recon-all Wed Mar 24 20:01:23 EDT 2021 


#---------------------------------
# New invocation of recon-all Wed Mar 24 20:01:33 EDT 2021 
#--------------------------------------------
#@# MotionCor Wed Mar 24 20:01:33 EDT 2021

 cp /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/orig/001.mgz /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/rawavg.mgz 


 mri_convert /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/rawavg.mgz /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/orig.mgz --conform 


 mri_add_xform_to_header -c /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/transforms/talairach.xfm /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/orig.mgz /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/orig.mgz 

#--------------------------------------------
#@# Talairach Wed Mar 24 20:01:39 EDT 2021

 mri_nu_correct.mni --no-rescale --i orig.mgz --o orig_nu.mgz --ants-n4 --n 1 --proto-iters 1000 --distance 50 



#---------------------------------
# New invocation of recon-all Wed Mar 24 23:01:37 EDT 2021 


#---------------------------------
# New invocation of recon-all Wed Mar 24 23:01:45 EDT 2021 
#--------------------------------------------
#@# MotionCor Wed Mar 24 23:01:45 EDT 2021

 cp /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/orig/001.mgz /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/rawavg.mgz 


 mri_convert /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/rawavg.mgz /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/orig.mgz --conform 


 mri_add_xform_to_header -c /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/transforms/talairach.xfm /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/orig.mgz /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/orig.mgz 

#--------------------------------------------
#@# Talairach Wed Mar 24 23:01:52 EDT 2021

 mri_nu_correct.mni --no-rescale --i orig.mgz --o orig_nu.mgz --ants-n4 --n 1 --proto-iters 1000 --distance 50 


 talairach_avi --i orig_nu.mgz --xfm transforms/talairach.auto.xfm 

talairach_avi log file is transforms/talairach_avi.log...

INFO: transforms/talairach.xfm already exists!
The new transforms/talairach.auto.xfm will not be copied to transforms/talairach.xfm
This is done to retain any edits made to transforms/talairach.xfm
Add the -clean-tal flag to recon-all to overwrite transforms/talairach.xfm

#--------------------------------------------
#@# Talairach Failure Detection Wed Mar 24 23:05:14 EDT 2021

 talairach_afd -T 0.005 -xfm transforms/talairach.xfm 


 awk -f /home/arevell/freesurfer/bin/extract_talairach_avi_QA.awk /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/transforms/talairach_avi.log 


 tal_QC_AZS /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/transforms/talairach_avi.log 

#--------------------------------------------
#@# Nu Intensity Correction Wed Mar 24 23:05:14 EDT 2021

 mri_nu_correct.mni --i orig.mgz --o nu.mgz --uchar transforms/talairach.xfm --proto-iters 1000 --distance 50 --n 1 --ants-n4 


 mri_add_xform_to_header -c /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/transforms/talairach.xfm nu.mgz nu.mgz 

#--------------------------------------------
#@# Intensity Normalization Wed Mar 24 23:08:41 EDT 2021

 mri_normalize -g 1 -seed 1234 -mprage nu.mgz T1.mgz 

#--------------------------------------------
#@# Skull Stripping Wed Mar 24 23:10:00 EDT 2021

 mri_watershed -keep brainmask.auto.mgz brainmask.mgz brainmask.mgz -T1 -brain_atlas /home/arevell/freesurfer/average/RB_all_withskull_2020_01_02.gca transforms/talairach_with_skull.lta T1.mgz brainmask.auto.mgz 


INFO: brainmask.mgz already exists!
The new brainmask.auto.mgz will not be copied to brainmask.mgz.
This is done to retain any edits made to brainmask.mgz.
Add the -clean-bm flag to recon-all to overwrite brainmask.mgz.

#-------------------------------------
#@# EM Registration Wed Mar 24 23:10:23 EDT 2021

 mri_em_register -uns 3 -mask brainmask.mgz nu.mgz /home/arevell/freesurfer/average/RB_all_2020-01-02.gca transforms/talairach.lta 

#--------------------------------------
#@# CA Normalize Wed Mar 24 23:20:44 EDT 2021

 mri_ca_normalize -c ctrl_pts.mgz -mask brainmask.mgz nu.mgz /home/arevell/freesurfer/average/RB_all_2020-01-02.gca transforms/talairach.lta norm.mgz 

#--------------------------------------
#@# CA Reg Wed Mar 24 23:21:47 EDT 2021

 mri_ca_register -nobigventricles -T transforms/talairach.lta -align-after -mask brainmask.mgz norm.mgz /home/arevell/freesurfer/average/RB_all_2020-01-02.gca transforms/talairach.m3z 

#--------------------------------------
#@# SubCort Seg Thu Mar 25 00:16:06 EDT 2021

 mri_seg_diff --seg1 aseg.auto.mgz --seg2 aseg.presurf.mgz --diff aseg.manedit.mgz 


 mri_ca_label -relabel_unlikely 9 .3 -prior 0.5 -align norm.mgz transforms/talairach.m3z /home/arevell/freesurfer/average/RB_all_2020-01-02.gca aseg.auto_noCCseg.mgz 

#--------------------------------------
#@# CC Seg Thu Mar 25 00:53:23 EDT 2021

 mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz -lta /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/transforms/cc_up.lta freesurfer 

#--------------------------------------
#@# Merge ASeg Thu Mar 25 00:53:50 EDT 2021

 cp aseg.auto.mgz aseg.presurf.mgz 

#--------------------------------------------
#@# Intensity Normalization2 Thu Mar 25 00:53:50 EDT 2021

 mri_normalize -seed 1234 -mprage -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz 

#--------------------------------------------
#@# Mask BFS Thu Mar 25 00:55:56 EDT 2021

 mri_mask -T 5 brain.mgz brainmask.mgz brain.finalsurfs.mgz 

#--------------------------------------------
#@# WM Segmentation Thu Mar 25 00:55:57 EDT 2021

 mri_binarize --i wm.mgz --min 255 --max 255 --o wm255.mgz --count wm255.txt 


 mri_binarize --i wm.mgz --min 1 --max 1 --o wm1.mgz --count wm1.txt 


 rm wm1.mgz wm255.mgz 


 cp wm.mgz wm.seg.mgz 


 AntsDenoiseImageFs -i brain.mgz -o antsdn.brain.mgz 


 mri_segment -wsizemm 13 -keep -mprage antsdn.brain.mgz wm.seg.mgz 


 mri_edit_wm_with_aseg -keep-in wm.seg.mgz brain.mgz aseg.presurf.mgz wm.asegedit.mgz 


 mri_pretess -keep wm.asegedit.mgz wm norm.mgz wm.mgz 

#--------------------------------------------
#@# Fill Thu Mar 25 00:58:30 EDT 2021

 mri_fill -a ../scripts/ponscc.cut.log -xform transforms/talairach.lta -segmentation aseg.presurf.mgz wm.mgz filled.mgz 

#--------------------------------------------
#@# Tessellate lh Thu Mar 25 01:00:03 EDT 2021

 mri_pretess ../mri/filled.mgz 255 ../mri/norm.mgz ../mri/filled-pretess255.mgz 


 mri_tessellate ../mri/filled-pretess255.mgz 255 ../surf/lh.orig.nofix 


 rm -f ../mri/filled-pretess255.mgz 


 mris_extract_main_component ../surf/lh.orig.nofix ../surf/lh.orig.nofix 

#--------------------------------------------
#@# Tessellate rh Thu Mar 25 01:00:07 EDT 2021

 mri_pretess ../mri/filled.mgz 127 ../mri/norm.mgz ../mri/filled-pretess127.mgz 


 mri_tessellate ../mri/filled-pretess127.mgz 127 ../surf/rh.orig.nofix 


 rm -f ../mri/filled-pretess127.mgz 


 mris_extract_main_component ../surf/rh.orig.nofix ../surf/rh.orig.nofix 

#--------------------------------------------
#@# Smooth1 lh Thu Mar 25 01:00:11 EDT 2021

 mris_smooth -nw -seed 1234 ../surf/lh.orig.nofix ../surf/lh.smoothwm.nofix 

#--------------------------------------------
#@# Smooth1 rh Thu Mar 25 01:00:11 EDT 2021

 mris_smooth -nw -seed 1234 ../surf/rh.orig.nofix ../surf/rh.smoothwm.nofix 

#--------------------------------------------
#@# Inflation1 lh Thu Mar 25 01:00:15 EDT 2021

 mris_inflate -no-save-sulc ../surf/lh.smoothwm.nofix ../surf/lh.inflated.nofix 

#--------------------------------------------
#@# Inflation1 rh Thu Mar 25 01:00:15 EDT 2021

 mris_inflate -no-save-sulc ../surf/rh.smoothwm.nofix ../surf/rh.inflated.nofix 

#--------------------------------------------
#@# QSphere lh Thu Mar 25 01:00:38 EDT 2021

 mris_sphere -q -p 6 -a 128 -seed 1234 ../surf/lh.inflated.nofix ../surf/lh.qsphere.nofix 

#--------------------------------------------
#@# QSphere rh Thu Mar 25 01:00:38 EDT 2021

 mris_sphere -q -p 6 -a 128 -seed 1234 ../surf/rh.inflated.nofix ../surf/rh.qsphere.nofix 

#@# Fix Topology lh Thu Mar 25 01:03:08 EDT 2021

 mris_fix_topology -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh -ga -seed 1234 freesurfer lh 

#@# Fix Topology rh Thu Mar 25 01:03:08 EDT 2021

 mris_fix_topology -mgz -sphere qsphere.nofix -inflated inflated.nofix -orig orig.nofix -out orig.premesh -ga -seed 1234 freesurfer rh 


 mris_euler_number ../surf/lh.orig.premesh 


 mris_euler_number ../surf/rh.orig.premesh 


 mris_remesh --remesh --iters 3 --input /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/lh.orig.premesh --output /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/lh.orig 


 mris_remesh --remesh --iters 3 --input /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/rh.orig.premesh --output /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/rh.orig 


 mris_remove_intersection ../surf/lh.orig ../surf/lh.orig 


 rm ../surf/lh.inflated 


 mris_remove_intersection ../surf/rh.orig ../surf/rh.orig 


 rm ../surf/rh.inflated 

#--------------------------------------------
#@# AutoDetGWStats lh Thu Mar 25 01:08:36 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_autodet_gwstats --o ../surf/autodet.gw.stats.lh.dat --i brain.finalsurfs.mgz --wm wm.mgz --surf ../surf/lh.orig.premesh
#--------------------------------------------
#@# AutoDetGWStats rh Thu Mar 25 01:08:40 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_autodet_gwstats --o ../surf/autodet.gw.stats.rh.dat --i brain.finalsurfs.mgz --wm wm.mgz --surf ../surf/rh.orig.premesh
#--------------------------------------------
#@# WhitePreAparc lh Thu Mar 25 01:08:44 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --wm wm.mgz --threads 4 --invol brain.finalsurfs.mgz --lh --i ../surf/lh.orig --o ../surf/lh.white.preaparc --white --seg aseg.presurf.mgz --nsmooth 5
#--------------------------------------------
#@# WhitePreAparc rh Thu Mar 25 01:12:38 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.rh.dat --wm wm.mgz --threads 4 --invol brain.finalsurfs.mgz --rh --i ../surf/rh.orig --o ../surf/rh.white.preaparc --white --seg aseg.presurf.mgz --nsmooth 5
#--------------------------------------------
#@# CortexLabel lh Thu Mar 25 01:18:57 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mri_label2label --label-cortex ../surf/lh.white.preaparc aseg.presurf.mgz 0 ../label/lh.cortex.label
#--------------------------------------------
#@# CortexLabel+HipAmyg lh Thu Mar 25 01:19:21 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mri_label2label --label-cortex ../surf/lh.white.preaparc aseg.presurf.mgz 1 ../label/lh.cortex+hipamyg.label
#--------------------------------------------
#@# CortexLabel rh Thu Mar 25 01:19:44 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mri_label2label --label-cortex ../surf/rh.white.preaparc aseg.presurf.mgz 0 ../label/rh.cortex.label
#--------------------------------------------
#@# CortexLabel+HipAmyg rh Thu Mar 25 01:20:09 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mri_label2label --label-cortex ../surf/rh.white.preaparc aseg.presurf.mgz 1 ../label/rh.cortex+hipamyg.label
#--------------------------------------------
#@# Smooth2 lh Thu Mar 25 01:20:34 EDT 2021

 mris_smooth -n 3 -nw -seed 1234 ../surf/lh.white.preaparc ../surf/lh.smoothwm 

#--------------------------------------------
#@# Smooth2 rh Thu Mar 25 01:20:34 EDT 2021

 mris_smooth -n 3 -nw -seed 1234 ../surf/rh.white.preaparc ../surf/rh.smoothwm 

#--------------------------------------------
#@# Inflation2 lh Thu Mar 25 01:20:40 EDT 2021

 mris_inflate ../surf/lh.smoothwm ../surf/lh.inflated 

#--------------------------------------------
#@# Inflation2 rh Thu Mar 25 01:20:40 EDT 2021

 mris_inflate ../surf/rh.smoothwm ../surf/rh.inflated 

#--------------------------------------------
#@# Curv .H and .K lh Thu Mar 25 01:21:15 EDT 2021

 mris_curvature -w -seed 1234 lh.white.preaparc 


 mris_curvature -seed 1234 -thresh .999 -n -a 5 -w -distances 10 10 lh.inflated 

#--------------------------------------------
#@# Curv .H and .K rh Thu Mar 25 01:21:15 EDT 2021

 mris_curvature -w -seed 1234 rh.white.preaparc 


 mris_curvature -seed 1234 -thresh .999 -n -a 5 -w -distances 10 10 rh.inflated 

#--------------------------------------------
#@# Sphere lh Thu Mar 25 01:22:10 EDT 2021

 mris_sphere -seed 1234 ../surf/lh.inflated ../surf/lh.sphere 

#--------------------------------------------
#@# Sphere rh Thu Mar 25 01:22:10 EDT 2021

 mris_sphere -seed 1234 ../surf/rh.inflated ../surf/rh.sphere 

#--------------------------------------------
#@# Surf Reg lh Thu Mar 25 01:39:26 EDT 2021

 mris_register -curv ../surf/lh.sphere /home/arevell/freesurfer/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif ../surf/lh.sphere.reg 

#--------------------------------------------
#@# Surf Reg rh Thu Mar 25 01:39:26 EDT 2021

 mris_register -curv ../surf/rh.sphere /home/arevell/freesurfer/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif ../surf/rh.sphere.reg 

#--------------------------------------------
#@# Jacobian white lh Thu Mar 25 01:54:17 EDT 2021

 mris_jacobian ../surf/lh.white.preaparc ../surf/lh.sphere.reg ../surf/lh.jacobian_white 

#--------------------------------------------
#@# Jacobian white rh Thu Mar 25 01:54:17 EDT 2021

 mris_jacobian ../surf/rh.white.preaparc ../surf/rh.sphere.reg ../surf/rh.jacobian_white 

#--------------------------------------------
#@# AvgCurv lh Thu Mar 25 01:54:19 EDT 2021

 mrisp_paint -a 5 /home/arevell/freesurfer/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 ../surf/lh.sphere.reg ../surf/lh.avg_curv 

#--------------------------------------------
#@# AvgCurv rh Thu Mar 25 01:54:19 EDT 2021

 mrisp_paint -a 5 /home/arevell/freesurfer/average/rh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif#6 ../surf/rh.sphere.reg ../surf/rh.avg_curv 

#-----------------------------------------
#@# Cortical Parc lh Thu Mar 25 01:54:20 EDT 2021

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 freesurfer lh ../surf/lh.sphere.reg /home/arevell/freesurfer/average/lh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.annot 

#-----------------------------------------
#@# Cortical Parc rh Thu Mar 25 01:54:20 EDT 2021

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 freesurfer rh ../surf/rh.sphere.reg /home/arevell/freesurfer/average/rh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.annot 

#--------------------------------------------
#@# WhiteSurfs lh Thu Mar 25 01:54:35 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --seg aseg.presurf.mgz --threads 4 --wm wm.mgz --invol brain.finalsurfs.mgz --lh --i ../surf/lh.white.preaparc --o ../surf/lh.white --white --nsmooth 0 --rip-label ../label/lh.cortex.label --rip-bg --rip-surf ../surf/lh.white.preaparc --aparc ../label/lh.aparc.annot
#--------------------------------------------
#@# WhiteSurfs rh Thu Mar 25 01:58:50 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.rh.dat --seg aseg.presurf.mgz --threads 4 --wm wm.mgz --invol brain.finalsurfs.mgz --rh --i ../surf/rh.white.preaparc --o ../surf/rh.white --white --nsmooth 0 --rip-label ../label/rh.cortex.label --rip-bg --rip-surf ../surf/rh.white.preaparc --aparc ../label/rh.aparc.annot
#--------------------------------------------
#@# T1PialSurf lh Thu Mar 25 02:03:17 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --seg aseg.presurf.mgz --threads 4 --wm wm.mgz --invol brain.finalsurfs.mgz --lh --i ../surf/lh.white --o ../surf/lh.pial.T1 --pial --nsmooth 0 --rip-label ../label/lh.cortex+hipamyg.label --pin-medial-wall ../label/lh.cortex.label --aparc ../label/lh.aparc.annot --repulse-surf ../surf/lh.white --white-surf ../surf/lh.white
#--------------------------------------------
#@# T1PialSurf rh Thu Mar 25 02:06:11 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --adgws-in ../surf/autodet.gw.stats.rh.dat --seg aseg.presurf.mgz --threads 4 --wm wm.mgz --invol brain.finalsurfs.mgz --rh --i ../surf/rh.white --o ../surf/rh.pial.T1 --pial --nsmooth 0 --rip-label ../label/rh.cortex+hipamyg.label --pin-medial-wall ../label/rh.cortex.label --aparc ../label/rh.aparc.annot --repulse-surf ../surf/rh.white --white-surf ../surf/rh.white
#@# white curv lh Thu Mar 25 02:09:07 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --curv-map ../surf/lh.white 2 10 ../surf/lh.curv
#@# white area lh Thu Mar 25 02:09:09 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --area-map ../surf/lh.white ../surf/lh.area
#@# pial curv lh Thu Mar 25 02:09:10 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --curv-map ../surf/lh.pial 2 10 ../surf/lh.curv.pial
#@# pial area lh Thu Mar 25 02:09:13 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --area-map ../surf/lh.pial ../surf/lh.area.pial
#@# thickness lh Thu Mar 25 02:09:14 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --thickness ../surf/lh.white ../surf/lh.pial 20 5 ../surf/lh.thickness
#@# area and vertex vol lh Thu Mar 25 02:10:02 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --thickness ../surf/lh.white ../surf/lh.pial 20 5 ../surf/lh.thickness
#@# white curv rh Thu Mar 25 02:10:04 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --curv-map ../surf/rh.white 2 10 ../surf/rh.curv
#@# white area rh Thu Mar 25 02:10:06 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --area-map ../surf/rh.white ../surf/rh.area
#@# pial curv rh Thu Mar 25 02:10:08 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --curv-map ../surf/rh.pial 2 10 ../surf/rh.curv.pial
#@# pial area rh Thu Mar 25 02:10:10 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --area-map ../surf/rh.pial ../surf/rh.area.pial
#@# thickness rh Thu Mar 25 02:10:11 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --thickness ../surf/rh.white ../surf/rh.pial 20 5 ../surf/rh.thickness
#@# area and vertex vol rh Thu Mar 25 02:10:54 EDT 2021
cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri
mris_place_surface --thickness ../surf/rh.white ../surf/rh.pial 20 5 ../surf/rh.thickness

#-----------------------------------------
#@# Curvature Stats lh Thu Mar 25 02:10:56 EDT 2021

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/lh.curv.stats -F smoothwm freesurfer lh curv sulc 


#-----------------------------------------
#@# Curvature Stats rh Thu Mar 25 02:11:00 EDT 2021

 mris_curvature_stats -m --writeCurvatureFiles -G -o ../stats/rh.curv.stats -F smoothwm freesurfer rh curv sulc 

#--------------------------------------------
#@# Cortical ribbon mask Thu Mar 25 02:11:04 EDT 2021

 mris_volmask --aseg_name aseg.presurf --label_left_white 2 --label_left_ribbon 3 --label_right_white 41 --label_right_ribbon 42 --save_ribbon --parallel freesurfer 

#-----------------------------------------
#@# Cortical Parc 2 lh Thu Mar 25 02:15:52 EDT 2021

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 freesurfer lh ../surf/lh.sphere.reg /home/arevell/freesurfer/average/lh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.a2009s.annot 

#-----------------------------------------
#@# Cortical Parc 2 rh Thu Mar 25 02:15:52 EDT 2021

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 freesurfer rh ../surf/rh.sphere.reg /home/arevell/freesurfer/average/rh.CDaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.a2009s.annot 

#-----------------------------------------
#@# Cortical Parc 3 lh Thu Mar 25 02:16:05 EDT 2021

 mris_ca_label -l ../label/lh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 freesurfer lh ../surf/lh.sphere.reg /home/arevell/freesurfer/average/lh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/lh.aparc.DKTatlas.annot 

#-----------------------------------------
#@# Cortical Parc 3 rh Thu Mar 25 02:16:05 EDT 2021

 mris_ca_label -l ../label/rh.cortex.label -aseg ../mri/aseg.presurf.mgz -seed 1234 freesurfer rh ../surf/rh.sphere.reg /home/arevell/freesurfer/average/rh.DKTaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs ../label/rh.aparc.DKTatlas.annot 

#-----------------------------------------
#@# WM/GM Contrast lh Thu Mar 25 02:16:18 EDT 2021

 pctsurfcon --s freesurfer --lh-only 

#-----------------------------------------
#@# WM/GM Contrast rh Thu Mar 25 02:16:18 EDT 2021

 pctsurfcon --s freesurfer --rh-only 

#-----------------------------------------
#@# Relabel Hypointensities Thu Mar 25 02:16:21 EDT 2021

 mri_relabel_hypointensities aseg.presurf.mgz ../surf aseg.presurf.hypos.mgz 

#-----------------------------------------
#@# APas-to-ASeg Thu Mar 25 02:16:37 EDT 2021

 mri_surf2volseg --o aseg.mgz --i aseg.presurf.hypos.mgz --fix-presurf-with-ribbon /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/mri/ribbon.mgz --threads 4 --lh-cortex-mask /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/lh.cortex.label --lh-white /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/lh.white --lh-pial /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/lh.pial --rh-cortex-mask /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/rh.cortex.label --rh-white /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/rh.white --rh-pial /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/rh.pial 


 mri_brainvol_stats freesurfer 

#-----------------------------------------
#@# AParc-to-ASeg aparc Thu Mar 25 02:16:47 EDT 2021

 mri_surf2volseg --o aparc+aseg.mgz --label-cortex --i aseg.mgz --threads 4 --lh-annot /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/lh.aparc.annot 1000 --lh-cortex-mask /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/lh.cortex.label --lh-white /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/lh.white --lh-pial /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/lh.pial --rh-annot /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/rh.aparc.annot 2000 --rh-cortex-mask /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/rh.cortex.label --rh-white /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/rh.white --rh-pial /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/rh.pial 

#-----------------------------------------
#@# AParc-to-ASeg aparc.a2009s Thu Mar 25 02:18:14 EDT 2021

 mri_surf2volseg --o aparc.a2009s+aseg.mgz --label-cortex --i aseg.mgz --threads 4 --lh-annot /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/lh.aparc.a2009s.annot 11100 --lh-cortex-mask /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/lh.cortex.label --lh-white /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/lh.white --lh-pial /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/lh.pial --rh-annot /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/rh.aparc.a2009s.annot 12100 --rh-cortex-mask /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/rh.cortex.label --rh-white /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/rh.white --rh-pial /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/rh.pial 

#-----------------------------------------
#@# AParc-to-ASeg aparc.DKTatlas Thu Mar 25 02:19:33 EDT 2021

 mri_surf2volseg --o aparc.DKTatlas+aseg.mgz --label-cortex --i aseg.mgz --threads 4 --lh-annot /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/lh.aparc.DKTatlas.annot 1000 --lh-cortex-mask /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/lh.cortex.label --lh-white /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/lh.white --lh-pial /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/lh.pial --rh-annot /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/rh.aparc.DKTatlas.annot 2000 --rh-cortex-mask /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/rh.cortex.label --rh-white /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/rh.white --rh-pial /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/rh.pial 

#-----------------------------------------
#@# WMParc Thu Mar 25 02:20:57 EDT 2021

 mri_surf2volseg --o wmparc.mgz --label-wm --i aparc+aseg.mgz --threads 4 --lh-annot /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/lh.aparc.annot 3000 --lh-cortex-mask /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/lh.cortex.label --lh-white /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/lh.white --lh-pial /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/lh.pial --rh-annot /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/rh.aparc.annot 4000 --rh-cortex-mask /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/label/rh.cortex.label --rh-white /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/rh.white --rh-pial /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/freesurfer/surf/rh.pial 


 mri_segstats --seed 1234 --seg mri/wmparc.mgz --sum stats/wmparc.stats --pv mri/norm.mgz --excludeid 0 --brainmask mri/brainmask.mgz --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --subject freesurfer --surf-wm-vol --ctab /home/arevell/freesurfer/WMParcStatsLUT.txt --etiv 

#-----------------------------------------
#@# Parcellation Stats lh Thu Mar 25 02:24:53 EDT 2021

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab freesurfer lh white 


 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.pial.stats -b -a ../label/lh.aparc.annot -c ../label/aparc.annot.ctab freesurfer lh pial 

#-----------------------------------------
#@# Parcellation Stats rh Thu Mar 25 02:24:53 EDT 2021

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab freesurfer rh white 


 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.pial.stats -b -a ../label/rh.aparc.annot -c ../label/aparc.annot.ctab freesurfer rh pial 

#-----------------------------------------
#@# Parcellation Stats 2 lh Thu Mar 25 02:25:06 EDT 2021

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.a2009s.stats -b -a ../label/lh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab freesurfer lh white 

#-----------------------------------------
#@# Parcellation Stats 2 rh Thu Mar 25 02:25:06 EDT 2021

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.a2009s.stats -b -a ../label/rh.aparc.a2009s.annot -c ../label/aparc.annot.a2009s.ctab freesurfer rh white 

#-----------------------------------------
#@# Parcellation Stats 3 lh Thu Mar 25 02:25:18 EDT 2021

 mris_anatomical_stats -th3 -mgz -cortex ../label/lh.cortex.label -f ../stats/lh.aparc.DKTatlas.stats -b -a ../label/lh.aparc.DKTatlas.annot -c ../label/aparc.annot.DKTatlas.ctab freesurfer lh white 

#-----------------------------------------
#@# Parcellation Stats 3 rh Thu Mar 25 02:25:18 EDT 2021

 mris_anatomical_stats -th3 -mgz -cortex ../label/rh.cortex.label -f ../stats/rh.aparc.DKTatlas.stats -b -a ../label/rh.aparc.DKTatlas.annot -c ../label/aparc.annot.DKTatlas.ctab freesurfer rh white 

#--------------------------------------------
#@# ASeg Stats Thu Mar 25 02:25:28 EDT 2021

 mri_segstats --seed 1234 --seg mri/aseg.mgz --sum stats/aseg.stats --pv mri/norm.mgz --empty --brainmask mri/brainmask.mgz --brain-vol-from-seg --excludeid 0 --excl-ctxgmwm --supratent --subcortgray --in mri/norm.mgz --in-intensity-name norm --in-intensity-units MR --etiv --surf-wm-vol --surf-ctx-vol --totalgray --euler --ctab /home/arevell/freesurfer/ASegStatsLUT.txt --subject freesurfer 

INFO: fsaverage subject does not exist in SUBJECTS_DIR
INFO: Creating symlink to fsaverage subject...

 cd /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate; ln -s /home/arevell/freesurfer/subjects/fsaverage; cd - 

#--------------------------------------------
#@# BA_exvivo Labels lh Thu Mar 25 02:26:35 EDT 2021

 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA1_exvivo.label --trgsubject freesurfer --trglabel ./lh.BA1_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA2_exvivo.label --trgsubject freesurfer --trglabel ./lh.BA2_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA3a_exvivo.label --trgsubject freesurfer --trglabel ./lh.BA3a_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA3b_exvivo.label --trgsubject freesurfer --trglabel ./lh.BA3b_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA4a_exvivo.label --trgsubject freesurfer --trglabel ./lh.BA4a_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA4p_exvivo.label --trgsubject freesurfer --trglabel ./lh.BA4p_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA6_exvivo.label --trgsubject freesurfer --trglabel ./lh.BA6_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA44_exvivo.label --trgsubject freesurfer --trglabel ./lh.BA44_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA45_exvivo.label --trgsubject freesurfer --trglabel ./lh.BA45_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.V1_exvivo.label --trgsubject freesurfer --trglabel ./lh.V1_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.V2_exvivo.label --trgsubject freesurfer --trglabel ./lh.V2_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.MT_exvivo.label --trgsubject freesurfer --trglabel ./lh.MT_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.entorhinal_exvivo.label --trgsubject freesurfer --trglabel ./lh.entorhinal_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.perirhinal_exvivo.label --trgsubject freesurfer --trglabel ./lh.perirhinal_exvivo.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.FG1.mpm.vpnl.label --trgsubject freesurfer --trglabel ./lh.FG1.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.FG2.mpm.vpnl.label --trgsubject freesurfer --trglabel ./lh.FG2.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.FG3.mpm.vpnl.label --trgsubject freesurfer --trglabel ./lh.FG3.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.FG4.mpm.vpnl.label --trgsubject freesurfer --trglabel ./lh.FG4.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.hOc1.mpm.vpnl.label --trgsubject freesurfer --trglabel ./lh.hOc1.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.hOc2.mpm.vpnl.label --trgsubject freesurfer --trglabel ./lh.hOc2.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.hOc3v.mpm.vpnl.label --trgsubject freesurfer --trglabel ./lh.hOc3v.mpm.vpnl.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.hOc4v.mpm.vpnl.label --trgsubject freesurfer --trglabel ./lh.hOc4v.mpm.vpnl.label --hemi lh --regmethod surface 


 mris_label2annot --s freesurfer --ctab /home/arevell/freesurfer/average/colortable_vpnl.txt --hemi lh --a mpm.vpnl --maxstatwinner --noverbose --l lh.FG1.mpm.vpnl.label --l lh.FG2.mpm.vpnl.label --l lh.FG3.mpm.vpnl.label --l lh.FG4.mpm.vpnl.label --l lh.hOc1.mpm.vpnl.label --l lh.hOc2.mpm.vpnl.label --l lh.hOc3v.mpm.vpnl.label --l lh.hOc4v.mpm.vpnl.label 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA1_exvivo.thresh.label --trgsubject freesurfer --trglabel ./lh.BA1_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA2_exvivo.thresh.label --trgsubject freesurfer --trglabel ./lh.BA2_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA3a_exvivo.thresh.label --trgsubject freesurfer --trglabel ./lh.BA3a_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA3b_exvivo.thresh.label --trgsubject freesurfer --trglabel ./lh.BA3b_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA4a_exvivo.thresh.label --trgsubject freesurfer --trglabel ./lh.BA4a_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA4p_exvivo.thresh.label --trgsubject freesurfer --trglabel ./lh.BA4p_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA6_exvivo.thresh.label --trgsubject freesurfer --trglabel ./lh.BA6_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA44_exvivo.thresh.label --trgsubject freesurfer --trglabel ./lh.BA44_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.BA45_exvivo.thresh.label --trgsubject freesurfer --trglabel ./lh.BA45_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.V1_exvivo.thresh.label --trgsubject freesurfer --trglabel ./lh.V1_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.V2_exvivo.thresh.label --trgsubject freesurfer --trglabel ./lh.V2_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.MT_exvivo.thresh.label --trgsubject freesurfer --trglabel ./lh.MT_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.entorhinal_exvivo.thresh.label --trgsubject freesurfer --trglabel ./lh.entorhinal_exvivo.thresh.label --hemi lh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/lh.perirhinal_exvivo.thresh.label --trgsubject freesurfer --trglabel ./lh.perirhinal_exvivo.thresh.label --hemi lh --regmethod surface 


 mris_label2annot --s freesurfer --hemi lh --ctab /home/arevell/freesurfer/average/colortable_BA.txt --l lh.BA1_exvivo.label --l lh.BA2_exvivo.label --l lh.BA3a_exvivo.label --l lh.BA3b_exvivo.label --l lh.BA4a_exvivo.label --l lh.BA4p_exvivo.label --l lh.BA6_exvivo.label --l lh.BA44_exvivo.label --l lh.BA45_exvivo.label --l lh.V1_exvivo.label --l lh.V2_exvivo.label --l lh.MT_exvivo.label --l lh.perirhinal_exvivo.label --l lh.entorhinal_exvivo.label --a BA_exvivo --maxstatwinner --noverbose 


 mris_label2annot --s freesurfer --hemi lh --ctab /home/arevell/freesurfer/average/colortable_BA.txt --l lh.BA1_exvivo.thresh.label --l lh.BA2_exvivo.thresh.label --l lh.BA3a_exvivo.thresh.label --l lh.BA3b_exvivo.thresh.label --l lh.BA4a_exvivo.thresh.label --l lh.BA4p_exvivo.thresh.label --l lh.BA6_exvivo.thresh.label --l lh.BA44_exvivo.thresh.label --l lh.BA45_exvivo.thresh.label --l lh.V1_exvivo.thresh.label --l lh.V2_exvivo.thresh.label --l lh.MT_exvivo.thresh.label --l lh.perirhinal_exvivo.thresh.label --l lh.entorhinal_exvivo.thresh.label --a BA_exvivo.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -th3 -mgz -f ../stats/lh.BA_exvivo.stats -b -a ./lh.BA_exvivo.annot -c ./BA_exvivo.ctab freesurfer lh white 


 mris_anatomical_stats -th3 -mgz -f ../stats/lh.BA_exvivo.thresh.stats -b -a ./lh.BA_exvivo.thresh.annot -c ./BA_exvivo.thresh.ctab freesurfer lh white 

#--------------------------------------------
#@# BA_exvivo Labels rh Thu Mar 25 02:27:38 EDT 2021

 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA1_exvivo.label --trgsubject freesurfer --trglabel ./rh.BA1_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA2_exvivo.label --trgsubject freesurfer --trglabel ./rh.BA2_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA3a_exvivo.label --trgsubject freesurfer --trglabel ./rh.BA3a_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA3b_exvivo.label --trgsubject freesurfer --trglabel ./rh.BA3b_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA4a_exvivo.label --trgsubject freesurfer --trglabel ./rh.BA4a_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA4p_exvivo.label --trgsubject freesurfer --trglabel ./rh.BA4p_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA6_exvivo.label --trgsubject freesurfer --trglabel ./rh.BA6_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA44_exvivo.label --trgsubject freesurfer --trglabel ./rh.BA44_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA45_exvivo.label --trgsubject freesurfer --trglabel ./rh.BA45_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.V1_exvivo.label --trgsubject freesurfer --trglabel ./rh.V1_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.V2_exvivo.label --trgsubject freesurfer --trglabel ./rh.V2_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.MT_exvivo.label --trgsubject freesurfer --trglabel ./rh.MT_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.entorhinal_exvivo.label --trgsubject freesurfer --trglabel ./rh.entorhinal_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.perirhinal_exvivo.label --trgsubject freesurfer --trglabel ./rh.perirhinal_exvivo.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.FG1.mpm.vpnl.label --trgsubject freesurfer --trglabel ./rh.FG1.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.FG2.mpm.vpnl.label --trgsubject freesurfer --trglabel ./rh.FG2.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.FG3.mpm.vpnl.label --trgsubject freesurfer --trglabel ./rh.FG3.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.FG4.mpm.vpnl.label --trgsubject freesurfer --trglabel ./rh.FG4.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.hOc1.mpm.vpnl.label --trgsubject freesurfer --trglabel ./rh.hOc1.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.hOc2.mpm.vpnl.label --trgsubject freesurfer --trglabel ./rh.hOc2.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.hOc3v.mpm.vpnl.label --trgsubject freesurfer --trglabel ./rh.hOc3v.mpm.vpnl.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.hOc4v.mpm.vpnl.label --trgsubject freesurfer --trglabel ./rh.hOc4v.mpm.vpnl.label --hemi rh --regmethod surface 


 mris_label2annot --s freesurfer --ctab /home/arevell/freesurfer/average/colortable_vpnl.txt --hemi rh --a mpm.vpnl --maxstatwinner --noverbose --l rh.FG1.mpm.vpnl.label --l rh.FG2.mpm.vpnl.label --l rh.FG3.mpm.vpnl.label --l rh.FG4.mpm.vpnl.label --l rh.hOc1.mpm.vpnl.label --l rh.hOc2.mpm.vpnl.label --l rh.hOc3v.mpm.vpnl.label --l rh.hOc4v.mpm.vpnl.label 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA1_exvivo.thresh.label --trgsubject freesurfer --trglabel ./rh.BA1_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA2_exvivo.thresh.label --trgsubject freesurfer --trglabel ./rh.BA2_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA3a_exvivo.thresh.label --trgsubject freesurfer --trglabel ./rh.BA3a_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA3b_exvivo.thresh.label --trgsubject freesurfer --trglabel ./rh.BA3b_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA4a_exvivo.thresh.label --trgsubject freesurfer --trglabel ./rh.BA4a_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA4p_exvivo.thresh.label --trgsubject freesurfer --trglabel ./rh.BA4p_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA6_exvivo.thresh.label --trgsubject freesurfer --trglabel ./rh.BA6_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA44_exvivo.thresh.label --trgsubject freesurfer --trglabel ./rh.BA44_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.BA45_exvivo.thresh.label --trgsubject freesurfer --trglabel ./rh.BA45_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.V1_exvivo.thresh.label --trgsubject freesurfer --trglabel ./rh.V1_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.V2_exvivo.thresh.label --trgsubject freesurfer --trglabel ./rh.V2_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.MT_exvivo.thresh.label --trgsubject freesurfer --trglabel ./rh.MT_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.entorhinal_exvivo.thresh.label --trgsubject freesurfer --trglabel ./rh.entorhinal_exvivo.thresh.label --hemi rh --regmethod surface 


 mri_label2label --srcsubject fsaverage --srclabel /media/arevell/sharedSSD/linux/revellLab/tools/mniTemplate/fsaverage/label/rh.perirhinal_exvivo.thresh.label --trgsubject freesurfer --trglabel ./rh.perirhinal_exvivo.thresh.label --hemi rh --regmethod surface 


 mris_label2annot --s freesurfer --hemi rh --ctab /home/arevell/freesurfer/average/colortable_BA.txt --l rh.BA1_exvivo.label --l rh.BA2_exvivo.label --l rh.BA3a_exvivo.label --l rh.BA3b_exvivo.label --l rh.BA4a_exvivo.label --l rh.BA4p_exvivo.label --l rh.BA6_exvivo.label --l rh.BA44_exvivo.label --l rh.BA45_exvivo.label --l rh.V1_exvivo.label --l rh.V2_exvivo.label --l rh.MT_exvivo.label --l rh.perirhinal_exvivo.label --l rh.entorhinal_exvivo.label --a BA_exvivo --maxstatwinner --noverbose 


 mris_label2annot --s freesurfer --hemi rh --ctab /home/arevell/freesurfer/average/colortable_BA.txt --l rh.BA1_exvivo.thresh.label --l rh.BA2_exvivo.thresh.label --l rh.BA3a_exvivo.thresh.label --l rh.BA3b_exvivo.thresh.label --l rh.BA4a_exvivo.thresh.label --l rh.BA4p_exvivo.thresh.label --l rh.BA6_exvivo.thresh.label --l rh.BA44_exvivo.thresh.label --l rh.BA45_exvivo.thresh.label --l rh.V1_exvivo.thresh.label --l rh.V2_exvivo.thresh.label --l rh.MT_exvivo.thresh.label --l rh.perirhinal_exvivo.thresh.label --l rh.entorhinal_exvivo.thresh.label --a BA_exvivo.thresh --maxstatwinner --noverbose 


 mris_anatomical_stats -th3 -mgz -f ../stats/rh.BA_exvivo.stats -b -a ./rh.BA_exvivo.annot -c ./BA_exvivo.ctab freesurfer rh white 


 mris_anatomical_stats -th3 -mgz -f ../stats/rh.BA_exvivo.thresh.stats -b -a ./rh.BA_exvivo.thresh.annot -c ./BA_exvivo.thresh.ctab freesurfer rh white 

