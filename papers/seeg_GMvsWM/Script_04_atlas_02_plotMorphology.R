path = "/media/arevell/sharedSSD/linux/papers/paper005/" #path to where the paper directory is stored - locally or remotely

fpath_morphology = file.path(path, "data/data_processed/atlas_morphology")
fpath_volume = file.path(fpath_morphology, "volume")
fpath_sphericity = file.path(fpath_morphology, "sphericity")
fname_atlases_csv = file.path( path, "data/data_raw/atlases/atlas_names.csv")


setwd(path)
library(ggplot2)
library("cowplot")

pdf(file.path(fpath_morphology, "volumes_and_sphericity.pdf"), width = 10, height = 9)
#png(file.path(fpath_morphology, "volumes_and_sphericity.png"), width = 12, height = 8.5,  units = "in", res = 600)

atlases = read.csv(fname_atlases_csv, header = TRUE, stringsAsFactors = FALSE)

#extracting data into lists
atlas_names = atlases["atlas_name"][,1]


data_volumes = list()
for (i in 1:nrow(atlases)){
  data_volumes[[i]] =  read.csv(file.path(fpath_volume, paste0(atlas_names[i], "_volume.csv")  ), stringsAsFactors = F, header = T)[,2]
  data_volumes[[i]] = data_volumes[[i]][-1] #removing first entry because this is region label = 0, which is outside the brain
  names(data_volumes)[i] = c(atlas_names[i])
  }

data_sphericity = list()
for (i in 1:nrow(atlases)){
  data_sphericity[[i]] =  read.csv(file.path(fpath_sphericity,  paste0(atlas_names[i], "_sphericity.csv")  ), stringsAsFactors = F, header = T)[,2]
  data_sphericity[[i]] = data_sphericity[[i]][-1] #removing first entry because this is region label = 0, which is outside the brain
  names(data_sphericity)[i] = c(atlas_names[i])
  }


standard_atlases_to_plot = c(
  'AAL1',	
  'AAL3', 
  'AAL600',	
  'AAL_JHU_combined',	
  'Craddock_200',	
  'Craddock_400',	
  'Hammersmith' , 
  'MNI_lobar_structural',	
  'DKT31_OASIS', 
  "Yeo_7_liberal", 
  "Yeo_17_liberal",	
  'Schaefer_17_100',	
  'Schaefer_17_1000' )

standard_atlases_legend = rep(NA, length(standard_atlases_to_plot))
for (i in 1:length(standard_atlases_legend)){
  standard_atlases_legend[i] = atlases["atlas_shortname"][which( atlas_names ==  standard_atlases_to_plot[i] ),1]
}
standard_atlases_legend[which(standard_atlases_legend == "Schaefer_17_1000")] = "Schaefer 1000"
standard_atlases_legend[which(standard_atlases_legend == "Schaefer_17_100")] = "Schaefer 100"
random_atlases_to_plot = c(
  'RandomAtlas0000010_v0001', 
  'RandomAtlas0000050_v0003',	
  'RandomAtlas0000100_v0001',	
  'RandomAtlas0000200_v0001',	
  'RandomAtlas0000500_v0001', 
  "RandomAtlas0001000_v0001", 
  "RandomAtlas0002000_v0001",	
  'RandomAtlas0005000_v0001',	
  'RandomAtlas0010000_v0001' )

random_atlases_legend = rep(NA, length(random_atlases_to_plot))
for (i in 1:length(random_atlases_legend)){
  random_atlases_legend[i] = atlases["atlas_shortname"][which( atlas_names ==  random_atlases_to_plot[i] ),1]
}
random_atlases_legend = as.numeric(substring(random_atlases_legend, 12,18) )


all_atlases_to_plot = c(standard_atlases_to_plot , random_atlases_to_plot)

standard_major_atlas_index = match( standard_atlases_to_plot, names(data_volumes))
random_atlas_index = match( random_atlases_to_plot, names(data_volumes))


#Calculating means
mean_volumes = lapply(data_volumes, mean)
mean_sphericity = lapply(data_sphericity, mean)
mean_volumes = lapply(mean_volumes, log10)

#taking log10 of volumes
for (i in 1:length(data_volumes)){
  data_volumes[[i]] = log10(data_volumes[[i]])
}


# #save this mean volumes for figure 4
# volumes_save_data = matrix(NA, nrow = length(names(data_volumes)), ncol = 1 )
# volumes_save_data[,1] = unlist(mean_volumes)
# volumes_save_data = data.frame(volumes_save_data)
#
# sphericity_save_data = matrix(NA, nrow = length(names(data_volumes)), ncol = 1 )
# sphericity_save_data[,1] = unlist(mean_sphericity)
# sphericity_save_data = data.frame(sphericity_save_data)
#
# row.names(volumes_save_data) = names(data_volumes)
# row.names(sphericity_save_data) = names(data_volumes)
# write.table(volumes_save_data,paste0(save_data_directory, "/mean_volumes.csv"), row.names = T, col.names=FALSE, sep = ",")
# write.table(volumes_save_data,paste0(save_data_directory, "/mean_sphericity.csv"), row.names = T, col.names=FALSE, sep = ",")

##########################################################################################################################################
#Standard Major Atlases
#graphing parameters
xlim = c(0,50000)
ylim = c(0,20e-5)




##########################################################################################################################################

index = standard_major_atlas_index
lim_vol = c(0.5,5.5)
density_lim = c(0,20e-0)
ylim = density_lim
lim_sphericity = c(0.0,0.8)














##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################



xlim = lim_vol
start = 1
pch = start:(start+length(data_volumes))
pch[length(standard_atlases_to_plot)] = 15

color_standard = rainbow(length(standard_atlases_to_plot), s = 0.7, v = 0.8, start = 0, end = 0.99, alpha = 0.65)
color_standard[4] = "#333399cc"
random_atlas_colors =  paste0(colorRampPalette(c("#0000ff", "#990099"))(4), "66")
random_atlas_colors =  rainbow(length(random_atlases_to_plot), s = 0.7, v = 0.9, start = 0.00, end = 0.90, alpha = 0.3)
colors = c(color_standard, random_atlas_colors)




par(new = F)
mar =  c(0,0,0,0); mar_1 = 4; mar_2 = 4; mar_3 = 2; mar[2] = mar_2; mar[3] = mar_3; par(mar = mar)
#plotting example volume distribution - Schaefer1000, AAL CPAC, and Glasser:
cutoffs = c(0.8,0.75)
distributions_to_plot = c("AAL_JHU_combined", "Craddock_200" , "Craddock_400", "AAL600", "Schaefer_17_1000") #corresponds to index in standard_atlases_to_plot
distributions_to_plot_index = match( distributions_to_plot, standard_atlases_to_plot)
for (i in 1:length(distributions_to_plot_index)){#VOLUME
  example_indx = which(names(data_volumes) == standard_atlases_to_plot[distributions_to_plot_index[i]])
  example_data = data_volumes[[example_indx]]

  if (i ==1){par(fig=c(0,0.45,0.75,0.92), new=F, xpd = F)}
  if (i > 1){par(new = T)}
  d <- density(example_data )
  plot(d,  xlim = lim_vol, ylim= c(0,5), xlab = "", ylab = "", axes=T, main = "",  bty='n', zero.line = F, las = 1, cex.axis = 0.8)
  polygon(d, col=color_standard[distributions_to_plot_index[i]], border=color_standard[distributions_to_plot_index[i]])

}
legend(x = "bottomleft", legend = standard_atlases_legend[distributions_to_plot_index],
       fill = color_standard[distributions_to_plot_index],border = substr(color_standard[distributions_to_plot_index],1,7),
       bty = "n", cex= 0.9 , xpd = T,
       y.intersp = 1, adj=0)
mtext(expression(paste("volume (log"[10]," mm"^"3", ")")), side = 1, outer = F, line = 2.0 , font = 1, cex = 1)
mtext("density", side = 2, outer = F, line = 2.5 , font = 1, cex = 1)
mtext("ROI Volume Distributions", side = 3, outer = F, line = 0.9 , font = 3, cex = 1.3)
#par(new = T)
#hist(example_data, xlim = c(0,5),breaks=12, xlab = "", ylab = "", axes=T, main = "",  bty='l', las =1)
#title(main = "Volume Distribution",line = -1.7, font.lab = 2, cex.main = 2)
mtext("Atlas Morphology: Sizes and Shapes", side = 3, outer = T, line = -2 , font = 1, cex = 2)

par(xpd=NA)
ltr_x = par("usr")[1]
ltr_y = par("usr")[4]
text(ltr_x - ltr_x*1.16 , ltr_y+ ltr_y*0.3  , labels = "A.", cex = 2, font = 2 )

par(new = T)
#plotting example SPHERICTY distribution -  Schaefer1000, AAL CPAC, and Glasser:
par(fig=c(0.5,0.95,0.75,0.92), new=T)
for (i in 1:length(distributions_to_plot_index)){#SPHERICITY
  example_indx = which(names(data_sphericity) == standard_atlases_to_plot[distributions_to_plot_index[i]])
  example_data = data_sphericity[[example_indx]]
  par(new = T)
  d <- density(example_data )
  plot(d,  xlim = c(0.2,0.8), ylim= c(0,11), xlab = "", ylab = "", axes=T, main = "",  bty='n',  zero.line = F, las = 1, cex.axis = 0.8)
  polygon(d, col=color_standard[distributions_to_plot_index[i]], border=color_standard[distributions_to_plot_index[i]])

}



mtext("sphericity", side = 1, outer = F, line = 2.0 , font = 1, cex = 1)
mtext("density", side = 2, outer = F, line = 2.5 , font = 1, cex = 1)
mtext("ROI Sphericity Distributions", side = 3, outer = F, line = 0.9 , font = 3, cex = 1.3)
#par(new = T)
#hist(example_data, xlim = c(0,0.8),breaks=12, xlab = "", ylab = "", axes=T, main = "",  bty='l', las =1)
#title(main = "Volume Distribution",line = -1.7, font.lab = 2, cex.main = 2)





par(xpd=NA)
ltr_x = par("usr")[1]
ltr_y = par("usr")[4]
text(ltr_x - ltr_x*0.5 , ltr_y+ ltr_y*0.3  , labels = "B.", cex = 2, font = 2 )













mar =  c(0,0,0,0)
mar_1 = 4
mar_2 = 4
mar_3 = 2
mar[2] = mar_2
mar[3] = mar_3
par(mar = mar)

par(xpd=F)

cutoffs = c(0.5,0.78)
par(fig=c(0.00,0.45,0.54,0.78), new=T)
xlim = lim_vol
ylim = density_lim
for (i in index){
  d <- density(data_volumes[[i]] )
  plot(d,  xlim = xlim, ylim= ylim, xlab = "", ylab = "", axes=FALSE, main = "", zero.line = F)
  
  color_index = which(names(data_volumes )[i] == all_atlases_to_plot)
  polygon(d, col=colors[color_index], border=colors[color_index])
  par(new = T)
}

mtext("Standard Atlases:\nVolumes vs Sphericity", side = 3, outer = F, line = -5.75 , font = 3, cex = 1.3)

#title(main = title,line = -1.7, font.lab = 2, cex.main = 2)

par(fig=c(0.45,0.5,0,0.54), new=T)
mar[1] = mar_1
mar[2] = 0
mar[3] = 0
par(mar = mar)
xlim = lim_sphericity
ylim = c(0,30)
for (i in index){

  par(xpd = F)
  d <- density(data_sphericity[[i]] )
  e= d
  e$x = d$y
  e$y = d$x
  plot(e,  xlim = (ylim), ylim= (xlim), xlab = "", ylab = "", axes=FALSE, main = "", zero.line = F)
  

  color_index = which(names(data_volumes )[i] == all_atlases_to_plot)
  polygon(e, col=colors[color_index], border=colors[color_index])
  par(new = T)
}



par(fig=c(0,0.45,0,0.545), new=T)
mar[2] = mar_2
par(mar = mar)
xlim = lim_vol
ylim = lim_sphericity
count = 1
for (i in index){

  x = data_volumes[[i]]
  y = data_sphericity[[i]]
  color_index = which(names(data_volumes )[i] == all_atlases_to_plot)
  plot(x,y,  xlim = xlim, ylim = ylim,
       col = colors[color_index], pch = pch[count], cex = 0.5,
       xlab = "", ylab = "", axes=FALSE, main = "")
  par(new = T)
  count = count + 1
}
box()
axis = par('usr')
axis(side=1,lwd=1)
axis(side=2,at=seq( floor( lim_sphericity[1]*10 )/10, ceiling( lim_sphericity[2]*10 )/10,0.1),lwd=1, las = 1)

par(xpd=NA)
ltr_x = par("usr")[1]
ltr_y = par("usr")[4]
text(ltr_x - ltr_x*1.16 , ltr_y+ ltr_y*0.2  , labels = "C.", cex = 2, font = 2 )


title(xlab = expression(paste("volume (log"[10]," mm"^"3", ")")), line = 2,font.lab = 1, cex.lab = 1.0)
title(ylab = "sphericity",line = 2.5, font.lab = 1, cex.lab = 1.0)
#mtext("Volume (log10 voxels)", side = 1, outer = T, line = -1.1 , font = 1, cex = 1.5)

par(new = T)
plot(unlist(mean_volumes[index]),  unlist(mean_sphericity[index]),
     col = "black",  bg =substr(colors[index],1,7),  xlim = xlim, ylim = ylim,
     xlab = "", ylab = "", axes=FALSE, main = "", cex = 1.8,
     pch = pch[1:length(index)])


#par(fig=c(cutoffs[1],1,cutoffs[2],1), new=T)


legend(x = "bottomleft", legend =standard_atlases_legend,
       col = substr(colors,1,7),
       bty = "n", cex= 1, pch = pch , xpd = T,
       y.intersp = 0.9, adj=0)

#legend(x = "bottomright", legend = "means", col = "black", bty = "n", cex= 1, pch = 23 , xpd = T, adj=0)


#text(  xlim[1], ylim[2], "Small & \nSpherical", adj = c(0, 1)  , font = 2, cex = 1.2)
#text(  xlim[2], ylim[1], "Large &      \nnon-spherical", adj = c(1, 0) , font = 2 , cex = 1.2)


#
#
#
# text(  text_x_1, text_y_1, " Small & \nSpherical", adj = c(0, 1)  , font = 2, cex = 1)
# text(  text_x_2, text_y_2, "Large &      \nnon-spherical", adj = c(1, 0) , font = 2 , cex = 1)
#
#
# arrows( text_x_1-0.05,  text_y_1+0.005, xlim[1], ylim[2], length = 0.1, angle = 30,
#         code = 2, col = par("fg"), lty = par("lty"),
#         lwd = 2)
# arrows(text_x_2+0.05, text_y_2-0.005, xlim[2], ylim[1], length = 0.1, angle = 30,
#        code = 2, col = par("fg"), lty = par("lty"),
#        lwd = 2)
#

text_x_1 =  xlim[1]+(xlim[2] - xlim[1])*0.1
text_y_1 = ylim[2]- ylim[2]*0.1
text_x_2 =  xlim[2]-xlim[2]*0.1
text_y_2 = ylim[1] + (ylim[2]- ylim[1])*0.1

text(  text_x_1, text_y_1, " Small & \nSpherical", adj = c(0, 1)  , font = 3, cex = 1)
text(  text_x_2, text_y_2, "Large &      \nnon-spherical", adj = c(1, 0) , font = 3 , cex = 1)


arrows( text_x_1-0.05,  text_y_1+0.005, xlim[1], ylim[2], length = 0.1, angle = 30,
        code = 2, col = par("fg"), lty = par("lty"),
        lwd = 2)
arrows(text_x_2+0.05, text_y_2-0.005, xlim[2], ylim[1], length = 0.1, angle = 30,
       code = 2, col = par("fg"), lty = par("lty"),
       lwd = 2)
































index= random_atlas_index
ylim = c(0,20e-0)
lim_sphericity = c(0.0,0.8)
#pdf(paste0(github_directory, "figure3/atlas_random_volumesVSsphericity.pdf"), width = 7, height = 7)
title = "Random Atlases:\n Volumes vs Sphericity"







mar =  c(0,0,0,0)
mar_1 = 4
mar_2 = 4
mar_3 = 2
mar[2] = mar_2
mar[3] = mar_3
par(mar = mar)



cutoffs = c(0.5,0.78)
par(fig=c(0.5,0.95,0.54,0.78), new=T)
par(xpd = F)
xlim = lim_vol
ylim = density_lim
count = 1
for (i in index){
  d <- density(data_volumes[[i]] )
  plot(d,  xlim = xlim, ylim= ylim, xlab = "", ylab = "", axes=FALSE, main = "", zero.line = F)
  polygon(d, col=random_atlas_colors[count], border=random_atlas_colors[count])
  count =  count + 1
  par(new = T)
}
mtext("Random Atlases:\nVolumes vs Sphericity", side = 3, outer = F, line = -5.75 , font = 3, cex = 1.3)


#title(main = title,line = -1.7, font.lab = 2, cex.main = 2)

par(fig=c(0.95,1,0,0.54), new=T)
mar[1] = mar_1
mar[2] = 0
mar[3] = 0
par(mar = mar)
xlim = lim_sphericity
ylim = c(0,30)
count =  1
for (i in index){

  d <- density(data_sphericity[[i]] )
  e= d
  e$x = d$y
  e$y = d$x
  plot(e,  xlim = (ylim), ylim= (xlim), xlab = "", ylab = "", axes=FALSE, main = "", zero.line = F)
  polygon(e, col=random_atlas_colors[count], border=random_atlas_colors[count])
  count =  count + 1
  par(new = T)
}





par(fig=c(0.5,0.95,0,0.545), new=T)
mar[2] = mar_2
par(mar = mar)
xlim = lim_vol
ylim = lim_sphericity
count =  1
pch_count = 0
for (i in index){

  x = data_volumes[[i]]
  y = data_sphericity[[i]]
  plot(x,y,  xlim = xlim, ylim = ylim,
       col = random_atlas_colors[count], pch = (pch_count+count), cex = 0.5,
       xlab = "", ylab = "", axes=FALSE, main = "")
  count =  count + 1
  par(new = T)
}
box()
axis = par('usr')
axis(side=1,lwd=1)
axis(side=2,at=seq( floor( lim_sphericity[1]*10 )/10, ceiling( lim_sphericity[2]*10 )/10,0.1),lwd=1, las = 1)

par(xpd=NA)
ltr_x = par("usr")[1]
ltr_y = par("usr")[4]
text(ltr_x - ltr_x*2.5 , ltr_y+ ltr_y*0.2  , labels = "D.", cex = 2, font = 2 )


title(xlab = expression(paste("volume (log"[10]," mm"^"3", ")")), line = 2,font.lab = 1, cex.lab = 1.0)
title(ylab = "sphericity",line = 2.5, font.lab = 1, cex.lab = 1.0)

par(new = T)
plot(unlist(mean_volumes[index]),  unlist(mean_sphericity[index]),
     col = "black",  bg =substr(random_atlas_colors,1,7),  xlim = xlim, ylim = ylim,
     xlab = "", ylab = "", axes=FALSE, main = "", cex = 1.8,
     pch = pch)


#par(fig=c(cutoffs[1],1,cutoffs[2],1), new=T)


legend(x = "bottomleft", legend = random_atlases_legend,
       col = substr(random_atlas_colors,1,7),
       bty = "n", cex= 1.2, pch = pch , xpd = T,
       y.intersp = 0.8, adj=0)

#legend(x = "bottomright", legend = "means", col = "black", bty = "n", cex= 1, pch = 23 , xpd = T, adj=0)




dev.off()

##########################################################################################################################################















#Supplemental Data
#Volumes and Sphericity individual graphs

#png(file.path(fpath_morphology, "volumes_and_sphericity_ALL.png"), width = 8.5, height = 11,  units = "in", res = 600)
pdf(file.path(fpath_morphology, "volumes_and_sphericity_ALL.pdf"), width = 8.5, height = 11)




atlas_names2 = atlases["atlas_shortname"][,1]
order_to_plot = c(1:38, seq(39, 108,5))

random_atlases_legend = rep(NA, length(random_atlases_to_plot))
for (i in 39:length(atlas_names2)){
  atlas_names2[i] = paste0("Random Atlas ", as.numeric(substring(atlas_names2[i] , 12,18) ))
}



lim_vol = c(0, 6)
lim_sphericity = c(0, 0.8)
mar =  c(0,0,0,0)
oma = c(4,4,3,0.1)
par(mfrow=c(9,6), mar = mar, oma = oma)
colors = rainbow(length(order_to_plot), s = 0.7, v = 0.8, start = 0, end = 0.99, alpha = 0.65)

for (i in 1:length(order_to_plot)){
  
  x = data_volumes[[order_to_plot[i]]]
  y = data_sphericity[[order_to_plot[i]]]
  #par(new = F)
  plot(x,y,  xlim = lim_vol, ylim = lim_sphericity,
       col = colors[i], pch = 16, cex = 1,
       xlab = "", ylab = "", axes=F, main = "")
  box()
  par(new = T)
  
  mean_v = unlist(mean_volumes[order_to_plot[i]])
  mean_s = unlist(mean_sphericity[order_to_plot[i]])
  plot(mean_v,  mean_s,
       col = "black",  xlim = lim_vol, ylim = lim_sphericity,
       xlab = "", ylab = "", axes=FALSE, main = "", cex = 1,
       pch = 16)
  text(  (par("usr")[2] - par("usr")[1])/2, par("usr")[3]  +  (par("usr")[4] - par("usr")[3])*0.15,  paste0("(", round(mean_v, 2), ", ", round(mean_s,2), ")"  ) , adj = c(0.5, 1)  , font = 3, cex = 1)
  
  if    (       any(i == c( seq(1, 60, 6)))   ){
    axis = par('usr')
    axis(side=2,at=seq( floor( lim_sphericity[1]*10 )/10, ceiling( lim_sphericity[2]*10 )/10,0.1),lwd=1, las = 1, cex.axis=0.7)
  }
  if    (     i>54  ){
    axis = par('usr')
    axis(side=1,lwd=1)
  }
  
  #Title
  text(  (par("usr")[2] - par("usr")[1])/2, par("usr")[3]  +  (par("usr")[4] - par("usr")[3])*0.98,  atlas_names2[[order_to_plot[i]]] , adj = c(0.5, 1)  , font = 3, cex = 1)
  
}

cex = 1
mtext(expression(paste("volume (log"[10]," mm"^"3", ")")), side = 1, outer = T, line = 3 , font = 1, cex = cex)
mtext("sphericity", side = 2, outer = T, line = 2.5 , font = 1, cex = cex)
mtext("Atlas Morphology: Sizes and Shapes", side = 3, outer = T, line = 1 , font = 1, cex = 1.5)



dev.off()









