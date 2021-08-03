path = "/media/arevell/sharedSSD/linux/papers/paper005/" #path to where the paper directory is stored - locally or remotely

ifpath_network_measures = file.path(path, "data/data_processed/network_measures/structure")
ifpath_mean_volumes_and_sphericity = file.path(path, "data/data_processed/volumes_and_sphericity_means")
ofpath_network_measures = file.path(path, "brainAtlas/figures/network_measures")

setwd(path)
library("ggplot2")
library("ggpubr")
library("cowplot")
library("gridExtra")
library("grid")

pdf(file.path(ofpath_network_measures, "network_measures.pdf"), width = 11, height = 6)
#png(file.path(ofpath_network_measures, "network_measures.png"), width = 11, height = 6,  units = "in", res = 600)
#Get mean volumes and sphericity data
data_volumes_means = read.csv(file.path(ifpath_mean_volumes_and_sphericity,"volumes_and_sphericity.csv"), stringsAsFactors = F, header = T) 





#Random Atlases
data_random = read.csv(file.path(ifpath_network_measures,"network_measures_random_atlas.csv"), stringsAsFactors = T, header = T)
names(data_random)[which(names(data_random) == "RID")] = "subject"
#adding mean volumes to data 
data_random = cbind(data_random,"volume" =   0)
for (i in 1:length(data_random$subject)){
  data_random$volume[i] =  log10( data_volumes_means[data_volumes_means$X == data_random$atlas[i], 2])
}
group_random_mean = aggregate(data_random[, c(4:9)], list( data_random$subject, data_random$atlas), mean)
group_random_sd = aggregate(data_random[, c(4:9)], list(data_random$subject, data_random$atlas), sd)

colnames(group_random_mean)[1] = colnames(data_random)[1]
colnames(group_random_mean)[2] = colnames(data_random)[2]


#Standard Atlases
data_standard = read.csv(file.path(ifpath_network_measures,"network_measures_standard_atlas.csv"), stringsAsFactors = T, header = T)
names(data_standard)[which(names(data_standard) == "RID")] = "subject"
#adding mean volumes to data 
data_standard = cbind(data_standard,"volume" =   0)
for (i in 1:length(data_standard$subject)){
  data_standard$volume[i] =  log10( data_volumes_means[data_volumes_means$X == data_standard$atlas[i], 2])
}

levels(data_standard$atlas)
#rename atlases
levels(data_standard$atlas)[levels(data_standard$atlas)=="AAL"] <- "AAL v1"
levels(data_standard$atlas)[levels(data_standard$atlas)=="AAL2"] <- "AAL v2"
levels(data_standard$atlas)[levels(data_standard$atlas)=="AAL3v1_1mm"] <- "AAL v3"
levels(data_standard$atlas)[levels(data_standard$atlas)=="AAL_JHU_combined"] <- "AAL-JHU"
levels(data_standard$atlas)[levels(data_standard$atlas)=="AICHA"] <- "AICHA"
levels(data_standard$atlas)[levels(data_standard$atlas)=="BN_Atlas_246_1mm"] <- "BN"
levels(data_standard$atlas)[levels(data_standard$atlas)=="cc200_roi_atlas"] <- "Craddock 200"
levels(data_standard$atlas)[levels(data_standard$atlas)=="cc400_roi_atlas"] <- "Craddock 400"
levels(data_standard$atlas)[levels(data_standard$atlas)=="ez_roi_atlas"] <- "EZ"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Gordon_Petersen_2016_MNI"] <- "Gordon"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Hammersmith_atlas_n30r83_SPM5"] <- "Hammersmith"
levels(data_standard$atlas)[levels(data_standard$atlas)=="HarvardOxford-combined"] <- "HO Cort + SubCort"
levels(data_standard$atlas)[levels(data_standard$atlas)=="HarvardOxford-cort-maxprob-thr25-1mm"] <- "HO Cort Symmetric"
levels(data_standard$atlas)[levels(data_standard$atlas)=="HarvardOxford-cort-NONSYMMETRIC-maxprob-thr25-1mm"] <- "HO Cort NonSymmetric"
levels(data_standard$atlas)[levels(data_standard$atlas)=="HarvardOxford-sub-ONLY_maxprob-thr25-1mm"] <- "HO SubCort"
levels(data_standard$atlas)[levels(data_standard$atlas)=="JHU-ICBM-labels-1mm"] <- "JHU"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Juelich-maxprob-thr25-1mm"] <- "Juelich"
levels(data_standard$atlas)[levels(data_standard$atlas)=="MNI-maxprob-thr25-1mm"] <- "MNI Lobar"
levels(data_standard$atlas)[levels(data_standard$atlas)=="OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_v2"] <- "DKT"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_100Parcels_17Networks_order_FSLMNI152_1mm"] <- "Schaefer 17 100"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm"] <- "Schaefer 7 100"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_200Parcels_17Networks_order_FSLMNI152_1mm"] <- "Schaefer 17 200"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm"] <- "Schaefer 7 200"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_300Parcels_17Networks_order_FSLMNI152_1mm"] <- "Schaefer 17 300"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_300Parcels_7Networks_order_FSLMNI152_1mm"] <- "Schaefer 7 300"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm"] <- "Schaefer 17 400"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm"] <- "Schaefer 7 400"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_500Parcels_17Networks_order_FSLMNI152_1mm"] <- "Schaefer 17 500"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_500Parcels_7Networks_order_FSLMNI152_1mm"] <- "Schaefer 7 500"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_600Parcels_17Networks_order_FSLMNI152_1mm"] <- "Schaefer 17 600"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_600Parcels_7Networks_order_FSLMNI152_1mm"] <- "Schaefer 7 600"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_700Parcels_17Networks_order_FSLMNI152_1mm"] <- "Schaefer 17 700"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_700Parcels_7Networks_order_FSLMNI152_1mm"] <- "Schaefer 7 700"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_800Parcels_17Networks_order_FSLMNI152_1mm"] <- "Schaefer 17 800"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_800Parcels_7Networks_order_FSLMNI152_1mm"] <- "Schaefer 7 800"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_900Parcels_17Networks_order_FSLMNI152_1mm"] <- "Schaefer 17 900"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_900Parcels_7Networks_order_FSLMNI152_1mm"] <- "Schaefer 7 900"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_1mm"] <- "Schaefer 17 1000"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_1mm"] <- "Schaefer 7 1000"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Talairach-labels-1mm"] <- "Talairach"
levels(data_standard$atlas)[levels(data_standard$atlas)=="MMP_in_MNI_resliced"] <- "Glasser"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask_resliced"] <- "Yeo 17 Liberal"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_resliced"] <- "Yeo 17"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask_resliced"] <- "Yeo 7 Liberal"
levels(data_standard$atlas)[levels(data_standard$atlas)=="Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_resliced"] <- "Yeo 7"



#sequestering  atlases
data_standard['Plot_Group'] = as.character(data_standard$atlas)
group_random_mean['Plot_Group'] = "Random"
data_standard[grepl("Schaefer", data_standard$atlas), ]$Plot_Group = "Schaefer"
data_standard[grepl("AAL v", data_standard$atlas), ]$Plot_Group = "AAL"
data_standard[grepl("Craddock", data_standard$atlas), ]$Plot_Group = "Craddock"
data_standard[grepl("HO", data_standard$atlas), ]$Plot_Group = "HO"
data_standard['Plot_Group'] = as.factor(data_standard$Plot_Group)
group_random_mean['Plot_Group'] = as.factor(group_random_mean$Plot_Group)


group_standard_mean = data_standard # aggregate(data_standard[, c(3:8)], list( data_standard$subject, data_standard$atlas), mean)
group_standard_sd = aggregate(data_standard[, c(3:8)], list(data_standard$subject, data_standard$atlas), sd)
data = rbind(group_standard_mean, group_random_mean)

#Sizes of points
data$size = 2
size = 0.8
data[grepl("Random", data$atlas), ]$size = size
data[grepl("Schaefer", data$atlas), ]$size = size
data[grepl("AAL v", data$atlas), ]$size = size
data[grepl("HO", data$atlas), ]$size = size
data[grepl("Craddock", data$atlas), ]$size = size


plot_index = c(1, 5, 8, 12, 19, 21, seq(22, 40, 2), 44, 46, 47:60)
data_plot = aggregate(data[, 3:8], list(data$atlas), mean)
data_plot = data_plot[plot_index, ]
names(data_plot)[1] = "atlas"
#sequestering  atlases
data_plot['Plot_Group'] = as.character(data_plot$atlas)
data_plot[grepl("Schaefer", data_plot$atlas), ]$Plot_Group = "Schaefer"
data_plot[grepl("Random", data_plot$atlas), ]$Plot_Group = "Random"
data_plot['Plot_Group'] = as.factor(data_plot$Plot_Group)
data_plot$size = 3
data_plot[grepl("Random", data_plot$atlas), ]$size = 2
data_plot[grepl("Schaefer", data_plot$atlas), ]$size = 2

levels(data_plot$Plot_Group)
colors = rainbow(nlevels(data_plot$Plot_Group)-3, s = 0.8, v = 0.8, start = 0.15)
colors_all = c("#993333", "#FD6A02", colors[1], "#1d731d","#5757ff","#5494d4",   "#00000077", "#33339999",  colors[6:7]  )
shapes = c(16,17,15,11,17, 25, 16, 16 , 8, 9)
font_size = 14


density <- ggplot(data = data_plot, aes(volume, Density , color=Plot_Group, shape=Plot_Group), size=data_plot$size) + geom_point(size =data_plot$size)  
degree_mean <- ggplot(data = data_plot, aes(volume, degree_mean , color=Plot_Group, shape=Plot_Group), size=data_plot$size) + geom_point(size =data_plot$size)  
clustering_coefficient_mean <- ggplot(data = data_plot, aes(volume, clustering_coefficient_mean , color=Plot_Group, shape=Plot_Group), size=data_plot$size) + geom_point(size =data_plot$size)  
characteristic_path_length <- ggplot(data = data_plot, aes(volume, characteristic_path_length , color=Plot_Group, shape=Plot_Group), size=data_plot$size) + geom_point(size =data_plot$size)  
small_worldness <- ggplot(data = data_plot, aes(volume, small_worldness , color=Plot_Group, shape=Plot_Group), size=data_plot$size) + geom_point(size =data_plot$size)  


p1 = density + scale_shape_manual(values=shapes) + scale_color_manual(values = colors_all) +  
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
        panel.background = element_blank(), axis.line = element_line(colour = "black"), 
        axis.title.y = element_text(size=14),legend.position = "none",
        plot.title = element_text(lineheight=0.8,hjust = 0.5,size=font_size), axis.text=element_text(size = font_size)) +
  stat_summary(data = data_plot, geom = "line", fun = "mean", size = 1,) +
  xlab("") +  ylab("") + ggtitle("Density") + scale_x_continuous(limits=c(2, 6)) 


p2 = degree_mean + scale_shape_manual(values=shapes) + scale_color_manual(values = colors_all) +  
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
        panel.background = element_blank(), axis.line = element_line(colour = "black"), 
        axis.title.y = element_text(size=14),legend.position = "none",
        plot.title = element_text(lineheight=0.8,hjust = 0.5,size=font_size), axis.text=element_text(size = font_size)) +
  stat_summary(data = data_plot, geom = "line", fun = "mean", size = 1,) +
  xlab("") +  ylab("") + ggtitle("Mean Degree")+ scale_x_continuous(limits=c(2, 6))


p3 = characteristic_path_length + scale_shape_manual(values=shapes) + scale_color_manual(values = colors_all) +  
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
        panel.background = element_blank(), axis.line = element_line(colour = "black"), 
        axis.title.y = element_text(size=14),legend.position = "none",
        plot.title = element_text(lineheight=0.8,hjust = 0.5,size=font_size), axis.text=element_text(size = font_size)) +
  stat_summary(data = data_plot, geom = "line", fun = "mean", size = 1,) +
  xlab("") +  ylab("")  + ggtitle("Characteristic Path Length")+ scale_x_continuous(limits=c(2, 6))

p4 = clustering_coefficient_mean + scale_shape_manual(values=shapes) + scale_color_manual(values = colors_all) +  
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
        panel.background = element_blank(), axis.line = element_line(colour = "black"), 
        axis.title.y = element_text(size=14),legend.position = "none",
        plot.title = element_text(lineheight=0.8,hjust = 0.5,size=font_size), axis.text=element_text(size = font_size)) +
  stat_summary(data = data_plot, geom = "line", fun = "mean", size = 1,) +
  xlab("") +  ylab("") + ggtitle("Mean Clustering Coefficient")+ scale_x_continuous(limits=c(2, 6))

p5 = small_worldness + scale_shape_manual(values=shapes) + scale_color_manual(values = colors_all) +  
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
        panel.background = element_blank(), axis.line = element_line(colour = "black"), 
        axis.title.y = element_text(size=14),legend.position = "none",
        plot.title = element_text(lineheight=0.8,hjust = 0.5,size=font_size), axis.text=element_text(size = font_size)) +
  stat_summary(data = data_plot, geom = "line", fun = "mean", size = 1,) +
  xlab("") +  labs(fill = "Atlas") +  ylab("") + ggtitle("Small Worldness")+ scale_x_continuous(limits=c(2, 6))
  
#top=textGrob("Network Measures Across Different Atlasese",gp=gpar(fontsize=40,font=3))

col3 <- guide_legend(ncol = 2, override.aes = list(size = 3))

small_worldness_leg = small_worldness + scale_shape_manual(values=shapes, guide = col3) +   scale_color_manual(values = colors_all) +
  theme(legend.key=element_blank(), legend.text = element_text(colour="black", size =font_size), legend.title=element_blank(), legend.spacing.x = unit(0.4, 'cm'),
        plot.title = element_text(lineheight=0.8,hjust = 0.5,size=font_size))  + ggtitle("Atlas")

l <- get_legend(small_worldness_leg)


title = textGrob("Network Topology Differences", gp=gpar(fontsize=font_size*1.35, fontface="bold"))
label = expression(paste("volume (log"[10]," mm"^"3", ")"))
xlab = textGrob(label, gp=gpar(fontsize=font_size))

grid.arrange(p1, p2, p3, p4, p5,l, nrow = 2, top = title, bottom = xlab, padding = unit(0.5, "line"),  heights=unit(c(2.6,2.6), c("in", "in")))


#title(xlab = "Volume (log10 Voxels)", line = 2,font.lab = 1, cex.lab = 1.0)
dev.off()


#legend.position =c(0.8, 0.6)
























#Plotting all network measures


#png(file.path(ofpath_network_measures, "ALL_network_measures.png"), width = 8.5, height = 11,  units = "in", res = 600)
pdf(file.path(ofpath_network_measures, "ALL_network_measures.pdf"), width = 8.5, height = 11)

#sequestering  atlases
data_standard['Plot_Group'] = as.character(data_standard$atlas)
group_random_mean['Plot_Group'] = "Random"
data_standard[grepl("Schaefer", data_standard$atlas), ]$Plot_Group = "Schaefer"
data_standard[grepl("AAL", data_standard$atlas), ]$Plot_Group = "AAL"
data_standard[grepl("Craddock", data_standard$atlas), ]$Plot_Group = "Craddock"
data_standard[grepl("HO", data_standard$atlas), ]$Plot_Group = "HO"
data_standard[grepl("Yeo", data_standard$atlas), ]$Plot_Group = "Yeo"
data_standard['Plot_Group'] = as.factor(data_standard$Plot_Group)
group_random_mean['Plot_Group'] = as.factor(group_random_mean$Plot_Group)


group_standard_mean = data_standard # aggregate(data_standard[, c(3:8)], list( data_standard$subject, data_standard$atlas), mean)
group_standard_sd = aggregate(data_standard[, c(3:8)], list(data_standard$subject, data_standard$atlas), sd)
data = rbind(group_standard_mean, group_random_mean)


data$Plot_Group

#Sizes of points
data$size = 2

mar =  c(0,2,0,0)
oma = c(4,1,3,0.5)
xlim = c(2,6)
ylim_density = c(0,1)
ylim_degree = c(0, 275)
ylim_clustering =  c(0.4, 1.05)
ylim_cpl = c(1, 3.5)
ylim_SW = c(0, 12)
pch = c(17,  15, 8, 5, 12, 2)
random_col =  "#00000055"
schafer_col =  "#33339955"
cex_title = 0.5

line = 0
par(mfrow=c(15,5), mar = mar, oma = oma)
for (i in 1:(nlevels(data$Plot_Group)  )){
  
  name = levels(data$Plot_Group)[i]
  x = data[grepl(name, data$atlas), ]
  
  if (  ( name == "Schaefer" | name == "Random") == F  ){
    
    
    x$Plot_Group = as.character(x$atlas)
    x$Plot_Group = as.factor(x$Plot_Group)
    
    random = data[grepl("Random", data$atlas), ]
    schaefer = data[grepl("Schaefer", data$atlas), ]
    
    plot = rbind(x, random, schaefer)
    colors = rainbow(length(unique(x$Plot_Group)), s = 0.8, v = 0.8)
    
    cex = 1
    #Desnity 
    plot(random$volume, random$Density, pch = 16, col = random_col, xlim = xlim, ylim = ylim_density, axes=FALSE); par(new = T )
    plot(schaefer$volume, schaefer$Density, pch = 16, col = schafer_col, xlim = xlim, ylim = ylim_density, axes=FALSE)
    for (a in 1:length(unique(x$Plot_Group))){par(new = T )
      x_plot = x[x$Plot_Group == x$Plot_Group[a],]
      plot(x_plot$volume, x_plot$Density, pch = pch[a], col = colors[a], xlim = xlim, ylim = ylim_density, cex = cex, axes=FALSE)}
    box()
    axis(side=2,lwd=1, las = 1, cex.axis=0.7)
    if (i == nlevels(data$Plot_Group)-1){axis(side=1,lwd=1)}
    if (i == 1){  title( main= "Density"   , font = 3, line =line, cex.main = cex_title)}
    
    legend(1.9, 1.1, legend=  levels(x$Plot_Group),  box.lty=0,  bty = "n",  y.intersp = 0.8, adj=0, xpd = T,cex = 0.8,
           col= colors, pch = pch[1:length(x$Plot_Group)])
    
    
    #degree_mean
    plot(random$volume, random$degree_mean, pch = 16, col = random_col, xlim = xlim, ylim = ylim_degree, axes=FALSE); par(new = T )
    plot(schaefer$volume, schaefer$degree_mean, pch = 16, col = schafer_col, xlim = xlim, ylim = ylim_degree, axes=FALSE)
    for (a in 1:length(unique(x$Plot_Group))){par(new = T )
      x_plot = x[x$Plot_Group == x$Plot_Group[a],]
      plot(x_plot$volume, x_plot$degree_mean, pch = pch[a], col = colors[a], xlim = xlim, ylim = ylim_degree, cex = cex, axes=FALSE)}
    box()
    axis(side=2,lwd=1, las = 1, cex.axis=0.7)
    if (i == nlevels(data$Plot_Group)-1){axis(side=1,lwd=1)}
    if (i == 1){  title( main= "Mean Degree"   , font = 3, line =line, cex.main = cex_title)}
    
    #characteristic_path_length 
    plot(random$volume, random$characteristic_path_length, pch = 16, col = random_col, xlim = xlim, ylim = ylim_cpl, axes=FALSE); par(new = T )
    plot(schaefer$volume, schaefer$characteristic_path_length, pch = 16, col = schafer_col, xlim = xlim, ylim = ylim_cpl, axes=FALSE)
    for (a in 1:length(unique(x$Plot_Group))){par(new = T )
      x_plot = x[x$Plot_Group == x$Plot_Group[a],]
      plot(x_plot$volume, x_plot$characteristic_path_length, pch = pch[a], col = colors[a], xlim = xlim, ylim = ylim_cpl, cex = cex, axes=FALSE)}
    box()
    axis(side=2,lwd=1, las = 1, cex.axis=0.7)
    if (i == nlevels(data$Plot_Group)-1){axis(side=1,lwd=1)}
    if (i == 1){  title( main= "Characteristic Path Length"   , font = 3, line =line, cex.main = cex_title)}
    
    #clustering_coefficient_mean 
    plot(random$volume, random$clustering_coefficient_mean, pch = 16, col = random_col, xlim = xlim, ylim = ylim_clustering, axes=FALSE); par(new = T )
    plot(schaefer$volume, schaefer$clustering_coefficient_mean, pch = 16, col = schafer_col, xlim = xlim, ylim = ylim_clustering, axes=FALSE)
    for (a in 1:length(unique(x$Plot_Group))){par(new = T )
      x_plot = x[x$Plot_Group == x$Plot_Group[a],]
      plot(x_plot$volume, x_plot$clustering_coefficient_mean, pch = pch[a], col = colors[a], xlim = xlim, ylim = ylim_clustering, cex = cex, axes=FALSE)}
    box()
    axis(side=2,lwd=1, las = 1, cex.axis=0.7)
    if (i == nlevels(data$Plot_Group)-1){axis(side=1,lwd=1)}
    if (i == 1){  title( main= "Mean Clustering Coefficient"   , font = 3, line =line ,cex.main = cex_title)}
    
    #small_worldness 
    plot(random$volume, random$small_worldness, pch = 16, col = random_col, xlim = xlim, ylim = ylim_SW, axes=FALSE); par(new = T )
    plot(schaefer$volume, schaefer$small_worldness, pch = 16, col = schafer_col, xlim = xlim, ylim = ylim_SW, axes=FALSE)
    for (a in 1:length(unique(x$Plot_Group))){par(new = T )
      x_plot = x[x$Plot_Group == x$Plot_Group[a],]
      plot(x_plot$volume, x_plot$small_worldness, pch = pch[a], col = colors[a], xlim = xlim, ylim = ylim_SW, cex = cex, axes=FALSE)}
    box()
    axis(side=2,lwd=1, las = 1, cex.axis=0.7)
    if (i == nlevels(data$Plot_Group)-1){axis(side=1,lwd=1)}
    if (i == 1){  title( main= "Small Worldness"   , font = 3, xpd = T, line =line ,cex.main = cex_title)}
    

  }
}
mtext( "Network Topology Differences: All Atlases", side = 3, outer = T, line = 1.2)
mtext( "                     Density                                   Mean Degree                   Characteristic Path Length    Mean Clustering Coefficient            Small Worldness", side = 3, outer = T, line = 0, font = 2, cex = 0.7, adj = 0)
mtext(  expression(paste("volume (log"[10]," mm"^"3", ")")), side = 1, outer = T, line = 3)
dev.off()


