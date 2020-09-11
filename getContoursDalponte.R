library(lidR)
library(parallel)
library(rgdal)

args <- commandArgs(trailingOnly=TRUE)
cat('running script with parameters ', args, '\n')
cat('current working directory ', getwd(), '\n')
data_path <- args[1] # path to folder containing tiles or canopy height models
ws <- as.numeric(args[2]) # window size 5
hmin <- as.numeric(args[3]) # minimum height
outdir <- args[4] # output directory
files <- list.files(path=data_path, pattern="*.tif", full.names=TRUE, recursive=FALSE)
cat(length(files), ' files in folder ', data_path, '\n')
getTtopsAndContours <- function(filename, ws, hmin, outdir) {
    # Should do the same than itcIMG
    cat('processing file ', filename, '\n')
    tile <- stack(filename) # Read multiband .tif file 
    chm <- tile[[461]] # Select CHM Channel
    chm <- focal(chm, w=matrix(1,3,3), data=chm[,], fun=function(x){mean(x, na.rm=T)}) #Smoothen chm
    chm[is.na(chm[])] <- 0 # Set NA values as zero
    ttops <- tree_detection(chm, lmf(ws=ws, hmin=hmin, shape="circular")) # Get treetops
    crowns <- dalponte2016(chm, ttops, th_tree=hmin, th_seed=0.65, th_cr=0.5,
                           max_cr=5)() # Segment
    
    # Postprocessing copy-pasted: https://github.com/cran/itcSegment/blob/master/R/itcIMG.R
    crowns.shp <- rasterToPolygons(crowns, n=4, na.rm=TRUE, digits=12, dissolve=TRUE) # Convert to polygons
    names(crowns.shp) <- "value"
    HyperCrowns <- crowns.shp[crowns.shp@data[,1]!=0,]
    HyperCrowns$X<-round(coordinates(HyperCrowns)[,1],2)
    HyperCrowns$Y<-round(coordinates(HyperCrowns)[,2],2)
    HyperCrowns$Height_m<-round(extract(chm,HyperCrowns,fun=max)[,1],2)
    HCbuf<-rgeos::gBuffer(HyperCrowns,width=-res(chm)[1]/2,byid=T)
    ITCcv<-rgeos::gConvexHull(HCbuf,byid=T)
    ITCcvSD<-sp::SpatialPolygonsDataFrame(ITCcv,data=HCbuf@data,match.ID=F)
    ITCcvSD$CA_m2<-round(unlist(lapply(ITCcvSD@polygons,function(x){methods::slot(x,"area")})),2)
    ITCcvSD<-ITCcvSD[ITCcvSD$CA_m2>1,]
    #proj4string(ITCcvSD)<-sp::CRS(paste("+init=epsg:32635"))
    #proj4string(ttops)<-sp::CRS(paste("+init=epsg:32635"))

    # Save to file
    tile_id <- sub(pattern="(.*)\\..*$", replacement="\\1", basename(filename))
    ttop_name <- paste('ttops_', tile_id, sep="")
    crown_name <- paste('crowns_', tile_id, sep="")
    writeOGR(obj=ttops, dsn=outdir, layer=ttop_name, driver='ESRI Shapefile')
    writeOGR(obj=ITCcvSD, dsn=outdir, layer=crown_name, driver='ESRI Shapefile')
    return()
}

mclapply(files, getTtopsAndContours, ws=ws, hmin=hmin, outdir=outdir, mc.cores=10)
