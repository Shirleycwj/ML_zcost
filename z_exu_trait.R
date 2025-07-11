## This script process original source of exudation data to extract root trait value
## Date: 241212

## Inialize
rm(list=ls())
library(readxl)
library(raster)
library(ggplot2)
library(ggpmisc)
library(patchwork)
library(MASS)
library(relaimpo)
library(car)
library(ggsci)
library(tidyverse)
# setwd(r"(C:\Users\wc1317\OneDrive - Imperial College London\Postdoc_240501\C allocation framework\root_exudation_collective)")
setwd(r"(D:\OneDrive - Imperial College London\Postdoc_240501\C allocation framework\root_exudation_collective)")
## read organized data
dt_exu <- read.csv("root_exu_trait.csv")
# dt_exu$Root.diameter <- as.numeric(dt_exu$Root.diameter)
olp <- stack("../OlsenP_mgkg-1_World_Aug2022_ver.tif")
z_cost <- stack("../apprent_z.nc")
olp_wgs84 <- projectRaster(olp,crs = crs(z_cost))

cn_map <- stack("../cn_map.nc")
ph_map <- stack("../ph_map.nc")

location = data.frame(x = dt_exu$Lon,
                      y = dt_exu$Lat)  # negative for southern hemisphere
coordinates(location) <- ~x+y
mypoints = SpatialPoints(location, CRS("+init=epsg:4326"))

p_loc <- raster::extract(olp_wgs84, mypoints)
dt_exu$olp <- p_loc

z_loc <- raster::extract(z_cost, mypoints)
dt_exu$z <- z_loc[,1]

cn_loc <- raster::extract(cn_map, mypoints)
dt_exu$cn <- cn_loc[,1]

ph_loc <- raster::extract(ph_map, mypoints)
dt_exu$ph <- ph_loc[,1]

## set up multiple linear regression
dt_mod <- dt_exu %>% 
  drop_na(z)

# stepwise model
fit <- lm(z~Exudation.rate+Root.diameter+Specific.root.length+Root.tissue.density+olp, data = dt_mod)
step <- stepAIC(fit, direction="both")
step$anova # display results

# calculate relative importance
library(relaimpo)
relimp <- calc.relimp(step,type=c("lmg"),
                      rela=TRUE)
#par(mfrow=c(1,1))
plot(relimp)

# partial regression plot
avPlots(step)

## separate to AM and ECM
dt_am <- dt_mod %>%
  filter(mt=="AM")
dt_ecm <- dt_mod %>%
  filter(mt=="EM")
# am
fit_am <- lm(z~Exudation.rate+Root.diameter+Specific.root.length+Root.tissue.density+olp, data = dt_am)
step_am <- stepAIC(fit_am, direction="both")
step_am$anova # display results
# calculate relative importance
relimp_am <- calc.relimp(step_am,type=c("lmg"),
                      rela=TRUE)
#par(mfrow=c(1,1))
plot(relimp_am)
# partial regression plot
avPlots(step_am)

# EcM
fit_ecm <- lm(z~Exudation.rate+Root.diameter+Specific.root.length+Root.tissue.density+olp, data = dt_ecm)
step_ecm <- stepAIC(fit_ecm, direction="both")
step_ecm$anova # display results
# calculate relative importance
library(relaimpo)
relimp_ecm <- calc.relimp(step_ecm,type=c("lmg"),
                         rela=TRUE)
#par(mfrow=c(1,1))  
plot(relimp_ecm)
# partial regression plot
avPlots(step_ecm)

## look at what affects exudation rate
fit_exu <- lm(Exudation.rate~cn+ph+Root.diameter+Specific.root.length+Root.tissue.density+olp, data = dt_am)
# fit_exu <- lm(Exudation.rate~Specific.root.length+olp, data = dt_am)

step_exu <- stepAIC(fit_exu, direction="both")
step_exu$anova # display results

# calculate relative importance
relimp_exu <- calc.relimp(step_exu,type=c("lmg"),
                      rela=TRUE)
#par(mfrow=c(1,1))
plot(relimp_exu)
# partial regression plot
avPlots(step_exu)



