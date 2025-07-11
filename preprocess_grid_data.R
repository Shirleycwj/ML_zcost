# extract soil properties from global gridded dataset
library(raster)
library(sp)
library(gstat)
library(sf)
library(dplyr)

root_depth <- stack(r"(C:\Users\shirl\OneDrive - Imperial College London\Postdoc_240501\ICCS_summer_school\zroot_cwd80.nc)")
setwd(r"(C:\Users\shirl\OneDrive - Imperial College London\Postdoc_240501\C allocation framework\root_exudation_collective)")
olp <- stack("../OlsenP_mgkg-1_World_Aug2022_ver.tif")
z_cost <- stack("../apprent_z.nc")
olp_wgs84 <- projectRaster(olp,crs = crs(z_cost))
ph_map <- stack("../ph_map.nc")
cn_map <- stack("../cn_map.nc")

writeRaster(olp_wgs84, filename = "olp_raster.nc", format = "CDF", overwrite = TRUE)

site_outer <- read.csv(r"(C:\Users\shirl\OneDrive - Imperial College London\Postdoc_240501\ICCS_summer_school\FR_alltraits_sites_outer.csv)")

# location = data.frame(x = site_outer$SiteLongitude,
#                       y = site_outer$SiteLatitude)  # negative for southern hemisphere
# coordinates(location) <- ~x+y
# mypoints = SpatialPoints(location, CRS("+init=epsg:4326"))

# Create data frame of coordinates
location <- data.frame(ID = seq(1,699),  # include ID for reference
                       lon = site_outer$SiteLongitude,
                       lat = site_outer$SiteLatitude)

# Convert to sf object
points_sf <- st_as_sf(location, coords = c("lon", "lat"), crs = 4326)  # EPSG:4326 = WGS84


## function of extraction and spatial interpolation #####
extract_with_buffer <- function(raster_data, points_sf, buffer_radius = 1) {
  # Ensure CRS match
  if (!st_crs(raster_data) == st_crs(points_sf)) {
    points_sf <- st_transform(points_sf, st_crs(raster_data))
  }
  
  # Get raster resolution
  res_x <- res(raster_data)[1]
  res_y <- res(raster_data)[2]
  
  # Convert grid-based buffer to distance
  dist_buffer <- buffer_radius * max(res_x, res_y)
  
  # Create buffer around points
  buffered_points <- st_buffer(points_sf, dist = dist_buffer)
  
  # Extract mean value within buffer
  extracted_values <- extract(raster_data, buffered_points, fun = mean, na.rm = TRUE)
  
  # Return extracted values with point ID
  return(data.frame(ID = points_sf$ID, value = extracted_values))
}

# Function 2: Interpolate NAs using Inverse Distance Weighting (IDW)
interpolate_na_values <- function(raster_data, points_sf, extracted_data) {
  # Merge extracted values to point geometry
  points_sf$value <- extracted_data$value
  
  # Identify points with NA
  na_points <- points_sf[is.na(points_sf$value), ]
  non_na_points <- points_sf[!is.na(points_sf$value), ]
  
  if (nrow(na_points) == 0) {
    return(points_sf) # No NA to interpolate
  }
  
  # Convert to Spatial for gstat
  non_na_sp <- as_Spatial(non_na_points)
  na_sp <- as_Spatial(na_points)
  
  # Fit IDW model
  idw_model <- gstat::gstat(formula = value ~ 1, data = non_na_sp, nmax = 7, set = list(idp = 2.0))
  
  # Predict missing values
  interpolated <- predict(idw_model, newdata = na_sp)
  
  # Fill interpolated values back
  points_sf$value[is.na(points_sf$value)] <- interpolated$var1.pred
  
  return(points_sf)
}



## apply to site_outer #########
site_gridded <- data.frame(site_lon = site_outer$SiteLongitude,
                           site_lat = site_outer$SiteLatitude,
                           root_dep = NA,
                           olp = NA,
                           z_cost = NA,
                           ph = NA,
                           cn = NA)

extracted <- extract_with_buffer(root_depth, points_sf)
site_gridded$root_dep <- interpolate_na_values(root_depth, points_sf, extracted)

extracted <- extract_with_buffer(olp_wgs84, points_sf)
site_gridded$olp <- interpolate_na_values(olp_wgs84, points_sf, extracted)

extracted <- extract_with_buffer(z_cost, points_sf)
site_gridded$z_cost <- interpolate_na_values(z_cost, points_sf, extracted)

extracted <- extract_with_buffer(ph_map, points_sf)
site_gridded$ph <- interpolate_na_values(ph_map, points_sf, extracted)

extracted <- extract_with_buffer(cn_map, points_sf)
site_gridded$cn <- interpolate_na_values(cn_map, points_sf, extracted)

write.csv(site_gridded,"site_gridded_filled.csv", row.names = F)

# organize data
site_data <- read.csv(r"(C:\Users\shirl\OneDrive - Imperial College London\Postdoc_240501\ICCS_summer_school\FR_alltraits_sites_filled_BHPMF.csv)")
site_gridded_demo <- data.frame(SiteLatitude = site_gridded$site_lat,
                                SiteLongitude = site_gridded$site_lon,
                                olp = site_gridded$olp$value,
                                z_cost = site_gridded$z_cost$value,
                                ph = site_gridded$ph$value,
                                cn = site_gridded$cn$value)


site_full <- inner_join(site_gridded_demo, site_data, by = c("SiteLatitude", "SiteLongitude"))

write.csv(site_full,"site_full.csv", row.names = F)

