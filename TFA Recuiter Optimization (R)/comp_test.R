library(sp)
library(rgdal)
setwd("c:/workspace_r/competition")


#import zips shapefile and transform CRS 

?readOGR
zips <- readOGR("zipcode/cb_2015_us_zcta510_500k.shp")
zips <- spTransform(zips, CRS("+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0"))

View(zips)


train_dt = fromJSON("train.json")
#names(train_data)
vars <- setdiff(names(train_dt), c("photos", "features"))
train_data <- map_at(train_dt, vars, unlist) %>% tibble::as_tibble(.)

train_latlon = as.data.frame(train_data[,c("longitude","latitude")])
colnames(train_latlon) = c("lon", "lat")

head(zips)
?spTransform
class(zips)
str(zips)
#here is a sample with three cities in New York State and their coordinates      
df <- as.data.frame(matrix(nrow = 3, ncol =3))
colnames(df) <- c("lat", "lon", "city")

df$lon <- c(43.0481, 43.1610, 42.8864)
df$lat <- c(-76.1474, -77.6109,-78.8784)
df$city <- c("Syracuse", "Rochester", "Buffalo")

df
lat     lon      city
1 -76.1474 43.0481  Syracuse
2 -77.6109 43.1610 Rochester
3 -78.8784 42.8864   Buffalo

#extract only the lon/lat                   
class(xy) <- df[,c(1,2)]

#transform coordinates into a SpatialPointsDataFrame
spdf <- SpatialPointsDataFrame(coords = train_latlon, data = train_latlon, proj4string = CRS("+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0"))

#subset only the zipcodes in which points are found
zips_subset <- zips[spdf, ]
View(zips_subset)

#NOTE: the column in zips_subset containing zipcodes is ZCTA5CE10
#use over() to overlay points in polygons and then add that to the original dataframe

train_latlon$zip <- over(spdf, zips_subset[,"ZCTA5CE10"])