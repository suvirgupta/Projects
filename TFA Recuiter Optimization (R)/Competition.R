
my.install <- function(pkg) {
  if (!(pkg %in% installed.packages()[,1])) {
    install.packages(pkg)
  }
  return (require(pkg,character.only=TRUE))
}

my.install("rgdal")
my.install("ggmap")
my.install("zipcode")

setwd("c:/workspace_r/competition")
library(dplyr)
library(purrr)
library(jsonlite)
library(data.table)
library(tidyr)
library(rgdal)
library(ggplot2)
library(ggmap)
library(zipcode)
library(dplyr)
#library(MASS)
library(class)


data(zipcode)   # load Zipcode data from the library

#training = fromJSON("train.json",flatten = T)
#train = as.data.table(training)

train_dt = fromJSON("train.json")
#names(train_data)
vars <- setdiff(names(train_dt), c("photos", "features"))
train_data <- map_at(train_dt, vars, unlist) %>% tibble::as_tibble(.)
str(train_data)


#zip = mutate(zipcode , pin =paste(zipcode$latitude,zipcode$longitude,sep = " "))
#pincode = zip["pin"==latlong,]
#longlat = as.factor(paste(train_data$longitude,train_data$latitude, sep = ","))
train_data$street_address[1]

#loglat=as.numeric(geocode(train_data$street_address[1]))    # api limited to 2500 queries per day 
#zip2 = revgeocode(loglat, output = "more") # serves only one zipcode per query
#lag

#View(zipcode)
#class(zipcode[,1])
zipcode_latlon = zipcode[zipcode$state=="NY",c(1,4,5)]
train_latlon = train_data[,c("latitude","longitude")]


zip1 = rep(10007, nrow(train_latlon))
zip1 = as.character(zip1)


train_latlon = cbind(zip1, train_latlon)
colnames(train_latlon) = c("zip","latitude","longitude")

knn_fit = knn(zipcode_latlon, train_latlon,zipcode_latlon$zip, k=1)
knn_fit[10000:12000]
View(train_data)

knn_fit
train_data[,'pincode']
train_data = mutate(train_data, pincode = knn_fit)
train_data = mutate(train_data, Price_segment= cut(train_data$price, c(0,2000,6000,Inf)))
train_data = mutate(train_data, bedroom_segment = cut(train_data$bedrooms,c(0,1,2,3,Inf) , include.lowest = T))
train_data = mutate(train_data,bathroom_segment = cut(train_data$bathrooms,c(0,1,1.5,2,Inf), include.lowest = T))
names(train_data)


plot(as.factor(train_data$pincode), as.factor(train_data$interest_level))
levels(train_data$pincode)
head(train_data)
train_data$bathrooms
x=levels(train_data$bedroom_segment)
y= levels(train_data$bathroom_segment)
i=1
j=1


opar=par()
par(bg = "White", col="red", las=2,mfrow=c(4,4))
for( i in length(x))
{
  for(j in length(y))
  {
  z =  train_data[(train_data$bedroom_segment==x[i]) & (train_data$bathroom_segment==y[j]),]
    seg_plot(z[,1:15])
  }
  
  
}
data_seg = train_data %>% group_by(bedroom_segment,bathroom_segment) %>% nest()
data_seg

par(opar)

train_data[,]
table(bedroom_segment,bathroom_segment)
class(train_data$bathrooms)
?paste
newyork_map <- get_map(location = "New York", maptype = "satellite", zoom = 11)
seg_plot = function(plot_data)
{
#range(train_data$price)
ggmap(newyork_map, extent = "device") + geom_point(aes(x = longitude, y = latitude ,colour = interest_level), 
                                                 data = data_seg$data[[1]]) 

ggmap(newyork_map, extent = "device") + geom_point(aes(x = longitude, y = latitude ,colour = Price_segment), 
                                                 data = data_seg$data[[4]]) 

ggmap(newyork_map, extent = "device") + geom_point(aes(x = longitude, y = latitude , label = zip), 
                                                 data = zipcode_latlon) 

}
table(as.factor(train_data$bathrooms),as.factor(train_data$bedrooms))

library(help= rgdal)
countries <- readOGR("nybb.ship",layer = "nybb")

sum(is.na(train_data[,c(8,10)]))

?readOGR

train1 = train[,c(-7,-12)]
str(train1)
View(train)
#unclass(train)
train2=train[c(-1043,-2084),]
plot(train$bedrooms, train$price, ylim= c(1000,20000))
plot(train$bathrooms,train$price, col="red", pch=19, cex = 1 , ylim= c(1000,20000) )
len = sapply(train$features, length)
plot(lengthtrain)
plot(len,train$price, ylim = c(1000,20000))
fwrite(train, file = "training.csv")
train_data$interest_level<-as.factor(train_data$interest_level)
levels(train_data$interest_level)
names(train_data)
a= train_data[,c(1,2)]
symbols(train_data$price, rectangles =a, bg = "blue")





