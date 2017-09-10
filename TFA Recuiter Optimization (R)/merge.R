
install.packages("xlsx")
library(xlsx)
library(dplyr)
library(data.table)

library(help=dplyr)
setwd("c:/workspace_r/competition/")

undergrad2016 = read.xlsx2("Undergrad2016.xlsx",1)
university = read.xlsx2("University.xlsx",1)
colnames(jointable_1617)
levels(jointable2016$Major.1)
colnames(undergrad2017)
jointable2016 = merge(undergrad2016,university, by.x = "Undergrad.University.ID",by.y = "University.ID" )
jointable2017 = merge(undergrad2017,university, by.x = "Undergrad.University.ID",by.y = "University.ID")
jointable_1617 = merge(undergrad2016,undergrad2017, by = "Undergrad.University.ID")
jointable2016[,]
names()
write.csv(jointable2016,"join2016.csv")
write.csv(jointable2017,"join2017.csv")

undergrad2017 = read.xlsx2("Undergrad2017.xlsx",1)
View(jointable)
names
new_join2016 = data.table()

