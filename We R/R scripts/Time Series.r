# installing all the required packages mentioned below
# xts package installation
install.packages("xts")
# forecast package installation
install.packages("forecast")

# loading data.table package
library(data.table)
# loading xts package
library(xts)

# loading the input data present in DJIA_table.csv file
DJIA_table <- fread("R:/Study Material/R/Project/SE-Pred/DJIA_table.csv")

# for performance improvements
DJIA_table <- setDT(DJIA_table)
DJIA_table <- setkey(DJIA_table,Date)
DJIA_table <- xts(DJIA_table[,-1],order.by = as.Date(DJIA_table$Date,"%Y-%m-%d"))

# getting the structure of this table
str(DJIA_table)
# plotting this table information
plot(DJIA_table[,6])

cycle(DJIA_table)
# getting plots related to time series analysis
x = ts(DJIA_table[,6],frequency = 12, start = 2008  )
plot(as.xts(x), major.format = "%Y")
plot(aggregate(DJIA_table[,6],mean,by= week))


month <- function(x)format(x, '%Y-%w')

aggregate(DJIA_table[,6],mean,by=month)
#boxplot(DJIA_table~cycle(DJIA_table))
install.packages("tseries")
library(tseries)


## first difference plot
plot(diff(DJIA_table[,6]))
## first diff with log transformation plot
plot(log(diff(DJIA_table[,6])))

adf.test(diff(as.ts(DJIA_table[,6])), alternative="stationary", k=0)

## ACF Plots
#1. ACF with differencing
acf(diff(as.ts(DJIA_table[,6])))

#2. ACF with log differencing
#acf(log(diff(as.ts(DJIA_table[,6])))) ## not much difference from the previous one

## PACF for the plot 

pacf(diff(as.ts(DJIA_table[,6])))
#pacf(log(diff(as.ts(DJIA_table[,6]))))


library(forecast)
par(mfrow=c(2,2))
fit <- auto.arima(DJIA_table[,6])

# plotting the summary of the forecasting model fit
summary(fit)
plot(fit)
plot(fit$residuals,xlab = " Residual PLot")
hist(fit$residuals , xlab = "Histogram : Residual" , main = "Residual Histogram")
plot(fit$fitted,ylab = "Fitted value",main = "Fitted Graph")

# residual fit plot of the analysis
Acf(residuals(fit))

# plot of ARIMA forecasting
plot(forecast(fit), main = "ARIMA Forecasting")
