getwd()

setwd("C:/workspace_r")
getwd()

## Two Sample T test
## Non Parametric test 

twosample = read.csv("C:/workspace_r/R_Class/Session 4/Data/twosample.csv")
treatment = twosample[twosample$group == "Treatment",2]
control = twosample[twosample$group == "Control",2]
nt = length(treatment)
nc = length(control)

tstat = abs(mean(treatment)- mean(control))

## describe the population and the create synthetic sample 

f1 = function()
{
  x= c(treatment, control)
  x= sample(x)
  m1 = mean(x[1:nt])
  m2 = mean(x[(nt+1):(nt+nc)])
  return(m1-m2)
}


sdist = replicate(10000,f1())
plot(density(sdist))
gap = abs(mean(sdist)-tstat)
abline(v= mean(sdist)-gap,col="red")
abline(v=mean(sdist)+gap,col="red")
s1= sdist[(sdist<(mean(sdist)-gap)) | (sdist>(mean(sdist)+gap))]
pvalue = length(s1)/length(sdist)


## Hypothesis that median value of control group and treatment are same

tstat = abs(median(treatment)- median(control))

f1 = function()
{
  x= c(treatment, control)
  x= sample(x)
  m1 = median(x[1:nt])
  m2 = median(x[(nt+1):(nt+nc)])
  return(m1-m2)
}



tstat = abs(quantile(treatment,0.75)- quantile(control,0.75))

f1 = function()
{
  x= c(treatment, control)
  x= sample(x)
  m1 = quantile(x[1:nt],0.75)
  m2 = quantile(x[(nt+1):(nt+nc)],0.75)
  return(m1-m2)
}


## multivariate distribution
## Tstatistic for correlation of two variables 
admission <- read.csv("C:/workspace_r/R_Class/Session 4/Data/admission.csv")
GMAT = admission$GMAT
GPA = admission$GPA

install.packages("mvrnorm")
library(mvtnorm)

M =c(10,6)
S = matrix(c(4,2,2,5),nrow = 2,ncol= 2)

rmvnorm(n=50 , mean = M, sigma = S)

## Tstat 
tstat = cor(GPA,GMAT)

f1 = function()
{
  M = c(mean(GPA),mean(GMAT))
  
  S = matrix(c(var(GPA),0.6*sd(GPA)*sd(GMAT),0.6*sd(GPA)*sd(GMAT),var(GMAT)), nrow = 2, ncol = 2)
  x= rmvnorm(length(GMAT),mean = M, sigma = S )
  return(cor(x[,1],x[,2]))

}






sdist = replicate(10000,f1())
plot(density(sdist))
gap = abs(mean(sdist)-tstat)
abline(v= mean(sdist)-gap,col="red")
abline(v=mean(sdist)+gap,col="red")
s1= sdist[(sdist<(mean(sdist)-gap)) | (sdist>(mean(sdist)+gap))]
pvalue = length(s1)/length(sdist)



##############################################################################################################


## Statistical estimation

data1 = read.csv("C:/workspace_r/R_Class/Session 5/Data/data1.csv")

x1 = data1$x1
dnorm(x1,mean= 5,sd = 2)  ## probbability of finding points in x1 in the normal distribution 
## compute log likelihood in r 
## dnorm gives the probability of event in the distribution
sum(dnorm(x1,mean= 5,sd = 2,log =T)) ##this is the log likelihood function summatiion of logs of probabilities  
hist(dnorm(x1,mean= 6,sd = 2,log =T))
hist(dnorm(x1,mean= 6,sd = 2))


mseq = seq(0,10,by=0.05)
mseq

plot(mseq)
f1 = function(m)
{
  LL = sum(dnorm(x1,mean=m,sd=2,log=T))
  return(LL)
}

?dnorm


llres = sapply(mseq, f1)
plot(llres)

i= which.max(llres)

mseq[i]
?dnorm
plot(rpois(10000,.1))


