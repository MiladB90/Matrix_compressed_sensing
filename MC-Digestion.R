setwd("/Users/awd/Projects/MultiverseExperiments/MatrixCompletion")
read.csv(file="mc-0003.csv",header=T) -> mc
# modulo noise cosL~ cosR so we take geometric mean
sinR = sqrt(1-mc$cosL*mc$cosR);
un    <- sort(unique(mc$n))
# only work with largest problem size for now
sel   <- mc$n == 500
mcs   <- mc[sel,];
sinRs <- sinR[sel]; 
mcs$sinR <- sinRs
# would-be prediction formula
pred.sinR <- function(p,snr) { pmin(1/(sqrt(p)*snr),1) }
mcs$pred <- with(mcs,pred.sinR(p,snr));
# make plots
library(ggplot2)
ggplot(mcs,aes(x=pred,sinR,col=p))+geom_point()
ggplot(mcs,aes(x=pred,sinR,group=p))+geom_point(aes(color=p))
ggplot(mcs,aes(x=pred,sinR,group=snr))+geom_point(aes(color=snr))
pairs(mcs[,c("sinR","pred","p","snr")])
pairs(mcs[,c("sinR","pred","p","snr","n")])
#
# validate that prediction slope approx 1 in each subgroup windowed by p
#
par(mfrow=c(3,3)); 
for( pee_lo in seq(.1,.9,length=9)){ 
  subsel <- (mcs$p >= pee_lo) & (mcs$p < (pee_lo + 0.1))
  if(any(subsel)){
    with(mcs[subsel,],
         { F <- pred.sinR(p,snr)
           co <- lm(sinR ~ F,subset=F<0.9)$coefficients;
           plot(F,sinR,xlab = "pred",
                main=sprintf("p~%0.3f:slope~%0.3f",pee_lo+.05,co[2]))
           })
  }
}


