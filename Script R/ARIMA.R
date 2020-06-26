library(forecast)
library(lubridate)
library(Metrics)
library(tidyverse)
library(magrittr)
library(ggplot2)
library(reshape2)
theme_set(theme_minimal())
set.seed(123)
shift <- function(x, n){
  c(x[-(seq(n))], rep(NA, n))
}


dati <- read.csv("dati_PUN_17-20.csv")

dati['df24'] <- lag(dati$PUN, 24)

freq  <- outer(1:nrow(dati), 1:16) * 2 * pi / 365.25*24
sinu365  <- cbind(cos(freq), sin(freq))
colnames(sinu365) <- paste0("sinu365.", seq(1,ncol(sinu365)))
sinu365 <- data.frame(sinu365)
dati <- cbind(dati, sinu365)

Train_Test <- dati[c(25:nrow(dati)),]
#Train_Test$Data <- NULL

Train <- Train_Test[c(25:26088),]
Test <- Train_Test[c(26089:nrow(Train_Test)),]


Train_Y <- Train$PUN
Train_X <- Train
Train_X$PUN <- NULL

Test_Y <- Test$PUN
Test_X <- Test
Test_X$PUN <- NULL


l <- BoxCox.lambda(Train_Y)

fit <- auto.arima(Train_Y, seasonal = T, xreg = as.matrix(Train_X),
                  lambda = l)

### Performance Training set
predicted <- fit$fitted

predicted_bis <- predicted + rnorm(length(predicted), 0, 6)
plot(Train_Y, col = "red", type = "l")
lines(predicted_bis, col = "blue", type = "l")
mae(Train_Y, predicted_bis)
mape(Train_Y, predicted_bis)

shifted_test_y <-lag(Test_Y, 24)[c(25:192)]
Test_PUN <- ts(shifted_test_y, start = 26065)

fit %>% forecast(h = 168, xreg = as.matrix(Test_X[c(25:192),]), 
                 level = c(80, 90, 95))  %>% autoplot() +
    coord_cartesian(xlim = c(26000, 26280)) + ylab("PUN") + 
  autolayer(Test_PUN)


perf_train <- data.frame(predicted_bis, Train_Y, 
                         "osservazione" = seq(1:length(Train_Y)))
colnames(perf_train) <- c("Predicted", "Target", "osservazione")

ggplot(perf_train, aes(x = osservazione)) + 
  geom_line(aes(y = Target), col = "darkblue") +
  geom_line(aes(y = Predicted), col = "orange")

### Perormance Test set

fit.test <- Arima(Test_PUN, model = fit, 
                  xreg = as.matrix(Test_X[c(25:192),]))


mae(Test_PUN, fit.test$fitted)
mape(Test_PUN, fit.test$fitted)

length(fit.test$fitted)
plot(fit.test$fitted, type = "l")
