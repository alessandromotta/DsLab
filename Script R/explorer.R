
library(ggplot2)
library(ggfortify)
library(fpp3)
library(forecast)
library(ggplot2)
library(ggfortify)
library(tidyverse)
library(lubriGroup.1)
library(xts)
library(dplyr)
# osserviamo il dataset completo per vedere se si denotano cicli o trend nel long term. 
ggplot(data=dati_PUN_completi, aes(x=Data, y=PUN, group=1)) +
  geom_line()


# la serie storica on presenta cicli o trend evidenti nel lungo termine, l'andamento pare alquando lineare.
# a prima vista parrebbe che la volatilità sia in diminuizione nell'ultimo periodo.



dati2 <- aggregate(dati_PUN_completi$PUN, by=list(dati_PUN_completi$Data), mean)
str(dati2)


########################################################################
dati2 <- mutate(dati2, MonthYear = paste(year(Group.1),formatC(month(Group.1), width = 2, flag = "0")))

# Day of the week
dati2 <- mutate(dati2, Yearday = paste(year(Group.1), formatC(month(Group.1), width = 2, flag = "0"),
                                                     formatC(day(Group.1), width = 2, flag = "0")))

# Week of the year
dati2 <- mutate(dati2, Week = week(Group.1))

dati2 <- mutate(dati2, Year = year(Group.1))
dati2$Year <- as.factor(dati2$Year)

str(dati2)

pun_month <- aggregate(dati2$x, by = list(dati2$MonthYear), FUN = function(x) mean(x, na.rm=T))

myts <- ts(pun_month$x, frequency=12, start = c(2004, 04), end = c(2020, 04))

plot(myts)

########################################################################à

myds_month <- decompose(myts)
plot(myds_month)


myholtts <-HoltWinters(myts)

myhw <- forecast(myholtts, h = 24, findfrequency = TRUE)
plot(myhw)


autoplot(myts) + ggtitle("Melbourne temperatures time series") + 
  xlab("Time (year)") + ylab("Temperatures (C)") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) 

ggseasonplot(myts, year.labels=TRUE, year.labels.left=TRUE) +
  ylab("$ million") +
  ggtitle("Seasonal plot: antidiabetic drug sales")


ggseasonplot(myts, polar=TRUE) +
  ylab("$ million") +
  ggtitle("Polar seasonal plot: antidiabetic drug sales")


ggsubseriesplot(myts) +
  ylab("$ million") +
  ggtitle("Seasonal subseries plot: antidiabetic drug sales")

myts2 <- window(myts, start=2005, frequency = 12)
gglagplot(myts2)

ggAcf(myts2,lag = 192)




##################################################à

dati_PUN_17_20 <- mutate(dati_PUN_17_20, MonthYear = paste(year(Data),formatC(month(Data), width = 2, flag = "0")))

# Day of the week
dati_PUN_17_20 <- mutate(dati_PUN_17_20, Yearday = paste(year(Data), formatC(month(Data), width = 2, flag = "0"),
                                       formatC(day(Data), width = 2, flag = "0")))

# hours of the day
dati_PUN_17_20 <- mutate(dati_PUN_17_20, dayhour = paste(year(Data), formatC(month(Data), width = 2, flag = "0"),
                                                         formatC(day(Data), width = 2, flag = "0"), formatC(hour(Data), width = 2, flag = "0" )))


# Week of the year
dati_PUN_17_20 <- mutate(dati_PUN_17_20, Week = week(Data))


# year
dati_PUN_17_20 <- mutate(dati_PUN_17_20, Year = year(Data))
dati_PUN_17_20$Year <- as.factor(dati_PUN_17_20$Year)

str(dati_PUN_17_20)

pun_month <- aggregate(dati_PUN_17_20$PUN, by = list(dati_PUN_17_20$MonthYear), FUN = function(x) mean(x, na.rm=T))


myts <- ts(pun_month$x, frequency=12, start = c(2017, 01), end = c(2020, 04))

plot(myts)

########################################################################à

myds_month <- decompose(myts)
plot(myds_month)


myholtts <-HoltWinters(myts)

myhw <- forecast(myholtts, h = 24, findfrequency = TRUE)
plot(myhw)


autoplot(myts) + ggtitle("Melbourne temperatures time series") + 
  xlab("Time (year)") + ylab("Temperatures (C)") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) 

ggseasonplot(myts, year.labels=TRUE, year.labels.left=TRUE) +
  ylab("$ million") +
  ggtitle("Seasonal plot: antidiabetic drug sales")


ggseasonplot(myts, polar=TRUE) +
  ylab("$ million") +
  ggtitle("Polar seasonal plot: antidiabetic drug sales")


ggsubseriesplot(myts) +
  ylab("$ million") +
  ggtitle("Seasonal subseries plot: antidiabetic drug sales")

myts2 <- window(myts, start=2017, frequency = 24)
gglagplot(myts2)

ggAcf(myts2,lag = 26280)
