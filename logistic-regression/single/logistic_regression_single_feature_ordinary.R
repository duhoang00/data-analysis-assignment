#Load Library
library(MASS)
library(foreign)
library(ggplot2)
library(reshape2)
library(Hmisc)

#Read Data
dat <- read.dta("https://stats.idre.ucla.edu/stat/data/ologit.dta")

#View Data
head(dat)

#Descriptive statistics of these variables
lapply(dat[, c("apply", "pared", "public")], table)
ftable(xtabs(~ public + apply + pared, data = dat))
summary(dat$gpa)
sd(dat$gpa)

#Plot all of the marginal relationships
ggplot(dat, aes(x = apply, y = gpa)) +
  geom_boxplot(size = .75) +
  geom_jitter(alpha = .5) +
  facet_grid(pared ~ public, margins = TRUE) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))
#Fit ordered logit model and store results 'm'
m <- polr(apply ~ pared + public + gpa, data = dat, Hess=TRUE)

#View a summary of the model
summary(m)

#Store table
(ctable <- coef(summary(m)))

#Calculate and store p values
p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2

#Combined table
(ctable <- cbind(ctable, "p value" = p))

#Default method gives profiled CIs
(ci <- confint(m))

#CIs assuming normality
confint.default(m)

#Odds ratios
exp(coef(m))

#OR and CI
exp(cbind(OR = coef(m), ci))

#The summary function
sf <- function(y) {
  c('Y>=1' = qlogis(mean(y >= 1)),
    'Y>=2' = qlogis(mean(y >= 2)),
    'Y>=3' = qlogis(mean(y >= 3)))
}

(s <- with(dat, summary(as.numeric(apply) ~ pared + public + gpa, fun=sf)))
#
glm(I(as.numeric(apply) >= 2) ~ pared, family="binomial", data = dat)
#
glm(I(as.numeric(apply) >= 3) ~ pared, family="binomial", data = dat)

#Print
s[, 4] <- s[, 4] - s[, 3]
s[, 3] <- s[, 3] - s[, 3]
s 
plot(s, which=1:3, pch=1:3, xlab='logit', main=' ', xlim=range(s[,3:4]))
#
newdat <- data.frame(
  pared = rep(0:1, 200),
  public = rep(0:1, each = 200),
  gpa = rep(seq(from = 1.9, to = 4, length.out = 100), 4))

newdat <- cbind(newdat, predict(m, newdat, type = "probs"))

#show first few rows
head(newdat)
#
lnewdat <- melt(newdat, id.vars = c("pared", "public", "gpa"),
  variable.name = "Level", value.name="Probability")
  
#view first few rows
head(lnewdat)
#
ggplot(lnewdat, aes(x = gpa, y = Probability, colour = Level)) +
  geom_line() + facet_grid(pared ~ public, labeller="label_both")

