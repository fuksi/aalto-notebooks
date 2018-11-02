setwd("C:\\Repos\\aalto-notebooks\\timeseries\\atk1")
getwd()

# load data
emis=read.table("emissions.txt", header=T, sep="\t")
# fixate randomness in the program, in order the reproduce the same result later
set.seed(123)

# a)
# summary of the data frame: 1st qu. ~ the value of which 25% population is below, similar to 3rd qu.
summary(emis)

# plotting the history gram of nox
# notice emis[2] return information about a column in a context of data frame 
# emis[,2] return the values for that column (which is then a vector of values)
# in most case we want values
# or just emis[, "NOx"]
hist(emis[, "NOx"])

# generate correlation matrix
cor(emis)

# b) create the regresson lineal model
# syntax model <- lm(res~ explan1 + explan2 + ...)
# the syntax is to create a linear model (using fn lm), the left of ~ is the response var, the right are explanatory vars
# obviously the name of the variables should match the columns of the data
fit1 <- lm(NOx~Humidity+Pressure+Temp, data=emis)
summary(fit1)
