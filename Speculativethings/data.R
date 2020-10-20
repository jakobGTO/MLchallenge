data <- read.csv2("irisX.txt", sep = ",", head = FALSE)

for (i in 1:(ncol(data))) {
    data[, i] <- as.numeric(data[, i])
}


index <- data[, 1] > 10
data <- data[index, ]
