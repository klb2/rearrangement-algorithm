library("qrmtools")
method <- "best.VaR"
level <- 0.1
num <- 10
qF = rep(list(qexp), 5)
results <- qrmtools::RA(level, qF, num, method=method)
print(results)
