library(EnvStats)
N = 1000
x_min = 11.3
alpha = 2.5
data = rpareto(N, x_min, alpha)
write.table(data, "dados_pareto", row.names = FALSE, col.names = FALSE)
