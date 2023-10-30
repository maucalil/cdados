# rvencio 2023-set-29
	# função normal multivariada ou gaussiana multidimensional
	# x e m vetor coluna k linhas 1 coluna
	# S é matrix k por k, onde k é a dimensão do problema
	f = function(x, m, S){
		k = ncol(S)
		dentro = -1/2 * t(x-m) %*% solve(S) %*% (x-m)
		out = (2*pi)^(-k/2) * 1/sqrt(det(S)) * exp( dentro )
		return(out)
	}
	
	x = matrix( c(0.5, 0.5), ncol=1)
	mu = matrix( c(0, 0), ncol=1)
	Sigma = matrix( c(1, 3/5, 3/5, 2), ncol=2, nrow=2)
	f(x, m = mu, S = Sigma)
	
	N = 100
	Z = matrix(0, ncol=N, nrow=N)
	x1 = x2 = seq(-3, 3, len=N) # x1 = x2 porque eu quero quadrado
	for(i in 1:N){
		for(j in 1:N){
			x = c(x1[i], x2[j]) # é o ponto do grid da vez
			Z[i,j] = f( x , m = mu, S = Sigma)
		}
	}
	
	image(x1, x2, Z)
	
	
	
	
	
