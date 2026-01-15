demander si on devrait ou pas "corriger" la façon dont on applique les coeffs à y quand on a plusieurs signaux concaténés
(code commenté dans le R:

  # if (length(p)>1)
  # {
  #   for (k in 2:length(p))
  #   {
  #     pif=round(seq(10,100,length.out = 2*length(int.coef))*p[k]/100)
  #
  #     sumX=matrix(0,n,length(int.coef))
  #     for (i in 1:length(int.coef))
  #     {
  #       sumX[,i]=apply(X[,(sum(p[1:(k-1)])+pif[i]):(sum(p[1:(k-1)])+pif[i+1])],1,function(u) sum(u))
  #     }
  #     y0=y0+sumX%*%int.coef
  #
  #   }
  # }
  
)


Aussi demander à vérifier la génération du spectre