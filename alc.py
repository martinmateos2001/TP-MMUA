import numpy as np
# Librería de Funciones.
def inversa(A):
  L,U,cantop = calculaLU(A)
  detU = U[0,0]
  for i in range(1,U.shape[1]):
    detU = detU * U[i,i]
  if detU == 0:
    return None

  Ainv = np.zeros((A.shape[0],A.shape[1]))
  I = np.eye(A.shape[0])
  for j in range(0,A.shape[1]):
    e_j = I[:,j] # estoy tomando todas las filas de la columna j
    y = res_tri(L,e_j)
    x = res_tri(U,y,inferior=False)
    Ainv[:,j] = x
  return Ainv

def res_tri(L,b,inferior=True):
  if inferior == False:
    y = np.zeros(b.shape)
    for i in range(L.shape[0]-1,-1,-1):
      sumatoria = 0
      for j in range(i+1,L.shape[0]):
        sumatoria = sumatoria + L[i,j] * y[j]
      y[i] = (b[i] - sumatoria) / L[i,i]
  else:
    y = np.zeros(b.shape)
    for i in range(0,L.shape[0]):
      sumatoria = 0
      for j in range(0,i):
        sumatoria = sumatoria + L[i,j] * y[j]
      y[i] = (b[i]-sumatoria) / L[i,i]
  return y

def calculaLU(A):
  cant_op = 0
  if A is None:
    return None,None,0
  n=A.shape[0]
  m=A.shape[1]
  Ac = A.copy()
  if n!=m:
    return None,None,0

  for k in range(0,n-1): # pivote
      if Ac[k,k] == 0:
        return None,None,0
      for i in range(k+1,n): # Aca digo que los elementos abajo del pivote, pues voy a modificar todas las filas de abajo aca.
        m = Ac[i,k]/Ac[k,k] # numero por el cual tengo que multiplicar para que quede 0 en el espacio de la fila.
        Ac[i,k] = m # seria mi numerito en rojo
        cant_op+=1 # division
        for j in range(k+1,n): # Ahora me falta modificar toda la otra parte de la fila.
          Ac[i,j] = Ac[i,j] - m*Ac[k,j] # para cada fila i modifico los elementos restandole la fila pivote multiplicada por mi m. osea lo que esta a la derecha de mi numero rojito.
          cant_op+=2 # multiplicacion por m y resta

  U = np.zeros((n,n))
  L = np.eye(n)
  for i in range(0,n):
    for j in range(0,n):
        if i<=j: # matriz superior
          U[i,j] = Ac[i,j]
        elif i>j: # matriz inferior
          L[i,j] = Ac[i,j]

  return L, U, cant_op

def QR_con_HH(A,tol=1e-12,extras=False):
  #  A una matriz de m x n (m>=n)
  m , n= A.shape[0] , A.shape[1]
  if m>=n:
    R = A.copy()
    Q = np.eye(m)
    listamatricesQ = []
    listamatricesR = []
    for k in range(0,n):
      x = R[k:m,k].copy()
      norma = normavector2(x)
      signo = 1 if x[0] >=0 else -1
      alfa = -signo * norma
      u = x.copy()
      u[0] = u[0] - alfa
      norma_u = normavector2(u)

      if norma_u>tol:
        u = u/norma_u
        u_col = u.reshape(-1,1)
        H_k = np.eye(m - k) - 2 * multMat(u_col, u_col.T)

        Htilde_k = np.eye(m)
        Htilde_k[k:m,k:m]= H_k

        R = multMat(Htilde_k,R)
        for i in range(0,n):
          for j in range(0,n):
            if abs(R[i,j]) < tol:
              R[i,j] = 0
        Q = multMat(Q,Htilde_k.T)
        listamatricesR.append(R)
        listamatricesQ.append(Q)
    if extras == True:
      extra_info = {'R_matrices': listamatricesR , 'Q_matrices' : listamatricesQ}
      return Q,R, extra_info
    return Q,R
  else:
    return None

def normaExacta(A, p=[1,'inf']):
    filas = A.shape[0]
    cols = A.shape[1]
    res = 0
    # Calculo la norma 1
    if(p == 1 and isinstance(p, (int, float))):
        for i in range(cols):
            s = 0
            for f in A:
                s = s + abs(f[i])
            if(s > res):
                res = s
        return res
    # calculo la norma infinito
    if(p=='inf' and isinstance(p, str)):
        for j in range(filas):
            s = 0
            for e in A[j]:
                s = s + abs(e)
            if(s > res):
                res = s
        return res
    if(isinstance(p, (tuple, list))):
        res = [normaExacta(A, p[0]), normaExacta(A,p[1])]
        return res
    return None

def multMat(A,B):
  if A.ndim == 1:
      A = A.reshape(1,-1)
  if B.ndim == 1:
      B = B.reshape(-1,1)
  if A.shape[1] == B.shape[0]:
    C = np.zeros((A.shape[0],B.shape[1]))
    for i in range(0,A.shape[0]):
      for j in range(0,B.shape[1]):
        suma = 0
        for k in range(0,A.shape[1]):
          suma = suma + A[i,k] * B[k,j]
        C[i,j] = suma
    return C

def productoPunto(v,w):
  if len(v) == len(w):
    suma = 0
    for i in range(0,len(v)):
      suma = suma + v[i]*w[i]
    return suma

def normavector2(x):
  norma = 0
  for i in range(0,len(x)):
    norma = norma + x[i]**2
  norma = norma ** (1/2)
  return norma

def normalizar(x):
  norma = normavector2(x)
  x = x / norma
  return x

def metpot2k(A,tol=1e-15,kmax=1000):
  if A.shape[0] == A.shape[1]:
    cantops = 0
    n = A.shape[0]
    if np.all(np.abs(A) < 1e-14):
      autovector = np.zeros(n)
      autovector[0] = 1.0
      return autovector, 0.0, 0

    v = np.random.rand(n)
    vtilde = normalizar(multMat(A,normalizar(multMat(A,v))))

    e = productoPunto(vtilde,v)
    if normavector2(vtilde) < 1e-14:
      autovector = np.zeros(n)
      autovector[0] = 1.0
      return autovector, 0.0, 0
    k = 0
    while abs(e-1) > tol and k<kmax:
      v = vtilde
      vtilde = normalizar(multMat(A,normalizar(multMat(A,v))))
      e = productoPunto(vtilde,v)
      k+=1
    autovector = vtilde
    autovalor = productoPunto(vtilde,multMat(A,vtilde))
    error = e-1
    return autovector,autovalor,k

def diagRH(A, tol=1e-14, K=1000): # retorna S, D tq A = S @ D @ S.T
    if A.shape[0] == A.shape[1]:
        n = A.shape[0]
        v1, l1, _ = metpot2k(A)

        # 3. Reflexión de Householder estable (evita u = e1 - v1 que produce 0/0)
        u = v1.copy()
        if u[0] >= 0:
            u[0] += 1.0
        else:
            u[0] -= 1.0

        norma_u = normavector2(u)
        if norma_u < 1e-14:
            Hv1 = np.eye(n)
        else:
            u = u / norma_u
            u_col = u.reshape(-1, 1) if u.ndim == 1 else u
            Hv1 = np.eye(n) - 2 * multMat(u_col, u_col.T)

        if n == 2:
            S = Hv1
            D = multMat(Hv1, multMat(A, Hv1.T))
        else:
            B = multMat(Hv1, multMat(A, Hv1.T))
            Atilde = B[1:n, 1:n]
            Stilde, Dtilde = diagRH(Atilde)

            D = np.zeros((n, n))
            Schica = np.zeros((n, n))

            D[0, 0] = float(l1)
            D[1:, 1:] = Dtilde

            Schica[0, 0] = 1.0
            Schica[1:, 1:] = Stilde

            S = multMat(Hv1, Schica)

        return S, D
    
def svd_reducida(A, k="max", tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k)
    """
    m, n = A.shape
    se_traspuso = False
    # Paso (iii): Si m < n, trabajamos sobre A^T para procesar una matriz mas chica y evitar ceros de mas
    if m < n:
        A = A.T
        m, n = A.shape
        se_traspuso = True

    AtA = multMat(A.T, A)
    S, D = diagRH(AtA) # D tiene los autovalores y V los autovectores
    V = S
    autovalores = np.zeros(n)
    for i in range(0,n):
      autovalores[i] = D[i,i]
    # ahora debo ordenar autovalores y autovectores de mayor a menor
    for i in range(0,len(autovalores)):
      idmax = i
      for j in range(i+1,n):
        if autovalores[j]>autovalores[idmax]:
          idmax = j
      autovalores[i], autovalores[idmax] = autovalores[idmax], autovalores[i] # cambio de lugar el maximo y mi indice de ahora
      tempautovector = V[:,i].copy()
      V[:,i] = V[:,idmax]
      V[:,idmax] = tempautovector
    # ya tengo ordenados todos los autovalores y autovectores de mayor a menor toca filtrar los valores singulares con la tolerancia
    tol_autovalor = max(tol**2, 1e-12)

    singulares = []
    autovectores = []
    for i in range(len(autovalores)):
      valor = autovalores[i]
      if valor > tol_autovalor:
        valorsing = np.sqrt(valor)
        if valorsing>tol: # miro si el valor singular es mayor a la tolerancia, sino es 0
          singulares.append(valorsing)
          autovectores.append(V[:,i])
    rango = len(singulares) # Guardo el rango maximo de la matriz viendo cuantos valores singulares tiene
    if k =='max' or k>rango: # si pide rango maximo o mas(osea piden k= 13 y la matriz rango 12) tomo kretenido = rango de la matriz
      kretenido = rango
    else:
      kretenido = int(k)
    if kretenido ==0: # si piden rango 0
      hatU = np.zeros((m, 0))
      hatSig = np.array([])
      hatV = np.zeros((n, 0))
      return hatU, hatSig, hatV # basicamente devuelvo todo vacio
    # Ahora caso si k>0
    hatSig = np.array(singulares[:kretenido]) # es una lista pues me pedian eso para mi funcion en este caso
    hatV = np.zeros((n,kretenido)) # creo matriz nxk
    for i in range(0,kretenido):
      hatV[:,i] = autovectores[i] # copio los autovectores columna para V

    # me falta crear U, lo voy a hacer con la formula de que Ui = 1/sigmai * Avi
    # Puedo calcular mas facil si guardo todo Avi de una en una matriz B
    B = multMat(A,hatV) # m*k dimensiones
    hatU = np.zeros((m,kretenido))
    for j in range(kretenido):
      hatU[:,j] = B[:,j] / hatSig[j]

    if se_traspuso:
        hatU, hatV = hatV, hatU
    return hatU, hatSig, hatV
def puntosfotoi(i,uv):
  puntos = []
  for j in range(0,uv.shape[1]-1,2):
    puntos.append(uv[i,j:j+2])
    uvp = np.array(puntos)
  return uvp

# como tengo 12 puntos, mi Q sera de 2*12x12 osea 24x12
# ahora quiero convertir mis P a Pi= [U,V,W,1]
def convertirP(pcorners):
  pi = []
  for i in range(0,pcorners.shape[0]):
    pi.append(pcorners[i].tolist() + [1])
    Puntosi = np.array(pi)
  return Puntosi

def formarQ(P,uvpuntos):
  Q = np.zeros((24,12))
  for i in range(0,12):
    Q[2*i,0:4] = P[i]
    Q[2*i,8:12] = -uvpuntos[i][0]*P[i]
    Q[2*i+1,4:8] = P[i]
    Q[2*i+1,8:12] = -uvpuntos[i][1]*P[i]
  return Q

def devolverMatricesCalibracion3d(M,uigualv = False, k12igual0 = False):
  # M = resolverDLT3D(Q) se hace directamente desde M
  A = M[:,0:3].copy()
  b = M[:,3].copy()
  Ainv = inversa(A)
  Qinv, Rinv = QR_con_HH(Ainv)
  K = inversa(Rinv)
  R = Qinv.T
  Knormalizada = K/K[2,2]
  t = multMat(inversa(K),b.reshape(-1,1))
  t = t.reshape(1,-1)
  if uigualv == True:
    Knormalizada[1,1] = Knormalizada[0,0]
  if k12igual0 == True:
    Knormalizada[0,1] = 0
  # se supone que si el det(R) es -1 entonces tengo que multiplicar t y R por -1 pues me queda espejado por householder, pero habria que preguntar. tampoco tengo funcion de determinante
  return R , t, Knormalizada

# 2b resolverDLT3D(Q) encuentra la matriz M a partir de Q, por 1b el vector m es el menor autovector asociado al menor valor singular osea V[:,-1] para nuestra SVD
def resolverDLT3D(Q):
  U,S,V = svd_reducida(Q)
  m = V[:,-1].copy() # tomo -1 para tomar el ultimo sin importar la dimension de V
  m = m.reshape(3,4) # segun la ecuacion que aparece en el pdf es solo reorganizar los componentes del vector m en una matriz M de 3x4
  return m