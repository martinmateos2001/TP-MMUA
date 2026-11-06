from math import pi

import numpy as np
from pandas import pivot
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
  '''
  - A,B: Arreglos numpy.
  Retorna la multiplicación matricial de A y B.
  '''
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
  '''
  - v,w: vectores de numpy.
  Retorna el prodcuto punto de v y w.
  '''
  if len(v) == len(w):
    suma = 0
    for i in range(0,len(v)):
      suma = suma + v[i]*w[i]
    return suma

def normavector2(x):
  '''
  - x: vector de numpy.
  Retorna la Norma-2 del vector x.
  '''
  norma = 0
  for i in range(0,len(x)):
    norma = norma + x[i]**2
  norma = norma ** (1/2)
  return norma

def normalizar(x):
  '''
  - x: vector de numpy.
  Retorna el vector x normalizado.
  '''
  norma = normavector2(x)
  x = x / norma
  return x

def metpot2k(A,tol=1e-15,kmax=1000):
  """
  - A, matriz n x n
  - tol, tolerancia en la diferencia entre un paso y el siguiente en la estimación del autovector.
  - K, numero máximo de iteraciones a realizarse.
  Retorna:
      v, l, k
  Donde v es un autovector A, l es el autovalor de A con autovector v, k el número de iteraciones. 
  """
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
    """
    A: matriz n x n simétrica.
    tol: tolerancia en la estimación del autovector. 
    K: numero de iteraciones máxima.
    Retorna matriz de autovectores S y la matriz de autovalores D, tal que A = S@D@S.T.
    En caso de que no sea simetrica retorna None.
    """
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

            D[0, 0] = l1.reshape(-1)[0]
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
  S, D = diagRH(AtA)  # D tiene los autovalores y V los autovectores
  V = S
  autovalores = np.zeros(n)
  for i in range(0, n):
    autovalores[i] = D[i, i]

  # ahora debo ordenar autovalores y autovectores de mayor a menor
  for i in range(0, len(autovalores)):
    idmax = i
    for j in range(i + 1, n):
      if autovalores[j] > autovalores[idmax]:
        idmax = j
    autovalores[i], autovalores[idmax] = (
        autovalores[idmax],
        autovalores[i],
    )  # cambio de lugar el maximo y mi indice de ahora
    tempautovector = V[:, i].copy()
    V[:, i] = V[:, idmax]
    V[:, idmax] = tempautovector

  # Si k == 'max', debemos retener los n autovectores sin descartar el espacio nulo (autovalores cercanos a 0)
  if k == "max":
    kretenido = n
    hatSig = np.zeros(n)
    hatV = np.zeros((n, n))
    for i in range(n):
      valor = max(autovalores[i], 0.0)
      hatSig[i] = np.sqrt(valor)
      hatV[:, i] = V[:, i]
  else:
    tol_autovalor = max(tol**2, 1e-12)
    singulares = []
    autovectores = []
    for i in range(len(autovalores)):
      valor = autovalores[i]
      if valor > tol_autovalor:
        valorsing = np.sqrt(valor)
        if valorsing > tol:  # miro si el valor singular es mayor a la tolerancia, sino es 0
          singulares.append(valorsing)
          autovectores.append(V[:, i])

    rango = len(singulares)  # Guardo el rango maximo de la matriz viendo cuantos valores singulares tiene
    if k > rango:
      kretenido = rango
    else:
      kretenido = int(k)

    if kretenido == 0:  # si piden rango 0
      hatU = np.zeros((m, 0))
      hatSig = np.array([])
      hatV = np.zeros((n, 0))
      return hatU, hatSig, hatV  # basicamente devuelvo todo vacio

    # Ahora caso si k>0
    hatSig = np.array(
        singulares[:kretenido]
    )  # es una lista pues me pedian eso para mi funcion en este caso
    hatV = np.zeros((n, kretenido))  # creo matriz nxk
    for i in range(0, kretenido):
      hatV[:, i] = autovectores[i]  # copio los autovectores columna para V

  # me falta crear U, lo voy a hacer con la formula de que Ui = 1/sigmai * Avi
  # Puedo calcular mas facil si guardo todo Avi de una en una matriz B
  B = multMat(A, hatV)  # m*k dimensiones
  hatU = np.zeros((m, kretenido))
  for j in range(kretenido):
    if hatSig[j] > 1e-14:  # Evita divisiones por cero en el espacio nulo
      hatU[:, j] = B[:, j] / hatSig[j]

  if se_traspuso:
    hatU, hatV = hatV, hatU

  return hatU, hatSig, hatV

# --- Ejercicio 2

def puntosfotoi(i,uv):
  '''
  - i: Entero que representa la i-ésima foto de puntos 2D.
  - uv: Tabla de fotos donde cada fila es representa una foto que se compone de 12 pares de float, es decir uv es (210 x 24) 

  '''
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

def nomalizacionDePuntos(imgPts, objPts):
  '''
  Dados los puntos 2D y los puntos 3D asociados, retorna ambas listas de puntos normalizadas 
  y las matrices de normalizacion T_img y T_world
  '''
  pts2d=np.copy(imgPts)
  pts3d=np.copy(objPts)

  # centroides 2D y 3D
  cen2d = np.array([0,0])
  cen3d = np.array([0,0,0])
  n = len(imgPts)
  for i in range(len(imgPts)):
    cen2d += imgPts[i]
    cen3d += objPts[i]
  cen2d = cen2d / n
  cen3d = cen3d / n

  # traslacion
  for i in range(n):
    pts2d[i] -= cen2d
    pts3d[i] -= cen3d

  # promedio de las distancias al origen
  prom2d = 0
  prom3d = 0
  for i in range(n):
    prom2d += normavector2(pts2d[i])
    prom3d += normavector2(pts3d[i])
  prom2d = prom2d / n
  prom3d = prom3d / n

  # factores de normalizacion
  s2d = (2**(0.5)) / prom2d
  s3d = (3**(0.5)) / prom3d
  for i in range(n):
    pts2d[i] = pts2d[i] * s2d
    pts3d[i] = pts3d[i] * s3d

  T_img = np.array([
     [s2d, 0, -s2d * cen2d[0]],
     [0, s2d, -s2d * cen2d[1]],
     [0, 0, 1]
  ])

  T_world = np.array([
     [s3d, 0, 0, -s3d * cen3d[0]],
     [0, s3d, 0, -s3d * cen3d[1]],
     [0, 0, s3d, -s3d * cen3d[2]],
     [0, 0, 0, 1]
  ])

  return pts2d, pts3d, T_img, T_world
# 2A
def obtenerQ3d(imgPts, objPts):
  '''
  - imgPts: arreglo numpy (12 x 2), coordenadas de pixeles de una imagen.
  - objPts: arreglo numpy (12 x 3), coordenadas 3D asociadas a cada coordenada de imgPts.
  Retorna la matriz Q asociada a la ecuacion Qm=0 utilizada para obtener la matriz de calibración M.
  '''
  Q = np.zeros((24,12))
  for i in range(0,12):
    Q[2*i,0:4] = objPts[i]
    Q[2*i,8:12] = -imgPts[i][0]*objPts[i]
    Q[2*i+1,4:8] = objPts[i]
    Q[2*i+1,8:12] = -imgPts[i][1]*objPts[i]
  return Q

# 2B resolverDLT3D(Q) encuentra la matriz M a partir de Q, por 1b el vector m es el menor autovector asociado al menor valor singular osea V[:,-1] para nuestra SVD
def resolverDLT3D(Q):
  '''
  - Q: matriz asociada a la ecuación Qm = 0 para obtener la matriz de proyeccion M.
  Retorna la matriz de proyeccion M.
  '''
  _, _, V = svd_reducida(Q)
  m = V[:,-1].copy() # tomo -1 para tomar el ultimo sin importar la dimension de V
  m = m.reshape(3,4) # segun la ecuacion que aparece en el pdf es solo reorganizar los componentes del vector m en una matriz M de 3x4
  return m

# 2: C y D
def devolverMatricesCalibracion3d(M,uigualv = False, k12igual0 = False):
  '''
  - M: matriz de proyeccion.
  - uigualv: booleano, cuando es true indica que K_{1,1} = K_{2,2} (el factor de escala en los ejes es el mismo)
  - k12igual0: booleano, cuando es true K_{1,2} = 0 (sesgo o la falta de perpendicularidad en los ejes es nulo/a).
  Retorna la matriz de rotación R, el vector de traslacion t, la matriz de parametros intrinsecos K.
  '''
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
  if det(R) == -1:
    R = -R
    t = -t
  return R , t, Knormalizada

# --- Ejercicio 3 ---

def obtenerQ2d(imgPts, objPts): 
    '''
    - imgPts: arreglo (m x 2) de coordenadas de pixels.
    - objPts: arreglo (m' x 3) de coordenadas 3D asociado a imgPts.
    Requisito: m >= 4 y m = m'.

    Retorna la matriz Q asociada a la ecuacion Qh = 0 para la obtención de una homografía.
    -  Si m != m' entonces retorna None.
    '''
    m = imgPts.shape[0]
    if(m != objPts.shape[0]):
        print("Las coordenadas de puntos 2D y 3D no forman pares.")
        return None
    Q = np.zeros((2*m,9))
    for i in range(0, m):
        P = objPts[i].copy()
        u = imgPts[i][0]
        v = imgPts[i][1]
        Q[2*i, 0:3] = P
        Q[2*i, 6:] = -u*P
        Q[2*i + 1, 3:6] = P
        Q[2*i + 1, 6:] = -v*P
    return Q

def resolverQh(Q):
    '''
    - Q: Matriz asociada a la ecuación Qh=0.
    Resuelve la ecuacion y retorna la matriz H asociada a h.
    '''
    _, _, V = svd_reducida(Q)
    h = V[:,-1].copy() # tomo -1 para tomar el ultimo sin importar la dimension de V
    H = h.reshape(3,3) # Reorganizo
    return H

def vijK(H, i, j):
   return np.array([
    H[0,i]*H[0,j],
    H[0,i]* H[1,j] + H[1,i]*H[0,j],
    H[1,i]*H[1,j],
    H[0,i]*H[2,j] + H[2,i]*H[0,j],
    H[1,i]*H[2,j] + H[2,i]*H[1,j],
    H[2,i]*H[2,j]])

def VK(H):
   '''
   - H: Homografía (3 x 3)
   Retorna la matriz (2 x 6) que tiene como filas a los dos vectores que forman filas de A.
   '''
   return np.array([vijK(H, 0, 1), (vijK(H, 0, 0)- vijK(H, 1, 1))])

def armarA(Hlst):
    '''
    - Hlst: lista de matrices H_j correspondientes a N capturas de mira, obtenidas a partir de la ecuacion Qh = 0.
    Retorna la matriz A asociada a la ecuacion Ab = 0 del Metodo de Tsai.
    '''
    N = len(Hlst)
    A = np.zeros((2*N, 6))
    for i in range(0, N):
      H = Hlst[i]
      A[2*i, :] = VK(H)[0]
      A[2*i+1, :] = VK(H)[1]
    return A

def calculaLU(A):
    '''
    - A: matriz definida positiva.
    Retorna las matrices L, U y el número de operaciones, donde A=LU.
    '''
    if(A is None):
        return None, None, 0
    # dimensiones
    m = A.shape[0]
    n= A.shape[1]
    if( m!=n):
        return None, None, 0
    Ac = A.copy()
    # Identidad para crear L
    L = np.eye(n)
    ops = 0
    j = 0
    while(j < n-1):
        # si el pivote es nulo no puedo operar
        if(Ac[j][j] == 0):
            return None, None, 0
        i=j+1
        
        while(i<n):
            if(Ac[i][j] != 0):
                m = Ac[i][j] / Ac[j][j]
                ops += 1
                L[i][j]= m
                Ac[i][j]=0.0 # asigno para no contar una operacion.
                Ac[i][j+1:] = Ac[i][j+1:] - m * Ac[j][j+1:]
                ops += 2*(n-(j+1)) # resto los que cambian, de la columna j+1 para atras son ceros.
            i+=1
        j+=1
    return L, Ac, ops

def calculaLDV(A):
    LU = calculaLU(A)
    L = LU[0]
    U = LU[1]
    Uc = np.copy(U)
    D = np.zeros(A.shape)
    n = A.shape[0]
    for i in range(0, n):
        D[i,i] = Uc[i,i]
        for j in range(i, n):
            if(Uc[i,j] != 0):
                Uc[i,j] = Uc[i,j]/D[i,i]
    return L, D, Uc

def esSDP(A, atol=1e-5):
    n=A.shape[0]
    if(n != A.shape[1]):
        return False
    L,D,V= calculaLDV(A)
    for i in range(0,n):
        if(D[i,i] <= 0):
            return False
        for j in range(0, i+1):
            if not (np.isclose(L[i,j], V[j,i], atol=atol)):
                return False
    return True

def descCholesky(A):
    '''
    Dada A simetrica y definida positiva, retorna la matriz L triangular inferior tal que LL^t = A. 
    '''
    n = A.shape[0]
    L = np.zeros((n,n))
    for i in range(n):
      for k in range(n):
        s = 0
        if i == k:
            for j in range(k):
              s += L[k,j]**2
            L[k, k] = np.sqrt(A[k,k] - s)
        if i > k:
            for j in range(k):
              s += L[i,j]*L[k,j]
            L[i,k] = (A[i,k] - s)/L[k,k]
    return L

def calcularMatrizK2D(A):
  '''
  - A: matriz asociada a la ecuacion Ab = 0 del Metodo de Tsai.
  Retorna la matriz K.
  '''
  # Resuelvo el sistema con cuadrados mínimos homogeneos con restriccion de norma.
  # uso SVD reducida otra vez.
  _, _, V = svd_reducida(A)
  b = V[:,-1].copy()
  B = np.array( # B simetrica tq K.-t * k.-1 = B
    [
          [b[0], b[1], b[3]],
          [b[1], b[2], b[4]],
          [b[3], b[4], b[5]]
    ])
  # Aca si B no es sdp entonces me quedo el vector con el signo invertido, entonces multiplico por -1
  if B[0, 0] < 0:
    B = -B
  # 4. Verificación de seguridad antes de Cholesky
  if not esSDP(B):
    print("B no es SDP. Autovalores de B:", np.linalg.eigvalsh(B))
    raise ValueError(
        "La matriz B no es definida positiva. Revisar las homografías H o el número de vistas."
  )
  L = descCholesky(B)
  K = inversa(L.T)
  Knormalizada = K / K[2, 2]
  return Knormalizada

def productoCruz(a, b):
  # Aseguramos que los vectores estén desdoblados en 1D
  a = a.reshape(-1)
  b = b.reshape(-1)

  c1 = a[1] * b[2] - a[2] * b[1]
  c2 = a[2] * b[0] - a[0] * b[2]
  c3 = a[0] * b[1] - a[1] * b[0]

  return np.array([c1, c2, c3])

def obtenerPose2D(Hj, K):
  '''
  Retorna la pose de la vista j como R_j y t_j a partir de la matriz K y la H_j correspondiente.
  '''
  Kinv = inversa(K)
  # como ya voy a calcuar Kinv * c1 , c2 ,c3 multiplico todo y luego saco las columnas me ahorro cuentas
  Kci = multMat(Kinv,Hj)
  Kc1 = Kci[:,0]
  Kc2 = Kci[:,1]
  Kc3 = Kci[:,2]
  lam = 1 / normavector2(Kc1) # calculamos lambda

  r1 = lam * Kc1
  r2 = lam * Kc2
  r3 = productoCruz(r1,r2)
  t = (lam * Kc3).reshape(3,1)

  Rraw = np.array([r1,r2,r3]).T # las pongo como filas y traspongo para que queden como columnas

  U,_,V = svd_reducida(Rraw)
  R = multMat(U,V.T)
  if det(R) < 0:
     R = -R
     t = -t
  
  return R,t

def det(matriz):
    '''
    Obtencion del determinante de una matriz por el metodo de triangulacion por eliminacion Gaussiana.
    '''
    A = np.array(matriz)
    A = np.float32(A)
    m, n = A.shape
    if(m != n):
        raise ValueError("La matriz no es cuadrada")
    signo = np.float32(1)
    res = np.float32(1)

    for i in range(n):     
        max_p = i
        fila = i + 1
        # busco el max pivote
        while fila < n:
            if abs(A[fila, i]) > abs(A[max_p, i]):
                max_p = fila
            fila += 1
        pivote = A[max_p, i]
        if pivote == 0.0:
           return np.float32(0)
        
        # intercambio de filas
        if(max_p != i):
           f1 = A[i].copy()
           f2 = A[max_p].copy()
           A[i] = f2
           A[max_p] = f1
           signo = signo * (-1)
        pivote = A[i,i]
        res = res * pivote

        # eliminación gaussiana
        for j in range(i + 1, n):
            multiplicador = A[j,i] / pivote
            A[j,i + 1:] -= multiplicador * A[i, i+1:]
            A[j, i] = np.float32(0)
    return res * signo

# Para ejercicio 6
def pseudoinversa(A,tol=1e-10):
  """ calcula la pseudoinversa a partir de svd reducida"""
  U,S,V = svd_reducida(A)
  for i in range(len(S)):
    if S[i]>tol: # si no es nulo lo invierto haciendo 1/S[i]
      S[i] = 1 / S[i]

  Sinv = np.diag(S) # Creo la matriz diagonal pues recordemos mi svd me devuelve en forma de lista los valores singulares.
  Apinv = multMat(multMat(V,Sinv),U.T) # Como mi SVD reducida devuelve V en vez de Vt, no tengo que trasponerla

  return Apinv




def calculoLambdaMu(I0,J0):
  """
    Parametros:
        I0,J0 Vectores de proyección ortografica parciales en los ejes horizontal y vertical de la camara
        I0: Array 2D [I0x, I0y].
        J0: Array 2D [J0x, J0y].
    Retorna:
        lambda1: Solucion para Iz (componente Z del vector I).
        mu1: Solucion para Jz (componente Z del vector J).
        lambda2: Segunda Iz (-lambda1).
        mu2: Segunda Jz (-mu1).
    """
  # Calculo normas
  normasqI0 = normavector2(I0**2)
  normasqJ0 = normavector2(J0**2)
  puntoI0J0 = productoPunto(I0,J0)

  real = normasqJ0 - normasqI0
  imag = -2 * puntoI0J0
  # R es el modulo
  R = np.sqrt(real**2+imag**2)

  lambda1 = np.sqrt((R+real)/2)
  mu_abs = np.sqrt((R-real)/2)

  if imag>=0: # El signo de mu1 se ve con el signo original de la parte imaginaria.
    mu1 = mu_abs
  else:
    mu1 = -mu_abs

  lambda2 = -lambda1
  mu2 = - mu1
  """
  Si se puede usar complex entonces
  C2 = complex(real,imag)
  C1 = np.sqrt(C2)

  lambda1 = C1.real
  lamba2 = -lambda1

  mu1 = C1.imag
  mu2 = -mu1
  """
  return lambda1,lambda2,mu1,mu2





def calcularSalidaPOSIT(Is, Js, fx, m0, M0):
  """
  Parametros:
      Is: Vector 3D [I0x, I0y, lambda].
      Js: Vector 3D [J0x, J0y, mu].
      fx: Distancia focal en x (K[0,0]).
      m0: Array/lista 2D con la posicion centrada [x0, y0].
      M0: Array/lista 3D con la posicion del objeto [U0, V0, W0].
  Retorna:
      R: Matriz de rotacion (3x3).
      t: Vector de traslacion (3x1).
      Z0: Profundidad numero
  """
  normaI = normavector2(Is)
  normaJ = normavector2(Js)
  escala = (normaI+normaJ)/2 # podria usar tanto la norma de I como la norma de J en teoria pero tomo el promedio

  Z0 = fx/escala

  inormalizado = Is / normaI
  jnormalizado = Js / normaJ
  k = productoCruz(inormalizado,jnormalizado)
  irow = inormalizado.reshape(3)
  jrow = jnormalizado.reshape(3)
  krow = k.reshape(3)

  R = np.array([irow,jrow,krow])

  x0,y0 = m0[0],m0[1] # es lo mismo nomas lo desempaqueto para poder leer mejor que son

  tx = (x0 * Z0) / fx
  ty = (y0 * Z0) / fx
  tz = Z0

  T = np.array([tx,ty,tz])
  t = T.reshape(3,1) - multMat(R,M0.reshape(3,1))
  # Retornamos la pose con R,t, y la profundidad con Z0
  return R,t,Z0





def evaluarPose(R, t, obj_vecs, img_vecs, K):
  """
  Parametros:
      R : Matriz de rotacion de 3x3.
      t : Vector de traslacion de 3x1 o 1D de 3 elementos.
      obj_vecs : Matriz de Nx3 con puntos 3D del objeto (metros).
      img_vecs : Matriz de Nx2 con posiciones observadas (pixeles).
      K : Matriz de calibracion intrinseca de 3x3.
  Retorna:
      cant_validos (int): Numero de puntos proyectados con Zc > 0.
      rms (float): Error RMS de reproyeccion en pixeles.
  """
  cantvalidos = 0
  sumaErrores = 0

  cantPuntos = len(obj_vecs)
  for i in range(0,cantPuntos):
    Mi = obj_vecs[i] # los que me dieron puntos reales.
    mi = img_vecs[i] # estos es los pixeles
    Pcam = np.array([multMat(R,Mi)]) + t
    Pcam = Pcam.reshape(-1)
    Zi = Pcam[2]
    if Zi > 0:
      cantvalidos+=1

      Kp = multMat(K,Pcam)

      x_proy = Kp[0] / Zi # dividimos por la profundidad
      y_proy = Kp[1] / Zi # dividimos por la profundidad

      dx = x_proy - mi[0] # diferencia entre el proyectado y el observado
      dy = y_proy - mi[1]

      sumaErrores += dx**2 + dy**2

  if cantvalidos > 0:
    rms = np.sqrt(sumaErrores/cantvalidos)
  else:
    rms = None
  return cantvalidos, rms



def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt, atol):
  """
  Parametros:
      obj_vecs : Matriz Nx3 de puntos 3D reales del objeto.
      img_vecs : Matriz Nx2 de coordenadas de pixel originales (u,v).
      K : Matriz de calibracion intrinseca de 3x3.
      MaxIt : Numero maximo de iteraciones.
      atol : Tolerancia absoluta para la diferencia del RMS entre
        iteraciones.

  Retorna:
      R : Matriz de rotacion final de 3x3.
      t : Vector de traslacion final de 3x1.
  """
  # Parametros de K
  fx = K[0, 0]
  cu, cv = K[0, 2], K[1, 2]
  # Coordenadas centradas iniciales
  x_orig = img_vecs[:, 0] - cu
  y_orig = img_vecs[:, 1] - cv

  # Punto de referencia M0 (primer punto 3D) y m0 (su proyección 2D centrada)
  m0 = np.array([x_orig[0], y_orig[0]])
  M0 = obj_vecs[0]

  # Pseudoinversa de los puntos 3D del objeto (constante en todas las iteraciones)
  B = pseudoinversa(obj_vecs)
  # Vectores x y que se van ajustando con cada iteracion
  xk = x_orig.copy()
  yk = y_orig.copy()
  rmsPrevio = 0

  Rmejor, tmejor = None,None

  for i in range(0,MaxIt):
    Ik = multMat(B,xk)
    Jk = multMat(B,yk)
    # Obtener las dos raices de lambda y mu
    l1,l2,mu1,mu2 = calculoLambdaMu(Ik,Jk)
    # Formar vectores Is y Js para ambas ramas
    Is1 = np.array([Ik[0], Ik[1], l1])
    Js1 = np.array([Jk[0], Jk[1], mu1])

    Is2 = np.array([Ik[0], Ik[1], l2])
    Js2 = np.array([Jk[0], Jk[1], mu2])

    # Calcular las dos poses posibles
    R1, t1, Z01 = calcularSalidaPOSIT(Is1, Js1, fx, m0, M0)
    R2, t2, Z02 = calcularSalidaPOSIT(Is2, Js2, fx, m0, M0)

    # Evaluar ambas poses con los pixeles originales
    val1, rms1 = evaluarPose(R1, t1, obj_vecs, img_vecs, K)
    val2, rms2 = evaluarPose(R2, t2, obj_vecs, img_vecs, K)

    # Seleccionar la rama con mayor cantidad de puntos validos o menor RMS
    if val1 > val2:
      Rk, tk, Z0k, rmsk = R1, t1, Z01, rms1
    elif val2 > val1:
      Rk, tk, Z0k, rmsk = R2, t2, Z02, rms2
    else:
      # Si ambas tienen los mismos puntos validos, elegir la de menor RMS
      if rms1 <= rms2:
        Rk, tk, Z0k, rmsk = R1, t1, Z01, rms1
      else:
        Rk, tk, Z0k, rmsk = R2, t2, Z02, rms2

    Rmejor, tmejor = Rk, tk

    # Aca frenamos por convergencia si la diferencia entre el RMS anterior y el actual es muy chiquita.
    diferenciaRMS = abs(rmsPrevio - rmsk)
    if diferenciaRMS < atol:
      break
    # Si no es el caso entonces seguimos.
    rmsPrevio = rmsk

    # Actualización de factores de corrección de perspectiva epsilon
    R3 = Rk[2, :]  # Tercera fila de la matriz de rotación (eje Z)
    epsilon = np.zeros(len(obj_vecs))

    for i in range(0,len(obj_vecs)):
      Z0k = Z0k.reshape(()) # tengo que hacer esto pues mi Z0k era una lista de un numero y se bugeaba. sino se podia usar tambien [0]
      epsilon[i] = productoPunto(R3, obj_vecs[i]) / Z0k
    # Corrección de los píxeles centrados para la siguiente iteración
    xk = x_orig * (1 + epsilon)
    yk = y_orig * (1 + epsilon)

  return Rmejor,tmejor
