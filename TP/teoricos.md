# Ejercicio 1
#### _Respuesta inciso a_.
##### ¿Cual es el rango de $Q\in \mathbb{R}^{2n \times 12}$ para que la solución sea no nula única a menos de un factor de escala?
Para obtener una solución no nula $\text{dim}(\text{Nu}(Q)) \geq 1$ y para que sea a un factor de escala entonces $\text{Nu}(Q) = 1$, si $Q$ es la matriz de una transformación lineal $T$ entonces $T: \mathbb{V} \to \mathbb{W}$ donde $\mathbb{V} = \mathbb{R}^{12}$ y $\text{Im}(T) \subset \mathbb{R}^{2n}$. Por el teorema de la dimensión: $\text{dim}(\text{Im}(Q))+ \text{dim}(\text{Nu}(Q)) = \text{dim}(\text{Im}(Q)) + 1 = 12 \iff \text{dim}(\text{Im}(Q)) = 11$. Por lo tanto, como $\text{dim}(\text{Im}(Q)) = \text{rg}(Q)$, se tiene que el rango de $Q$ es igual a 11.
##### ¿Cuántos puntos, como mínimo, aseguran la solución y qué condición deben cumplir?
Teniendo en cuenta que cada punto aporta dos ecuaciones lineales independientes por $\mathbb{V} = \mathbb{R}^{2n}$ con $n \in \mathbb{N}$ como el número de puntos y que $2n \geq \text{rg}(Q)$, entonces $2n \geq 11 \implies n=6$. Por lo tanto, el número mínimo de puntos necesarios que aseguran la solución son 6.
La condición que deben cumplir es que una de las 12 filas de $Q$ debe ser combinación lineal de las otras 11 porque si $\text{rg}(Q)=12 \implies \text{Nu}(Q) = 0 \implies m = \vec{0}$, luego la matriz $M$ es nula. Si $\text{rg}(Q) \leq 10$ entonces hay puntos que pertenecen al mismo plano y $\text{Nu}(Q) \geq 2 $ provocando infinitas soluciones. Por ejemplo, si $\text{Nu}(Q) = 2 $ entonces existen $m_1, m_2$ con matrices asociadas $M_1, M_2$ y luego $M = Q(\lambda_1 m_1 + \lambda_2 m_2)$.
En conclusión 11 de las 12 filas de $Q$ son linealmente independientes y los puntos no pueden pertenecer a un mismo plano, pues entonces habría dependencia lineal, lo cual disminuiría el rango de la matriz y en consucuencia aumentando la dimensión de su núcleo obteniendo así infinitas soluciones.

#### _Respuesta inciso b_.
En teoría deberíamos hallar la solucion del sistema $Qm = 0$ donde $rg(Q) = 11$. Pero en la práctica, generalmente los errores podrían provocar que $rg(Q) = 12$, obteniendo la solución trivial y la matriz $M$ resultaría nula. Para evitar obtener la solución trivial se busca resolver el problema de mínimos cuadrados homogéneos con restricción de norma.
Toda matriz rectangular posee descomposición en valores singulares por lo que $Q = U \Sigma V^t$. Como $U$ es unitaria se tiene que:
$$
\begin{aligned}
Qm &= U \Sigma V^t m \\
&=  U (\Sigma V^t m) \\
&= \Sigma V^t m
\end{aligned}
$$
Entonces $||Qm||_2 ^2 = ||\Sigma V^t m||_2 ^2$.
Como $V$ es unitaria, tomando como vectores sus columnas, estos forman una base ortonormal de $\mathbb{R}^{12}$ de modo que $m = \sum_{i=1}^{12}\alpha_i v_i$ con $\sum_{i=1}^{12}\alpha_i^2 = 1$. Sea $A = (\alpha_1, \alpha_2, ... , \alpha_{12}) \in \mathbb{R}^{12}$, de forma matricial $m = VA$, luego
$$
\begin{aligned}
\Sigma V^t m &= \Sigma V^t VA \\
&= \Sigma (V^t V)A \\
&= \Sigma A
\end{aligned}
$$
De esta manera,
$$
\begin{aligned}
||Qm||_2 ^2 &= ||\Sigma A||_2 ^2 \\
&= \sum_{i=1}^{12} \sigma_i^2 \alpha_i^2 \\
&\geq \sigma_{12}^2 \sum_{i=1}^{12} \alpha_i^2 = \sigma_{12}^2
\end{aligned}
$$
El valor mínimo de la norma es $\sigma_{12}^{2}$. Este mínimo se alcanza cuando $A=\pm(0, 0, ... , 1)$ y por lo tanto $m = \pm v_{12}$.

# Ejercicio 3
#### ¿Qué rango debe tener $Q \in \mathbb{R}^{2n \times 9}$ para asegurar una solución única de H a menos de un factor de escala?

La respuesta, con una argumentación análoga al ejercicio _'1.a'_, es que el rango de $Q$ sea estrictamente igual 8, pues como se explicó, un número menor provocaría infinitas soluciones a mas de un factor de escala y un número mayor provocaría que la solución fuese la trivial obteniendo una matriz $H$ nula.