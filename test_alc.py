import numpy as np
from alc import *

# test det(matriz)
A = np.array(
        [
            [1,0],
            [0,1]
        ]
    )
assert np.isclose(det(A), np.linalg.det(A))
A = np.array(
    [
        [0,1],
        [0,1]
    ]
)
assert np.isclose(det(A), np.linalg.det(A))

for k in range(10):
    random2x2 = [np.random.randint(0, 100, size=(2,2)) for i in range(10)]
    random3x3 = [np.random.randint(0, 100, size=(3,3)) for i in range(10)]
    for i in range(10):
        A2 = random2x2[i]
        A3 = random3x3[i]

        obtenido2 = det(A2)
        obtenido3 = det(A3)
        esperado2 = np.linalg.det(A2)
        esperado3 = np.linalg.det(A3)

        print(obtenido2)
        print(esperado2)
        assert np.isclose(obtenido2, esperado2)

        print(obtenido3)
        print(esperado3)
        assert np.isclose(obtenido3, esperado3)