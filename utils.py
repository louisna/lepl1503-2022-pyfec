'''
	LEPL1503 - Projet 3, année académique 2021-2022

	Opérations sur les vecteurs en Galois Field 2^8
	Auteur : Antoine Knockaert
'''

import numpy as np
from gf256_tables import gf256_mul_table, gf256_inv_table

# Multiplication d'un vecteur et d'une valeur en GF256
# Multiplie chacune des entrées du vecteur vect par
# la valeur val
# Retourne le nouveau vecteur obtenu
def gf_256_mul_vector(vect, val):
    rep = vect.copy()  # La copie n'est pas necessaire :-)
    for i in range(len(vect)):
        rep[i] = gf256_mul_table[int(vect[i])][val]
    return rep


# Division d'un vecteur par une valeur en GF256
# Divise chacune des entrées du vecteur vect par
# la valeur val (une division revient au même
# qu'une multiplication par l'inverse)
# Retourne le nouveau vecteur obtenu
def gf_256_inv_vector(vect, val):
    rep = vect.copy()
    for i in range(len(vect)):
        rep[i] = gf256_mul_table[vect[i]][gf256_inv_table[val]]
    return rep


# Additionne deux vecteurs en GF256
# Additionne chaque terme un à un
# Retourne le nouveau vecteur obtenu
# v1 et v2 sont les deux vecteurs à additionner
def gf_256_full_add_vector(v1, v2):
    rep = v1.copy()
    for i in range(len(v1)):
        rep[i] = np.bitwise_xor(v1[i], v2[i])
    return rep


