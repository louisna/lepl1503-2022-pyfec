'''
    LEPL1503 - Projet 3, année académique 2021-2022

    Programme principal
    Auteurs : 
        Antoine Knockaert
        Louis Navarre
'''
import numpy as np
import math

import argparse
import sys
import os

from gf256_tables import gf256_mul_table, gf256_inv_table
from utils import gf_256_full_add_vector, gf_256_mul_vector, gf_256_inv_vector
from tinymt32 import TINYMT32

parser = argparse.ArgumentParser(description="LEPL1503 - Détection et correction d'erreurs")
parser.add_argument("input_dir", help="Chemin vers le dossier contenant les binaires des messages encodés")
parser.add_argument("-f", help="Chemin vers le fichier de sortie", type=argparse.FileType("wb"), default=sys.stdout)
parser.add_argument("-v", help="\"verbose\" mode: si ajouté, affiche des informations sur l'exécution du programme", action="store_true")
args = parser.parse_args()

verbose = args.v
output_fd = args.f
seed = None
block_size = None
word_size = None
redundancy = None
message_size = 0

coeffs = None


if verbose:
    print(args, file=sys.stderr)


def make_linear_system(unknown_indexes, nb_unk, current_block, block_size):
    """
    Construit un système linéaire Ax=b à partir des blocs donnés en argument.

    :param unknown_indexes: tableau des indexs des symboles source d'un bloc. L'entrée i est `True` si le symbole i est perdu
    :param nb_unk: le nombre d'inconnues dans le système - la taille du système
    :param current_block: le bloc de symboles à résoudre
    :param block_size: le nombre de symboles sources dans le bloc
    :return A: la matrice de coefficients
    :return B: le vecteur de termes indépendants. Chaque élément de B est de la même taille qu'un vecteur de données (paquet)
    """
    A = np.empty([nb_unk, nb_unk], dtype=int)
    B = np.empty([nb_unk, word_size], dtype=int)

    for i in range(nb_unk):
        B[i] = current_block[block_size + i]

    for i in range(nb_unk):
        temp = 0
        for j in range(block_size):
            if unknown_indexes[j]:  # Si c'est une inconnue
                A[i][temp] = coeffs[i][j]
                temp += 1
            else:
                B[i] = gf_256_full_add_vector(B[i], gf_256_mul_vector(current_block[j], coeffs[i][j]))
    
    if verbose:
        print_linear_system(A, B)
        print("Size :", nb_unk)

    return A, B


def matrix_solve(A, B):
    """
    Utilise la méthode de Gauss-Jordan pour résoudre le système linéaire Ax=B
    
    :param A: la matrice de coefficients du système
    :param B: le vecteur de termes indépendants. Chaque élément de B est de la même taille qu'un vecteur de données (paquet)
    :return B: lorsque le système est résolu, la matrice A est identitaire, et nous avons x=B
    """
    size = len(A)

    # Elimination de Gauss
    for k in range(size):
        for i in range(k+1, size):
            factor = gf256_mul_table[A[i][k]][gf256_inv_table[A[k][k]]]
            for j in range(size):
                A[i][j] = np.bitwise_xor(A[i][j], gf256_mul_table[A[k][j]][factor])
            B[i] = gf_256_full_add_vector(B[i], gf_256_mul_vector(B[k], factor))

    # Substitution arrière
    factor_tab = np.empty(word_size, dtype=int)
    for i in range(size-1, -1, -1):
        factor_tab.fill(0)
        for j in range(i+1, size):
            factor_tab = gf_256_full_add_vector(factor_tab, gf_256_mul_vector(B[j], A[i][j]))
        B[i] = gf_256_inv_vector(gf_256_full_add_vector(B[i], factor_tab), A[i][i])
    
    if verbose:
        print_linear_system(A, B)

    return B


def print_linear_system(A, B):
    """
    Affiche sur la console de sortie standard le système linéaire Ax=B

    :param A: la matrice de coefficients du système
    :param B: le vecteur de termes indépendants. Chaque élément de B est de la même taille qu'un vecteur de données (paquet)
    """
    print(">> Système linéaire")
    for i in range(len(A)):
        print(A[i], end = "")
        print("\t", end = "")
        print(B[i])
    print()


def make_block(data, size):
    """
    Construit le bloc sur base des données et de la taille d'un bloc

    :param data: les données du bloc en format binaire. Si le fichier d'input est bien formé, celui-ci est découpé
                 `size` symboles de taille `word_size` bytes, suivis de `redundancy` symboles de taille `word_size`
    :param size: le nombre de symboles sources dans un bloc
    :return block: le block construit, sous la forme d'une matrice (une ligne = un symbole)
    """
    block = np.empty([size + redundancy, word_size], dtype=int)
    for i in range(size + redundancy):
        for j in range(word_size):
            block[i][j] = data[i * word_size + j]
    return block


def process_block(block, size):
    """
    Sur base d'un bloc, trouve les inconnues (i.e., symboles sources perdus) et construit le système linéaire
    correspondant. Cette version simple considère qu'il y a toujours autant d'inconnues que de symboles de redondance,
    c'est-à-dire qu'on va toujours construire un système avec autant d'équations que de symboles de redondances.

    :param block: le bloc en question
    :param size: la taille du bloc
    :return block: retourne le bloc résolu
    """
    unknown_indexes, unknowns = find_lost_words(block, size)
    A, B = make_linear_system(unknown_indexes, unknowns, block, size)
    sol = matrix_solve(A, B)

    temp = 0
    for i in range(size):
        if unknown_indexes[i]:
            block[i] = B[temp]
            temp += 1

    return block


def find_lost_words(block, size):
    """
    Sur base d'un bloc, trouve les symboles sources perdus et les répertorie dans `unknown_indexes`.
    Un symbole est considéré comme perdu dans le bloc si le symbole ne contient que des 0

    :param block: le bloc en question
    :param size: la taille du bloc
    :return unknown_indexes: tableau de taille `size` faisant un mapping avec les symboles sources.
                             L'entrée `i` est `True` si le symbole source `i` est perdu
    :return unknowns: le nombre de symboles sources perdus
    """
    unknown_indexes = [False] * size
    unknowns = 0
    for i in range(size):
        count = 0
        for j in range(word_size):
            count += block[i][j]
        if count == 0:  # Un symbole avec uniquement des 0 est considéré comme perdu
            unknown_indexes[i] = True
            unknowns += 1
    if verbose:
        print(unknown_indexes)
    return unknown_indexes, unknowns


def block_to_string(block, size):
    """
    Fonction d'aide. Retourne un string stocké en binaire dans le bloc passé en argument

    :param block: le bloc en question
    :param size: la taille du bloc
    :return s: le string du bloc converti en binaire
    """
    s = ""
    for i in range(size):
        for j in range(len(block[0])):
            if block[i][j] == 0:
                return s
            s = s + chr(block[i][j])
    return s


def write_block(output_file, block, size, word_size):
    """
    Ecrit dans le fichier `output_file` le bloc en binaire

    :param output_file: le descripteur de fichier de sortie
    :param block: le bloc en question
    :param size: la taille du bloc
    :param word_size: la taille de chaque symbole du bloc
    """
    for i in range(size):
        for j in range(word_size):
            if output_file == sys.stdout or output_file == sys.stderr:
                print(chr(block[i][j]), end="")
            else:
                output_file.write(int(block[i][j]).to_bytes(1, 'big'))


def write_last_block(output_file, block, size, word_size, last_word_size):
    """
    Ecrit dans le fichier `output_file` le blo en binaire. Cette fonction se différencie de la précédente
    puisqu'elle doit gérer le cas où le dernier symbole du dernier bloc n'est pas de taille identique aux autres
    symboles de son bloc.

    :param output_file: le descripteur de fichier de sortie
    :param block: le bloc en question
    :param size: la taille d'un bloc
    :word_size: la taille d'un symbole dit 'complet'
    :last_word_size: la taille du dernier mot du dernier bloc
    """
    for i in range(size - 1):
        for j in range(word_size):
            if output_file == sys.stdout or output_file == sys.stderr:
                print(chr(block[i][j]), end="")
            else:
                output_file.write(int(block[i][j]).to_bytes(1, 'big'))
    
    for j in range(last_word_size):
        if output_file == sys.stdout or output_file == sys.stderr:
            print(chr(block[size - 1][j]), end="")
        else:
            output_file.write(int(block[size - 1][j]).to_bytes(1, 'big'))


def get_file_infos(data):
    """
    Récupère les informations du bloc `data`, comme spécifiées dans l'énoncé

    :param data: les 24 premieers bytes brutes du fichier
    :return block_size: la taille d'un bloc - le nombre de symboles sources dans le bloc
    :return word_size: la taille d'un symbole 'complet' dans un bloc
    :return redundancy: le nombre de symboles de redondance dans le bloc
    :message_size: la taille (en bytes) du fichier initial que nous souhaitons récupérer.
                Cette valeur ne prend en compte que les données du fichier, donc sans symbole de réparation
                ni les informations repries ci-dessus
    """
    seed = int.from_bytes(data[0:4], "big")
    block_size = int.from_bytes(data[4:8], "big")
    word_size = int.from_bytes(data[8:12], "big")
    redundancy = int.from_bytes(data[12:16], "big")
    message_size = int.from_bytes(data[16:], "big")
    return seed, block_size, word_size, redundancy, message_size


def gen_coeffs():
    """
    Génère tous les coefficients du bloc. Il y a un coefficient r_{i, j} pour chaque 
    symbole de redondance i et symbole source j du bloc

    :return: les coefficients sous forme matricielle
    """
    coefs = np.empty([redundancy, block_size], dtype=int)
    tinymt32 = TINYMT32(seed)
    for i in range(redundancy):
        for j in range(block_size):
            coefs[i][j] = tinymt32.tinymt32_generate_uint32() % 256
            if coefs[i][j] == 0:
                coefs[i][j] = 1
    return coefs


def print_coeffs():
    """
    Affiche les coefficients sur la sortie standard
    """
    if coeffs is None:
        print("You have to generate coefficients before printing them!")
        return
    print(">> Coefficients : ")
    print(coeffs)


if __name__ == "__main__":
    for filename in os.listdir(args.input_dir):
        with open(os.path.join(args.input_dir, filename), "rb") as input_file:
            binary_data = input_file.read()
            seed, block_size, word_size, redundancy, message_size = get_file_infos(binary_data[:24])
            if verbose:
                print("Seed :", seed, ", block_size :", block_size, ", word_size :", word_size, ", redundancy :", redundancy)

            binary_data = binary_data[24:]

            # Génère les coefficients
            coeffs = gen_coeffs()
            if verbose:
                print_coeffs()
                print(binary_data)

            step = word_size * (block_size + redundancy)  # Taille de bloc en bytes 
            length = len(binary_data)

            # Ecrit le nom du fichier dans l'output
            if output_fd == sys.stdout or output_fd == sys.stderr:
                print(filename, file=output_fd)
            else:
                output_fd.write(filename.encode("ASCII"))
                output_fd.write("\n".encode("ASCII"))

            # Contient le nombre de blocs complets (sans le dernier bloc s'il n'est pas complet)
            nb_blocks = math.ceil(length / (word_size * (block_size + redundancy)))
            contains_uncomplete_block = False
            if message_size != nb_blocks * block_size * word_size:  # Dernier bloc non complet (i.e., moins de block_size symboles)
                nb_blocks -= 1
                contains_uncomplete_block = True

            readed = 0  # Nombre de bytes lus jusqu'à présent
            for i in range(nb_blocks):
                current_block = make_block(binary_data[i * step:(i + 1) * step], block_size)
                response = process_block(current_block, block_size)
                if verbose:
                    print(current_block)
                    print(block_to_string(response, block_size))
                
                write_block(output_fd, response, block_size, word_size)

                readed += step

            readed_symbols = block_size * word_size * nb_blocks
            nb_remaining_symbols = (len(binary_data[readed:]) // word_size) - redundancy
            if contains_uncomplete_block:
                last_block = make_block(binary_data[readed:], nb_remaining_symbols)
                decoded = process_block(last_block, nb_remaining_symbols)
                # Le dernier symbole peut ne pas etre complet et contenir du padding
                # => on utilise la taille totale du fichier pour retirer le padding
                padding = readed_symbols + nb_remaining_symbols * word_size - message_size
                true_length_last_symbol = word_size - padding
                if verbose:
                    print(last_block)
                
                write_last_block(output_fd, decoded, nb_remaining_symbols, word_size, true_length_last_symbol)
            
            # Séparation avec le fichier suivant
            if output_fd == sys.stdout or output_fd == sys.stderr:
                print("\n", file=output_fd)
            else:
                output_fd.write("\n\n".encode("ASCII"))
