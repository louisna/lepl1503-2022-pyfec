'''
	LEPL1503 - Projet 3, année académique 2021-2022

	Programme permettant de générer des fichiers binaires
	utilisés ensuite comme input pour le programme principal

	Auteurs : 
		Antoine Knockaert
		Louis Navarre
'''

import numpy as np
import random as rand
import math
import argparse
import sys
from utils import gf_256_full_add_vector, gf_256_mul_vector
from tinymt32 import TINYMT32
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="LEPL1503 : FEC - création de fichiers d'input")
parser.add_argument("input_file", help="Chemin vers le fichier d'input contenant le message",
                    type=argparse.FileType('rb'), default=sys.stdin.buffer)
parser.add_argument("output_file", help="Chemin vers le fichier de sortie",
                    type=argparse.FileType('wb'), default=sys.stdout)
parser.add_argument(
    "-b", help="Taille d'un bloc (i.e. nombre de symboles sources par bloc) - 4 par défaut", type=int, default=4)
parser.add_argument(
    "-w", help="Taille d'un mot (i.e. nombre de bytes par symbole source) - 2 par défaut", type=int, default=2)
parser.add_argument(
    "-r", help="Nombre de symboles de redondance par bloc - 2 par défaut", type=int, default=2)
parser.add_argument(
    "-s", help="La \"seed\" pour générer les coefficients pseudo-aléatoires - 42 par défaut", type=int, default=42)
parser.add_argument(
    "-n", help="Simule des pertes dans les symboles sources de chaque bloc", action="store_true")
parser.add_argument(
    "--loss-pattern", help="Le nombre de pertes dans chaque bloc: [1->`min(valeur de r, valeur de b)`] - aléatoire dans ce range de valeurs par défaut", type=int, default=-1)
parser.add_argument(
    "-v", help="\"verbose\" mode: si ajouté, affiche des informations sur l'exécution du programme", action="store_true")
args = parser.parse_args()


textfile = args.input_file.read()
outputFile = args.output_file
block_size = args.b
word_size = args.w
redundancy = args.r
seed = args.s
verbose = args.v
noise = args.n
step = word_size * block_size  # Taile d'un bloc en bytes plutot qu'en mots
loss_pattern = args.loss_pattern

if loss_pattern <= 0 and loss_pattern != -1:
    print(
        f"Le pattern de perte doit etre un entier positif, recu : {loss_pattern}", file=sys.stderr)
    sys.exit(1)
if loss_pattern > min(block_size, redundancy):
    print(
        f"Le pattern de perte doit etre au maximum equivalent au nombre de symboles dans un bloc, et plus petit ou egal au nombre de symboles de redondance, recu: {loss_pattern}", file=sys.stderr)
    sys.exit(1)

coeffs = None

state_file = open("states.txt", 'w')

# Imprime des information sur les arguments du programme
if verbose:
    print(args, file=sys.stderr)


def gen_coeffs():
    coefs = np.empty([redundancy, block_size], dtype=int)
    tinymt32 = TINYMT32(seed)
    for i in range(redundancy):
        for j in range(block_size):
            coefs[i][j] = tinymt32.tinymt32_generate_uint32() % 256
            if coefs[i][j] == 0:
                coefs[i][j] = 1
    return coefs


# Imprime les coefficients
def print_coeffs():
    if coeffs is None:
        print("You have to generate coefficients before printing them!")
        return
    print(">> Coefficients : ")
    print(coeffs)


# Construit et retourne un bloc de données
# sur base des données brutes data passées en argument
# "array" est un tableau de bytes des données  # TODO: vérifier si c'est bien un tableau de bytes
def make_block(array):
    block = np.zeros([block_size + redundancy, word_size], dtype=int)
    for i in range(block_size):
        for j in range(word_size):
            block[i][j] = array[i * word_size + j]
    return block


# Construit et retourne le dernier bloc de données
# Pour le padding, on utilise des bytes valant "0"
def make_last_block(array, nb_remaining_symbols):
    block = np.zeros([nb_remaining_symbols + redundancy, word_size], dtype=int)
    size_data = len(array)
    count = 0
    for i in range(nb_remaining_symbols):
        for j in range(word_size):
            if count < size_data:
                block[i][j] = array[i * word_size + j]
            else:
                block[i][j] = 0
            count += 1
    return block


# Calcule la valeur des coefficients de redondance et
# les ajoute au bloc passé en argument
# Retourne le bloc complété
def compute_redundancy(block, nb_remaining_symbols):
    for i in range(redundancy):
        for j in range(nb_remaining_symbols):
            block[nb_remaining_symbols + i] = gf_256_full_add_vector(
                block[nb_remaining_symbols + i], gf_256_mul_vector(block[j], coeffs[i][j]))
    return block


# Retire de manière aléatoire des mots dans le bloc passé en argument
# Retourne le nouveau bloc obtenu
def add_noise(block, block_size, loss_pattern):
    if loss_pattern == -1:
        nb_symbols_to_remove = rand.randint(0, min(block_size, redundancy))
    else:
        nb_symbols_to_remove = min([block_size, redundancy, loss_pattern]) - 1
    if nb_symbols_to_remove == 0:
        return block
    indexes_to_remove = rand.sample(range(0, block_size), nb_symbols_to_remove)
    for i in indexes_to_remove:
        block[i] = np.zeros(word_size, dtype=int)
    return block


# Écrit le bloc passé en argument dans le fichier de sortie
# spécifié en argument du programme
def write_block_in_file(block, block_size):
    for i in range(block_size + redundancy):
        for j in range(word_size):
            outputFile.write(int(block[i][j]).to_bytes(1, 'big'))


def my_tqdm(r):
    if verbose:
        return r
    else:
        return tqdm(r)


if __name__ == "__main__":
    # Génère les coefficients
    coeffs = gen_coeffs()
    if verbose:
        print_coeffs()

    # On écrit dans le fichier binaire de sortie la seed, la taille d'un bloc
    # la taille d'un mot et le nombre de redondance
    outputFile.write(seed.to_bytes(4, 'big'))
    outputFile.write(block_size.to_bytes(4, 'big'))
    outputFile.write(word_size.to_bytes(4, 'big'))
    outputFile.write(redundancy.to_bytes(4, 'big'))
    outputFile.write(len(textfile).to_bytes(8, 'big'))

    length = len(textfile)
    nb_blocks = math.floor(length / (word_size * block_size))

    if verbose:
        print("\n# blocks :", math.ceil(length / (word_size * block_size)))

    for k in my_tqdm(range(nb_blocks)):
        current_block = make_block(textfile[k * step:(k + 1) * step])
        current_block = compute_redundancy(current_block, block_size)
        if noise:
            current_block = add_noise(
                current_block, block_size, args.loss_pattern)
        if verbose:
            print()
            print(f">> Block #{k}")
            print(current_block)
        write_block_in_file(current_block, block_size)

    # Il reste le possible dernier block a mettre dans le fichier
    if len(textfile[nb_blocks * step:]) != 0:
        nb_remaining_symbols = math.ceil(
            len(textfile[nb_blocks * step:]) / word_size)
        last_block = make_last_block(
            textfile[nb_blocks * step:], nb_remaining_symbols)
        last_block = compute_redundancy(last_block, nb_remaining_symbols)
        if noise:
            last_block = add_noise(
                last_block, nb_remaining_symbols, args.loss_pattern)
        if verbose:
            print()
            print(f">> Block #{nb_blocks}")
            print(last_block)
        write_block_in_file(last_block, nb_remaining_symbols)

    outputFile.close()
