import numpy as np
from etc import config
nz = config["size_of_z_latent"]
# noise_vector is numpy


transfer_table=[
    [0,1],
    [0, 1, 3, 2],
    [0, 1, 3, 2, 6, 4, 5, 7],
    [0, 1, 3, 2, 6, 4, 5, 7, 15, 11, 9, 8, 10, 14, 12, 13],
    [0, 1, 3, 2, 6, 4, 5, 7, 15, 11, 9, 8, 10, 14, 12, 13, 29, 21, 17, 16, 18, 19, 23, 22, 20, 28, 24, 25, 27, 26, 30, 31],
    [0, 1, 3, 2, 6, 4, 5, 7, 15, 11, 9, 8, 10, 14, 12, 13, 29, 21, 17, 16, 18, 19, 23, 22, 20, 28, 24, 25, 27, 26, 30, 31, 63, 47, 39, 35, 33, 32, 34, 38, 36, 37, 45, 41, 40, 42, 43, 59, 51, 49, 48, 50, 54, 52, 53, 55]]

def de_map(noise_vector,sigma):
    # if sigma == 1:
    #     return de_map_1bits_v2(noise_vector)
    # if sigma == 2:
    #     return de_map_2bits_v2(noise_vector)
    # if sigma==3:
    #     return de_map_3bits_v2(noise_vector)
    # if sigma == 4:
    #     return de_map_4bits_v2(noise_vector)

    batch_size = noise_vector.shape[0]
    block=2/(2**sigma)
    de_secret = ''
    for batch in range(batch_size):
        for i in range(nz):
            noise_bit=noise_vector[0][i]
            for j in range(2**sigma):
                if noise_bit >= -1.0+block*j and noise_bit <= -1.0+block*(j+1):
                    de_secret+=bin(transfer_table[sigma-1][j])[2:].rjust(sigma,'0')
                    continue
    return de_secret


def de_map_inorder(noise_vector,sigma):
    batch_size = noise_vector.shape[0]
    block=2/(2**sigma)
    de_secret = ''
    for batch in range(batch_size):
        for i in range(nz):
            noise_bit=noise_vector[0][i]
            for j in range(2**sigma):
                if noise_bit >= -1.0+block*j and noise_bit <= -1.0+block*(j+1):
                    de_secret+=bin(j)[2:].rjust(sigma,'0')
                    continue
    return de_secret



