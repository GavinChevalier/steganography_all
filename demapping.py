import numpy as np
from etc import config
nz = config["size_of_z_latent"]
# noise_vector is numpy

def de_map_3bits(noise_vector):
    batch_size = noise_vector.shape[0]
    leng = noise_vector.shape[1]

    sigma=3

    de_secret=np.empty(leng*sigma, dtype=int)

    for batch in range(batch_size):
        i=0
        for ii in range(nz):
            noise_bit=noise_vector[0][ii][0][0]
            if noise_bit >=-1.0 and noise_bit < -0.750:
                de_secret[i] = 0
                de_secret[i + 1] = 0
                de_secret[i + 2] = 0
                i+=3
                continue
            if noise_bit >= -0.750 and noise_bit < -0.500:
                de_secret[i] = 0
                de_secret[i + 1] = 0
                de_secret[i + 2] = 1
                i += 3
                continue
            if noise_bit >= -0.500 and noise_bit < -0.250:
                de_secret[i] = 0
                de_secret[i + 1] = 1
                de_secret[i + 2] = 0
                i += 3
                continue
            if noise_bit >= -0.250 and noise_bit < -0.000:
                de_secret[i] = 0
                de_secret[i + 1] = 1
                de_secret[i + 2] = 1
                i += 3
                continue
            if noise_bit >= 0.000 and noise_bit < 0.250:
                de_secret[i] = 1
                de_secret[i + 1] = 0
                de_secret[i + 2] = 0
                i += 3
                continue
            if noise_bit >= 0.250 and noise_bit < 0.500:
                de_secret[i] = 1
                de_secret[i + 1] = 0
                de_secret[i + 2] = 1
                i += 3
                continue
            if noise_bit >= 0.500 and noise_bit < 0.750:
                de_secret[i] = 1
                de_secret[i + 1] = 1
                de_secret[i + 2] = 0
                i+=3
                continue
            if noise_bit >= 0.750 and noise_bit <= 1:
                de_secret[i] = 1
                de_secret[i + 1] = 1
                de_secret[i + 2] = 1
                i += 3
                continue
    return de_secret


# def de_map(noise_vector,sigma):
#     batch_size = noise_vector.shape[0]
#     de_secret=''
#
#     for batch in range(batch_size):
#         for i in range(nz):
#             noise_num=noise_vector[0][i][0][0]
#             m=((noise_num+1)*2**(sigma-1)+(noise_num-1)*2**(sigma-1))/2
#             de_secret+=bin(int(m))[2:].rjust(sigma,'0')
#     return de_secret


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



def de_map_1bits_v2(noise_vector):
    batch_size = noise_vector.shape[0]
    de_secret=''

    for batch in range(batch_size):
        for i in range(nz):
            noise_bit=noise_vector[0][i]
            if noise_bit >=-1.0 and noise_bit < 0:
                de_secret+='0'
                continue
            if noise_bit >= 0 and noise_bit <= 1:
                de_secret+='1'
                continue
    return de_secret

def de_map_2bits_v2(noise_vector):
    batch_size = noise_vector.shape[0]
    de_secret=''

    for batch in range(batch_size):
        for i in range(nz):
            noise_bit=noise_vector[0][i]
            if noise_bit >=-1.0 and noise_bit < -0.500:
                de_secret+='00'
                continue
            if noise_bit >= -0.500 and noise_bit < -0.000:
                de_secret+='01'
                continue
            if noise_bit >= 0.000 and noise_bit < 0.500:
                de_secret+='11'
                continue
            if noise_bit >= 0.500 and noise_bit <= 1:
                de_secret+='10'
                continue
    return de_secret

def de_map_3bits_v2(noise_vector):
    batch_size = noise_vector.shape[0]
    de_secret=''

    for batch in range(batch_size):
        for i in range(nz):
            noise_bit=noise_vector[0][i]
            if noise_bit >=-1.0 and noise_bit < -0.750:
                de_secret+='000'
                continue
            if noise_bit >= -0.750 and noise_bit < -0.500:
                de_secret+='001'
                continue
            if noise_bit >= -0.500 and noise_bit < -0.250:
                de_secret+='011'
                continue
            if noise_bit >= -0.250 and noise_bit < -0.000:
                de_secret+='010'
                continue
            if noise_bit >= 0.000 and noise_bit < 0.250:
                de_secret+='110'
                continue
            if noise_bit >= 0.250 and noise_bit < 0.500:
                de_secret+='100'
                continue
            if noise_bit >= 0.500 and noise_bit < 0.750:
                de_secret+='101'
                continue
            if noise_bit >= 0.750 and noise_bit <= 1:
                de_secret+='111'
                continue
    return de_secret

def de_map_4bits_v2(noise_vector):
    batch_size = noise_vector.shape[0]
    de_secret=''
    for batch in range(batch_size):
        for ii in range(nz):
            noise_bit=noise_vector[0][ii]
            if noise_bit >=-1.0 and noise_bit < -1+0.125:
                de_secret+='0000'
                continue
            if noise_bit >=-1.0+0.125 and noise_bit < -1+0.125*2:
                de_secret+='0001'
                continue
            if noise_bit >=-1.0+0.125*2 and noise_bit < -1+0.125*3:
                de_secret+='0011'
                continue
            if noise_bit >=-1.0+0.125*3 and noise_bit < -1+0.125*4:
                de_secret+='0010'
                continue
            if noise_bit >=-1.0+0.125*4 and noise_bit < -1+0.125*5:
                de_secret+='0110'
                continue
            if noise_bit >=-1.0+0.125*5 and noise_bit < -1+0.125*6:
                de_secret+='0100'
                continue
            if noise_bit >=-1.0+0.125*6 and noise_bit < -1+0.125*7:
                de_secret+='0101'
                continue
            if noise_bit >=-1.0+0.125*7 and noise_bit < -1+0.125*8:
                de_secret+='0111'
                continue
            if noise_bit >=-1.0+0.125*8 and noise_bit < -1+0.125*9:
                de_secret+='1111'
                continue
            if noise_bit >=-1.0+0.125*9 and noise_bit < -1+0.125*10:
                de_secret+='1011'
                continue
            if noise_bit >=-1.0+0.125*10 and noise_bit < -1+0.125*11:
                de_secret+='1001'
                continue
            if noise_bit >=-1.0+0.125*11 and noise_bit < -1+0.125*12:
                de_secret+='1000'
                continue
            if noise_bit >=-1.0+0.125*12 and noise_bit < -1+0.125*13:
                de_secret+='1010'
                continue
            if noise_bit >=-1.0+0.125*13 and noise_bit < -1+0.125*14:
                de_secret+='1110'
                continue
            if noise_bit >=-1.0+0.125*14 and noise_bit < -1+0.125*15:
                de_secret+='1100'
                continue
            if noise_bit >=-1.0+0.125*15 and noise_bit <= 1:
                de_secret+='1101'
                continue
    return de_secret




def de_map_4bits_inorder(noise_vector):
    batch_size = noise_vector.shape[0]
    de_secret=''
    for batch in range(batch_size):
        for ii in range(nz):
            noise_bit=noise_vector[0][ii]
            if noise_bit >=-1.0 and noise_bit < -1+0.125:
                de_secret+='0000'
                continue
            if noise_bit >=-1.0+0.125 and noise_bit < -1+0.125*2:
                de_secret+='0001'
                continue
            if noise_bit >=-1.0+0.125*2 and noise_bit < -1+0.125*3:
                de_secret+='0010'
                continue
            if noise_bit >=-1.0+0.125*3 and noise_bit < -1+0.125*4:
                de_secret+='0011'
                continue
            if noise_bit >=-1.0+0.125*4 and noise_bit < -1+0.125*5:
                de_secret+='0100'
                continue
            if noise_bit >=-1.0+0.125*5 and noise_bit < -1+0.125*6:
                de_secret+='0101'
                continue
            if noise_bit >=-1.0+0.125*6 and noise_bit < -1+0.125*7:
                de_secret+='0110'
                continue
            if noise_bit >=-1.0+0.125*7 and noise_bit < -1+0.125*8:
                de_secret+='0111'
                continue
            if noise_bit >=-1.0+0.125*8 and noise_bit < -1+0.125*9:
                de_secret+='1000'
                continue
            if noise_bit >=-1.0+0.125*9 and noise_bit < -1+0.125*10:
                de_secret+='1001'
                continue
            if noise_bit >=-1.0+0.125*10 and noise_bit < -1+0.125*11:
                de_secret+='1010'
                continue
            if noise_bit >=-1.0+0.125*11 and noise_bit < -1+0.125*12:
                de_secret+='1011'
                continue
            if noise_bit >=-1.0+0.125*12 and noise_bit < -1+0.125*13:
                de_secret+='1100'
                continue
            if noise_bit >=-1.0+0.125*13 and noise_bit < -1+0.125*14:
                de_secret+='1101'
                continue
            if noise_bit >=-1.0+0.125*14 and noise_bit < -1+0.125*15:
                de_secret+='1110'
                continue
            if noise_bit >=-1.0+0.125*15 and noise_bit <= 1:
                de_secret+='1111'
                continue
    return de_secret

