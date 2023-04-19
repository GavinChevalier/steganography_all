import random
import numpy as np
from etc import config
nz = config["size_of_z_latent"]

# 'secret' is a string of 0 or 1
# 'm' is a list of fragment secret; type(fragment secret)=string
def secret_transform(secret,sigma):
    m=[]
    len_s=len(secret)
    if len_s%sigma!=0:
        num_of_supplement_bits=sigma-len_s%sigma
        for i in range(num_of_supplement_bits):
            secret+='0'
    len_of_m=len(secret)//sigma
    for i in range(len_of_m):
        m.append(secret[sigma*i:sigma*i+sigma])
    return m


transfer_table=[
    [0,1],
    [0, 1, 3, 2],
    [0, 1, 3, 2, 6, 4, 5, 7],
    [0, 1, 3, 2, 6, 4, 5, 7, 15, 11, 9, 8, 10, 14, 12, 13],
    [0, 1, 3, 2, 6, 4, 5, 7, 15, 11, 9, 8, 10, 14, 12, 13, 29, 21, 17, 16, 18, 19, 23, 22, 20, 28, 24, 25, 27, 26, 30, 31],
    [0, 1, 3, 2, 6, 4, 5, 7, 15, 11, 9, 8, 10, 14, 12, 13, 29, 21, 17, 16, 18, 19, 23, 22, 20, 28, 24, 25, 27, 26, 30, 31, 63, 47, 39, 35, 33, 32, 34, 38, 36, 37, 45, 41, 40, 42, 43, 59, 51, 49, 48, 50, 54, 52, 53, 55]]


def secret_mapping_v2(secret,sigma,delta,batch_size):
    m=secret_transform(secret,sigma)
    noise=np.empty([batch_size, nz])
    i=0
    for item in m:
        m_ten=int(item,2)

        # if sigma==2:
        #     m_ten=index_2bits.index(m_ten)
        # if sigma == 3:
        #     m_ten=index_3bits.index(m_ten)
        # if sigma==4:
        #     m_ten=index_4bits.index(m_ten)

        m_ten=transfer_table[sigma-1].index(m_ten)

        a=m_ten/(2**(sigma-1))-1+delta
        b=(m_ten+1)/(2**(sigma-1))-1-delta
        # if a>b:
        #     raise Exception(print('a>b'))

        ret = random.uniform(a, b)
        noise[0][i]=ret
        i+=1
    return noise


def secret_mapping_inorder(secret,sigma,delta,batch_size):
    m=secret_transform(secret,sigma)
    noise=np.empty([batch_size, nz])
    i=0
    for item in m:
        m_ten=int(item,2)

        a=m_ten/(2**(sigma-1))-1+delta
        b=(m_ten+1)/(2**(sigma-1))-1-delta
        # if a>b:
        #     raise Exception(print('a>b'))
        ret = random.uniform(a, b)
        noise[0][i]=ret
        i+=1
    return noise
# def random_str(length=300):
#   r_str = ''
#   for i in range(length):
#     r_str+=str(random.randint(0, 1))
#   return r_str


def produce_noise_by_secret_v2(secret,sigma,delta,batch_size):
    # secret=random_str()
    noise_with_secret_np=secret_mapping_v2(secret,sigma,delta,batch_size)
    return noise_with_secret_np

def produce_noise_by_secret_inorder(secret,sigma,delta,batch_size):
    # secret=random_str()
    noise_with_secret_np=secret_mapping_inorder(secret,sigma,delta,batch_size)
    return noise_with_secret_np

