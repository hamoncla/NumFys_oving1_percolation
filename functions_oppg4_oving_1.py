from scipy.special import gammaln
import numpy as np
from tqdm import tqdm
from numba import jit
import matplotlib.pyplot as plt

def log_MCn(M, n):
    return gammaln(M + 1) - gammaln(n + 1) - gammaln(M - n + 1)  

def calculate_log_MCn(M):
    log_MCn_lst = np.zeros(M+1)

    for n in range(1, M+1):
        log_MCn_lst[n] = log_MCn(M, n)
    
    return log_MCn_lst

@jit(nopython=True)
def calculate_convolution_all_q(M, log_MCn_lst, pInfData, pInfSquaredData, average_sData, pInf0):
    q_lst = np.linspace(0, 1, 1000)
    
    q_log_lst = np.where(q_lst > 0, np.log(q_lst), -1e10)
    q_log_1_lst = np.where(q_lst < 1, np.log(1-q_lst), -1e10)

    n = np.arange(1, M+1)

    log_MCn_lst_reshaped = log_MCn_lst[n].reshape(-1, 1)

    pInfData_reshaped = pInfData[n-1].reshape(-1, 1)
    pInfSquaredData_reshaped = pInfSquaredData[n-1].reshape(-1, 1)
    average_sData_reshaped = average_sData[n-1].reshape(-1, 1)

    log_elements_pInf = log_MCn_lst_reshaped + np.outer(n, q_log_lst) + np.outer(M - n, q_log_1_lst) + np.log(pInfData_reshaped)
    log_elements_pInfSquared = log_MCn_lst_reshaped + np.outer(n, q_log_lst) + np.outer(M - n, q_log_1_lst) + np.log(pInfSquaredData_reshaped)
    log_elements_average_s = log_MCn_lst_reshaped + np.outer(n, q_log_lst) + np.outer(M - n, q_log_1_lst) + np.log(average_sData_reshaped)

    if np.any(np.isnan(log_elements_pInf)) or np.any(np.isinf(log_elements_pInf)):
        print("Warning: NaN or Inf values detected.")

    exp_sum_log_elements_pInf = np.exp(log_elements_pInf).sum(axis=0)
    exp_sum_log_elements_pInfSquared = np.exp(log_elements_pInfSquared).sum(axis=0)
    exp_sum_log_elements_average_s = np.exp(log_elements_average_s).sum(axis=0)
   
    pInfSquared0 = pInf0 ** 2
    average_s0 = 1

    result_pInf = pInf0 * (1 - q_lst) ** M + exp_sum_log_elements_pInf
    result_pInfSquared = pInfSquared0 * (1 - q_lst) ** M + exp_sum_log_elements_pInfSquared
    result_average_s = average_s0 * (1 - q_lst) ** M + exp_sum_log_elements_average_s

    result_lst = [result_pInf, result_pInfSquared, result_average_s]

    return result_lst

@jit(nopython=True)
def calculate_convolution_one_q(M, log_MCn_lst, pInfData, pInfSquaredData, average_sData, pInf0, q_lst, i):
    q = q_lst[i]
    
    q_log = np.log(q) if q > 0 else -np.inf
    q_log_1 = np.log(1-q) if q < 1 else -np.inf

    n = np.arange(1, M+1)

    coeff = np.exp(log_MCn_lst[n] + n * q_log + (M - n) * q_log_1)

    result_pInf = pInf0 * (1 - q) ** M + np.sum(coeff * pInfData[n-1])
    result_pInfSquared = (pInf0 ** 2) * (1 - q) ** M + np.sum(coeff * pInfSquaredData[n-1])
    result_average_s = 1 * (1 - q) ** M + np.sum(coeff * average_sData[n-1])


    return result_pInf, result_pInfSquared, result_average_s

latticeSizes = [10000, 40000, 90000, 160000, 250000, 490000, 810000, 1000000]
q_length = 10000

def perform_convolution_all_lattice_sizes(latticeSizes, q_length, lattice_type):
    for i in tqdm(range(len(latticeSizes))):
        N = latticeSizes[i]
        if lattice_type == 'hexagonal':
            M = int(N * 1.5)
        elif lattice_type == 'square':
            M = N * 2
        elif lattice_type == 'triangular':
            M = N * 3
        
        pInf0 = 1 / N

        q_lst = np.linspace(0,1,q_length)

        pInf_convoluted = np.zeros(q_length)
        pInfSquared_convoluted = np.zeros(q_length)
        average_s_convoluted = np.zeros(q_length)

        data = np.load(f'results_{lattice_type}/1000means{N}.npy')

        pInfData, pInfSquaredData, average_sData = data
        log_MCn_lst = calculate_log_MCn(M)


        for i in tqdm(range(len(q_lst))):
            pInf_convoluted[i], pInfSquared_convoluted[i], average_s_convoluted[i] = calculate_convolution_one_q(M, log_MCn_lst, 
                                                                                         pInfData, pInfSquaredData, average_sData, pInf0, q_lst, i)
            
        np.save(f'results_{lattice_type}/1000_sims_q_{q_length}_convoluted{N}.npy', (pInf_convoluted, pInfSquared_convoluted, average_s_convoluted))


#perform_convolution_all_lattice_sizes(latticeSizes, q_length, 'hexagonal')




