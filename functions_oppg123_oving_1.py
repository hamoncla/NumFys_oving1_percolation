import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit


def create_square_lattice(N, filename): 
    bonds = []
    L = int(np.sqrt(N))
    for i in range(L):
        for j in range(L):
            current = i * L + j 

            right = i * L + ((j + 1) % L) 

            below = ((i + 1) % L) * L + j 

            bonds.append((current, right))
            bonds.append((current, below))

    bonds = np.array(bonds)

    np.savetxt(f'square_lattices/{filename}', bonds, fmt = "%i")

def create_triangular_lattice(N, filename):
    bonds = []
    L = int(np.sqrt(N))

    for i in range(L):
        for j in range(L):
            current = i * L + j

            right = i * L + ((j + 1) % L)

            below = ((i + 1) % L) * L + j

            below_right = ((i + 1) % L) * L + ((j + 1) % L)

            bonds.append((current, right))
            bonds.append((current, below))
            bonds.append((current, below_right))

    bonds = np.array(bonds)

    np.savetxt(f'triangular_lattices/{filename}', bonds, fmt = "%i")

# def create_hexagonal_lattice1(N,filename):
#     bonds = []
#     L = int(np.sqrt(N))

#     for i in range(L):
#         for j in range(L):
#             current = i * L + j

#             if ((j + 1) % L) == 0:
#                 right = i * L + ((j + 1) % L)
#                 below_right = ((i + 1) % L) * L

#                 bonds.append((current, right))
#                 bonds.append((current, below_right))

#             elif (current % L) == 0:
#                 right = i * L + ((j + 1) % L)

#                 bonds.append((current, right))
            
#             elif (current % 2) == 0:
#                 right = i * L + ((j + 1) % L)
#                 below_left = ((i + 1) % L) * L + ((j - 1) % L)

#                 bonds.append((current, right))
#                 bonds.append((current, below_left))
            
#             else:
#                 right = i * L + ((j + 1) % L)
                
#                 bonds.append((current, right))
        
#     bonds = np.array(bonds)

#     np.savetxt(f'hexagonal_lattices/hexagonal_lattice_{L}x{L}.txt', bonds, fmt="%i")

def create_hexagonal_lattice(N, filename):
    bonds = []
    L = int(np.sqrt(N))

    for i in range(N):
        
        if (i // L) % 2 == 0:
            if i % 2 == 0:
                if (i+1) % L != 0:
                    bonds.append((i, i+1))
                else:
                    bonds.append((i, i-L+1))
        
        else:
            if i % 2 == 1:
                if (i + 1) % L != 0:
                    bonds.append((i, i+1))
                else:
                    bonds.append((i, i-L+1))
        
        if (i + L) < N:
            bonds.append((i, i+L))
        else:
            bonds.append((i, i-N+L))

    bonds = np.array(bonds)

    np.savetxt(f'hexagonal_lattices/{filename}', bonds, fmt="%i")



# 2.2: There are 2N bonds in a square lattice, and 

def read_lattice_file(filename, lattice_type):
    bonds = np.loadtxt(f'{lattice_type}_lattices/{filename}', dtype=int)

    N_bonds = bonds.shape[0]
    
    if lattice_type == 'hexagonal':
        N_sites = int(N_bonds/1.5)
    elif lattice_type == 'square':
        N_sites = int(N_bonds/2)
    elif lattice_type == 'triangular':
        N_sites = int(N_bonds/3)

    return bonds, N_bonds, N_sites

def shuffle_list(seed, N_bonds, bond_list):
    shuffled = bond_list.copy()
    
    random.seed(seed)

    for i in range(N_bonds-1):
        r = random.randint(i + 1, N_bonds - 1)

        shuffled[[i,r]] = shuffled[[r,i]]

    return shuffled

def create_sites(N):
    return -np.ones(N, dtype = int)

def initialize_file_list(L, filename):
    create_square_lattice(L, filename)

    bondList, Nbonds, Nsites = read_lattice_file(filename)

    sites = create_sites(L)

    return sites, Nsites, Nbonds, bondList

@jit(nopython=True)
def findRoot(s, sites):
    if sites[s] < 0:
        return s
    else:
        sites[s] = findRoot(sites[s], sites)
        return sites[s]

@jit(nopython=True)
def link(bond, sites, average_s):
    root1 = findRoot(bond[0], sites)
    root2 = findRoot(bond[1], sites)

    if root1 == root2:
        return root1, -sites[root1], average_s

    size1 = -sites[root1]
    size2 = -sites[root2]
    sub_ave_s = (size1 + size2)**2 - (size1**2 + size2**2) 

    if sites[root1] < sites[root2]:
        sites[root1] += sites[root2]
        sites[root2] = root1
        
        return root1, -sites[root1], average_s + sub_ave_s

    else:
        sites[root2] += sites[root1]
        sites[root1] = root2
        return root2, -sites[root2], average_s + sub_ave_s

@jit(nopython=True)
def calculate_values(i, bond, sites, Nbonds, Nsites, largestCluster, largestClusterRootNode, average_s):
    p = i/Nbonds

    newActivatedRoot, lenNewActivated, average_s = link(bond, sites, average_s)

    if lenNewActivated > largestCluster:
        largestCluster = lenNewActivated
        largestClusterRootNode = newActivatedRoot

    pInf = abs(sites[largestClusterRootNode]) / Nsites
    pInfSquared = (pInf)**2

    # sus = Nsites * np.sqrt(np.average(pInfSquared_lst) - (np.average(pInf_lst))**2)

    return p, pInf, pInfSquared, average_s, largestCluster, largestClusterRootNode #, sus


def simulate_loop(sites, Nsites, Nbonds, bondList):
    largestCluster = 0
    largestClusterRootNode = 0
    
    average_s = Nsites
    average_s_lst = np.zeros(Nbonds)

    p_lst = np.zeros(Nbonds)
    pInf_lst = np.zeros(Nbonds)
    pInfSquared_lst = np.zeros(Nbonds)
    # sus_lst = np.zeros(Nbonds)

    for i in range(Nbonds):
        bond = bondList[i]

        p, pInf, pInfSquared, average_s, largestCluster, largestClusterRootNode = calculate_values(i, bond, sites, Nbonds, Nsites,
                                                                                                    largestCluster, largestClusterRootNode, average_s)
                     #pInf_lst, pInfSquared_lst)

        if pInf < 1:
            average_s_lst[i] = (average_s - (Nsites*pInf)**2) / (Nsites*(1 - pInf))
        else:
            average_s_lst[i] = 0

        p_lst[i] = p
        pInf_lst[i] = pInf
        pInfSquared_lst[i] = pInfSquared
        # sus_lst[i] = sus

    return pInf_lst, pInfSquared_lst, average_s_lst #, sus_lst


def calculate_means_lattice(results, N, lattice_type, run_number):
    pInf_lst_all, pInfSquared_lst_all, average_s_lst_all = zip(*results)

    pInf_mean = np.mean(pInf_lst_all, axis=0)
    pInfSquared_mean = np.mean(pInfSquared_lst_all, axis=0)
    average_s_mean = np.mean(average_s_lst_all, axis=0)

    np.save(f'results_{lattice_type}/{run_number}means{N}.npy', [pInf_mean, pInfSquared_mean, average_s_mean])



