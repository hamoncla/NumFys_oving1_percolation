from functions_oppg123_oving_1 import *
from visualization import *
from functions_oppg4_oving_1 import *
import os

latticeSizes = [10000, 40000, 90000, 160000, 250000, 490000, 810000, 1000000]



def run_once(L, filename, seed):

    sites, Nsites, Nbonds, bondList = initialize_file_list(L, filename)

    bondList = shuffle_list(bondList, seed)

    p_lst, pInf_lst, pInfSquared_lst, average_s_lst = simulate_loop(sites, Nsites, Nbonds, bondList)

    return p_lst, pInf_lst, pInfSquared_lst, average_s_lst
    
def run_N_times_calculate_means(run_number, latticeSizes, lattice_type):
    
    for i in range(7,8):
        N = latticeSizes[i]
        L = int(np.sqrt(N))
        filename = f'{lattice_type}_lattice_{L}x{L}.txt'

        if not os.path.exists(f'{lattice_type}_lattices/{filename}'):
            if lattice_type == 'hexagonal':
                create_hexagonal_lattice(N, filename)
            elif lattice_type == 'square':
                create_square_lattice(N, filename)
            elif lattice_type == 'triangular':
                create_triangular_lattice(N, filename)
        else:
            print(f"File {filename} already exists, skipping creation.")

        bondList, Nbonds, Nsites = read_lattice_file(filename, lattice_type)

        results = []

        for j in tqdm(range(run_number)):
            bondList = shuffle_list(j, Nbonds, bondList)
            sites = create_sites(N)

            pInf_lst, pInfSquared_lst, average_s_lst = simulate_loop(sites, Nsites, Nbonds, bondList)

            results.append([pInf_lst, pInfSquared_lst, average_s_lst])
        
        calculate_means_lattice(results, N, lattice_type, run_number)






