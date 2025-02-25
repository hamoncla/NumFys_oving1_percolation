import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
from scipy import stats

@jit(nopython=True)
def suscesptibility(N, pInf_lst, pInfSquared_lst):
    return N * np.sqrt(np.average(pInfSquared_lst) - (np.average(pInf_lst))**2)

lattice_sizes = [10000, 40000, 90000, 160000, 250000, 490000, 810000, 1000000]

def save_susceptibility(lattice_sizes, lattice_type):
    for i in range(len(lattice_sizes)):
        N = lattice_sizes[i]
        pInf_lst, pInfSquared_lst, average_s_lst = np.load(f'results_{lattice_type}/q_{10000}_convoluted{N}.npy')
        suscesptibility_lst = []

        for j in range(1, len(pInf_lst)):
            suscesptibility_lst.append(suscesptibility(N, pInf_lst[:j], pInfSquared_lst[:j]))

        np.save(f'results_{lattice_type}/susceptibility_{N}.npy', suscesptibility_lst)

def find_qc(lattice_sizes, lattice_type, plot=False, q_length=10000, num_sims=200):
    max_R_squared = 0
    qc = 0
    slope = 0
    index = 0

    max_R_squared_1000 = 0
    qc_1000 = 0
    slope_1000 = 0
    index_1000 = 0

    log_pInf_lst_1000 = np.zeros((len(lattice_sizes), q_length))
    log_pInf_lst = np.zeros((len(lattice_sizes), q_length))

    for i in range(len(lattice_sizes)):
        N = lattice_sizes[i]

    
        pInf_lst_1000 = np.load(f'results_{lattice_type}/1000_sims_q_{q_length}_convoluted{N}.npy')[0]

        pInf_lst = np.load(f'results_{lattice_type}/q_{q_length}_convoluted{N}.npy')[0]

        log_pInf_lst_1000[i] = np.log(pInf_lst_1000)
        log_pInf_lst[i] = np.log(pInf_lst)
    
    #log_pInf_lst has shape (8, 10000)
    log_pInf_lst_transposed_1000 = log_pInf_lst_1000.T #shape (10000, 8)
    log_pInf_lst_transposed = log_pInf_lst.T #shape (10000, 8)

    q_lst = np.linspace(0, 1, q_length)
    eps_vals = np.sqrt(np.array(lattice_sizes))
    log_eps_vals = np.log(eps_vals)

    # if plot:
    #     plt.plot(q_lst, log_pInf_lst_transposed)
    #     plt.show()


    for j in range(len(q_lst)):
        if np.exp(log_pInf_lst_transposed[j][-1]) < 0.1:
            continue

        q = q_lst[j]

        regression_1000 = stats.linregress(log_eps_vals, log_pInf_lst_transposed_1000[j])
        regression = stats.linregress(log_eps_vals, log_pInf_lst_transposed[j])

        R_squared = (regression.rvalue)**2

        R_squared_1000 = (regression_1000.rvalue)**2

        if R_squared > max_R_squared:
            max_R_squared = R_squared

            qc = q
            slope = regression.slope
            intercept = regression.intercept

            index = j
        
        if R_squared_1000 > max_R_squared_1000:
            max_R_squared_1000 = R_squared_1000

            qc_1000 = q
            slope_1000 = regression_1000.slope
            intercept_1000 = regression_1000.intercept

            index_1000 = j

    if plot:
        import matplotlib.patheffects as path_effects

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].scatter(log_eps_vals, log_pInf_lst_transposed[index], label='200 simulations')
        axs[0].plot(log_eps_vals, slope * log_eps_vals + intercept, color='red', label='Fit (200 simulations)')
        axs[0].set_ylabel(r'log($P_{\infty}$)')
        axs[0].set_xlabel(r'log($\epsilon$)')
        axs[0].set_title(r'$q_c$ = ' + str(qc))
        axs[0].legend()

        axs[1].scatter(log_eps_vals, log_pInf_lst_transposed_1000[index_1000], label='1000 simulations')
        axs[1].plot(log_eps_vals, slope_1000 * log_eps_vals + intercept_1000, color='red', label='Fit (1000 simulations)')
        axs[1].set_ylabel(r'log($P_{\infty}$)')
        axs[1].set_xlabel(r'log($\epsilon$)')
        axs[1].set_title(r'$q_c$ = ' + str(qc_1000))
        axs[1].legend()

        annotation0 = axs[0].annotate('a)', xy=(0.01, 0.83), xycoords='axes fraction', fontsize=25, color='black')
        #annotation0.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
        annotation1 = axs[1].annotate('b)', xy=(0.01, 0.83), xycoords='axes fraction', fontsize=25, color='black')
        #annotation1.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])

        fig.suptitle(f'Hexagonal Lattice: 200 simulations vs 1000 simulations', fontsize=16)

        plt.tight_layout()
        plt.savefig(f'figures_hexagonal/plot_{lattice_type}_{num_sims}_1000.png', dpi=300)
        plt.close()

    return qc, slope, index

#slope = -beta/nu

# qc_square, minus_beta_over_nu_square, index_square = find_qc(lattice_sizes, 'square')

# qc_tri, minus_beta_over_nu_tri, index_tri = find_qc(lattice_sizes, 'triangular')

qc_hex, minus_beta_over_nu_hex, index_hex = find_qc(lattice_sizes, 'hexagonal', plot = True)

# qc_hex_1000_sims, minus_beta_over_nu_hex_1000_sims, index_hex_1000_sims = find_qc(lattice_sizes, 'hexagonal', num_sims=1000)


def find_gamma_nu(lattice_sizes, lattice_type, q_length=10000, plot=False, num_sims=200):
    log_s_max_lst = np.zeros(len(lattice_sizes))
    q_max = np.zeros(len(lattice_sizes))

    q_lst = np.linspace(0, 1, q_length)

    for i in range(len(lattice_sizes)):
        N = lattice_sizes[i]
        if num_sims == 1000:
            s = np.load(f'results_{lattice_type}/1000_sims_q_{q_length}_convoluted{N}.npy')[2]
        else:
            s = np.load(f'results_{lattice_type}/q_{q_length}_convoluted{N}.npy')[2]

        log_s_max_lst[i] = np.log(max(s))

        max_value = -np.inf
        max_index = 0
        for j in range(len(s)):
            if s[j] > max_value:
                max_value = s[j]
                max_index = j

        q_max[i] = q_lst[max_index]
    
    eps_vals = np.sqrt(np.array(lattice_sizes))
    log_eps_vals = np.log(eps_vals)

    regression = stats.linregress(log_eps_vals, log_s_max_lst)

    gamma_over_nu = regression.slope

    if plot:
        plt.scatter(log_eps_vals, log_s_max_lst)
        plt.plot(log_eps_vals, regression.slope * log_eps_vals + regression.intercept, color='red')
        plt.show()

    return gamma_over_nu, q_max

# gamma_over_nu_square, q_max_square = find_gamma_nu(lattice_sizes, 'square')

# gamma_over_nu_tri, q_max_tri = find_gamma_nu(lattice_sizes, 'triangular')

# gamma_over_nu_hex, q_max_hex = find_gamma_nu(lattice_sizes, 'hexagonal')

# gamma_over_nu_hex_1000_sims, q_max_hex_1000_sims = find_gamma_nu(lattice_sizes, 'hexagonal', num_sims=1000)

def find_nu(q_max_lst, qc, lattice_sizes, plot=False):
    eps_vals = np.sqrt(np.array(lattice_sizes))
    log_eps_vals = np.log(eps_vals)

    regression = stats.linregress(log_eps_vals, np.log(np.abs((q_max_lst - qc))))

    if plot:
        plt.scatter(log_eps_vals, np.log(np.abs((q_max_lst - qc))))
        plt.plot(log_eps_vals, regression.slope * log_eps_vals + regression.intercept, color='red')
        plt.show()

    nu = - 1 / regression.slope

    return nu

def find_gamma(nu, gamma_over_nu):
    return nu * gamma_over_nu

def find_beta(nu, minus_beta_over_nu):
    return -nu * minus_beta_over_nu

# nu_square = find_nu(q_max_square, qc_square, lattice_sizes)
# nu_tri = find_nu(q_max_tri, qc_tri, lattice_sizes)
# nu_hex = find_nu(q_max_hex, qc_hex, lattice_sizes)
# nu_hex_1000_sims = find_nu(q_max_hex_1000_sims, qc_hex_1000_sims, lattice_sizes)

# square_results = [qc_square, find_beta(nu_square, minus_beta_over_nu_square), find_gamma(nu_square, gamma_over_nu_square), nu_square]
# tri_results = [qc_tri, find_beta(nu_tri, minus_beta_over_nu_tri), find_gamma(nu_tri, gamma_over_nu_tri), nu_tri]
# hex_results = [qc_hex, find_beta(nu_hex, minus_beta_over_nu_hex), find_gamma(nu_hex, gamma_over_nu_hex), nu_hex]
# hex_results_1000_sims = [qc_hex_1000_sims, 
#                          find_beta(nu_hex_1000_sims, minus_beta_over_nu_hex_1000_sims), 
#                          find_gamma(nu_hex_1000_sims, gamma_over_nu_hex_1000_sims), 
#                          nu_hex_1000_sims]

def deviation_percent(results, exact_values):
    deviations = []
    for result, exact in zip(results, exact_values):
        deviations.append(abs((result - exact) / exact * 100))
    return deviations

# exact_values_square = [0.5, 5/36, 43/18, 4/3]
# exact_values_tri = [2*np.sin(np.pi/18), 5/36, 43/18, 4/3]
# exact_values_hex = [1 - 2*np.sin(np.pi/18), 5/36, 43/18, 4/3]

# deviations_square = deviation_percent(square_results, exact_values_square)
# deviations_tri = deviation_percent(tri_results, exact_values_tri)
# deviations_hex = deviation_percent(hex_results, exact_values_hex)
# deviations_hex_1000_sims = deviation_percent(hex_results_1000_sims, exact_values_hex)

def write_all_results_to_file(square_results, deviations_square, tri_results, deviations_tri, hex_results, deviations_hex, hex_results_1000_sims, deviations_hex_1000_sims):
    with open('results_all_lattices.txt', 'w') as f:
        f.write('Square Lattice Results:\n')
        f.write(f'Critical point: {square_results[0]} (Deviation: {deviations_square[0]}%)\n')
        f.write(f'Beta: {square_results[1]} (Deviation: {deviations_square[1]}%)\n')
        f.write(f'Gamma: {square_results[2]} (Deviation: {deviations_square[2]}%)\n')
        f.write(f'Nu: {square_results[3]} (Deviation: {deviations_square[3]}%)\n\n')

        f.write('Triangular Lattice Results:\n')
        f.write(f'Critical point: {tri_results[0]} (Deviation: {deviations_tri[0]}%)\n')
        f.write(f'Beta: {tri_results[1]} (Deviation: {deviations_tri[1]}%)\n')
        f.write(f'Gamma: {tri_results[2]} (Deviation: {deviations_tri[2]}%)\n')
        f.write(f'Nu: {tri_results[3]} (Deviation: {deviations_tri[3]}%)\n\n')

        f.write('Hexagonal Lattice Results:\n')
        f.write(f'Critical point: {hex_results[0]} (Deviation: {deviations_hex[0]}%)\n')
        f.write(f'Beta: {hex_results[1]} (Deviation: {deviations_hex[1]}%)\n')
        f.write(f'Gamma: {hex_results[2]} (Deviation: {deviations_hex[2]}%)\n')
        f.write(f'Nu: {hex_results[3]} (Deviation: {deviations_hex[3]}%)\n\n')

        f.write('Hexagonal Lattice Results (1000 sims):\n')
        f.write(f'Critical point: {hex_results_1000_sims[0]} (Deviation: {deviations_hex_1000_sims[0]}%)\n')
        f.write(f'Beta: {hex_results_1000_sims[1]} (Deviation: {deviations_hex_1000_sims[1]}%)\n')
        f.write(f'Gamma: {hex_results_1000_sims[2]} (Deviation: {deviations_hex_1000_sims[2]}%)\n')
        f.write(f'Nu: {hex_results_1000_sims[3]} (Deviation: {deviations_hex_1000_sims[3]}%)\n')

# write_all_results_to_file(square_results, deviations_square, tri_results,
#                           deviations_tri, hex_results, deviations_hex, hex_results_1000_sims, deviations_hex_1000_sims)



















    





