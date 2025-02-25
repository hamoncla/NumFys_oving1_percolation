from functions_oppg123_oving_1 import *

def visualize(L, sites, Nsites, largestClusterRootNode, image_index):
    grid = np.zeros(Nsites)

    for j in range(Nsites):
        root = findRoot(j, sites)

        if root == largestClusterRootNode:
            grid[j] = 1

    grid = grid.reshape((L, L))

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='Greys', vmin=0,vmax=1)
    fig.savefig(f"1000000_grid_{image_index}.jpg")
    plt.close()

    #take picture
        # image_index = 1
        # target_p_values = [0.25, 0.48, 0.5, 0.52, 0.75]
        # epsilon = 0.00000000002  # tol
        # if any(abs(p - target_p) < epsilon for target_p in target_p_values):
        #     visualize(L, sites, Nsites, largestClusterRootNode, image_index)
        #     image_index += 1

def plot_pInf_s(latticeSizes, AllpInfMean_lst, Allaverage_s_mean_lst, savefile_name):
    colors = ['purple', 'darkgreen', 'skyblue', 'orange', 'gold', 'blue', 'red', 'black']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    for i, N in enumerate(latticeSizes):
        p_lst = np.linspace(0, 1, len(AllpInfMean_lst[i]))
        ax1.plot(p_lst, AllpInfMean_lst[i], color=colors[i], label=f'N={N}')
    ax1.set_xlabel('p')
    ax1.set_ylabel(r'$P_{\infty}$')
    ax1.set_xlim(0.4, 0.6) 
    ax1.set_ylim(0, 1)
    ax1.legend()
    
    ax2 = axes[1]
    for i, N in enumerate(latticeSizes):
        p_lst = np.linspace(0, 1, len(Allaverage_s_mean_lst[i]))
        ax2.plot(p_lst, Allaverage_s_mean_lst[i], color=colors[i], label=f'N={N}')
    ax2.set_xlabel('p')
    ax2.set_ylabel(r'$\langle s \rangle$')
    ax2.set_xlim(0.4, 0.6) 
    ax2.legend()

    fig.savefig(f'figures/{savefile_name}', dpi = 300)

def plot_convolution(latticeSizes, q_length):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#000000']

    for idx, N in enumerate(latticeSizes):
        pInf_convoluted, pInfSquared_convoluted, average_s_convoluted = np.load(f'results_square/q_{q_length}_convoluted{N}.npy')

        q_lst = np.linspace(0, 1, q_length)

        axs[0].plot(q_lst, pInf_convoluted, label=f'N={N}', color=colors[idx])
        axs[1].plot(q_lst, average_s_convoluted, label=f'N={N}', color=colors[idx])

    axs[0].set_xlabel('q')
    axs[0].set_ylabel(r'${P_{\infty}}$')
    axs[0].set_xlim(0.4, 0.6)
    axs[0].set_ylim(0, 1)
    axs[0].set_title(r'Convoluted ${P_{\infty}}$')
    axs[0].legend()
    axs[0].grid(False)
    axs[0].set_xticks([0.4, 0.45, 0.5, 0.55, 0.6])

    axs[1].set_xlabel('q')
    axs[1].set_ylabel(r'$\langle s \rangle$')
    axs[1].set_xlim(0.4, 0.6)
    axs[1].set_title(r'Convoluted $\langle s \rangle$')
    axs[1].legend()
    axs[1].grid(False)
    axs[1].set_xticks([0.4, 0.45, 0.5, 0.55, 0.6])

    plt.tight_layout()
    plt.show()

    fig.savefig('figures/convoluted_plots_square.png', dpi = 300)

def plot_means_pInf_avgS(latticeSizes, lattice_type):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    for size in latticeSizes:
        f = np.load(f'results_{lattice_type}/means{size}.npy', allow_pickle=True)
        
        pInfMean = f[0]
        average_s_mean = f[2]
        
        p_lst = np.linspace(0, 1, len(pInfMean))
        
        axs[0].plot(p_lst, pInfMean, label=f'N={size}')
        axs[1].plot(p_lst, average_s_mean, label=f'N={size}')

    axs[0].set_xlabel('p')
    axs[0].set_ylabel(r'$P_{\infty}$')
    #axs[0].set_title('Plot of pInfMean vs p')
    axs[0].set_xlim(0.4, 0.6)
    axs[0].legend()
    axs[0].set_xticks([0.4, 0.45, 0.5, 0.55, 0.6])
    axs[0].grid(False)

    axs[1].set_xlabel('p')
    axs[1].set_ylabel(r'$\langle s \rangle$')
    #axs[1].set_title('Plot of average_s_mean vs p')
    axs[1].set_xlim(0.4, 0.6)
    axs[1].set_xticks([0.4, 0.45, 0.5, 0.55, 0.6])
    axs[1].legend()
    axs[1].grid(False)

    plt.tight_layout()
    
    fig.savefig(f'figures_{lattice_type}/means_{lattice_type}.png', dpi=300)

def plot_susceptibility(lattice_sizes, lattice_type):
    for i in range(0, len(lattice_sizes)):
        N = lattice_sizes[i]
        susceptibility_lst = np.load(f'results_{lattice_type}/susceptibility_{N}.npy')
        q_values = np.arange(1, len(susceptibility_lst) + 1)

        plt.plot(q_values, susceptibility_lst, label=f'N={N}')

    plt.xlabel('q')
    plt.ylabel('Susceptibility')
    plt.title('Susceptibility vs q')
    plt.legend()
    plt.show()

def plot_alot(lattice_sizes, lattice1, lattice2):
    fig, axs = plt.subplots(3, 2, figsize=(12, 9))

    for i in range(2):
        for j in range(2):
            ax = axs[i, j]
            if j == 0:
                lattice_type = lattice1
            else:
                lattice_type = lattice2

            for k in range(len(lattice_sizes)):
                N = lattice_sizes[k]
                pInf_lst, pInfSquared_lst, average_s_lst = np.load(f'results_{lattice_type}/q_10000_convoluted{N}.npy')

                q_values = np.linspace(0, 1, len(pInf_lst))

                if i == 0:
                    ax.plot(q_values, pInf_lst, label=f'N={N}')
                    ax.set_ylabel(r'$P_{\infty}$', fontsize=15)
                else:
                    ax.plot(q_values, average_s_lst, label=f'N={N}')
                    ax.set_ylabel(r'$\langle s \rangle$', fontsize=15)

                ax.set_xlabel('q')
                ax.legend()

    axs[0, 0].set_title(f'Convoluted values for {lattice1}', fontsize=20)
    axs[0, 1].set_title(f'Convoluted values for {lattice2}', fontsize=20)
    axs[0, 0].set_xlim(0.4, 0.6)
    axs[1, 0].set_xlim(0.4, 0.6)
    axs[0, 1].set_xlim(0.55, 0.75)
    axs[1, 1].set_xlim(0.55, 0.75)

    for j in range(2):
        ax = axs[2, j]
        if j == 0:
            lattice_type = lattice1
        else:
            lattice_type = lattice2

        for k in range(len(lattice_sizes)):
            N = lattice_sizes[k]
            susceptibility_lst = np.load(f'results_{lattice_type}/susceptibility_{N}.npy')
            q_values = np.linspace(0, 1, len(susceptibility_lst))

            ax.plot(q_values, susceptibility_lst, label=f'N={N}')
            ax.set_ylabel(r'$\chi$', fontsize=15)
            ax.set_xlabel('q')
            ax.legend()

    plt.tight_layout()
    #plt.show()
    fig.savefig(f'figures_{lattice1}/convoluted_plots_{lattice1}_{lattice2}_sus.png', dpi = 300)

lattice_sizes = [10000, 40000, 90000, 160000, 250000, 490000, 810000, 1000000]

import matplotlib.patheffects as path_effects

def load_and_plot_images(image_indices, p_vals):
    fig, axs = plt.subplots(1, len(image_indices), figsize=(12, 5))

    annotations = ['a)', 'b)', 'c)']

    for i, image_index in enumerate(image_indices):
        img = plt.imread(f'figures_square/snap_p_{image_index}.png')
        axs[i].imshow(img, cmap='Greys', vmin=0, vmax=1)
        axs[i].set_title(f'p = {p_vals[i]}', fontsize=20)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        annotation = axs[i].annotate(annotations[i], xy=(0.01, 0.92), xycoords='axes fraction', fontsize=25, color='red')
        annotation.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])

    plt.tight_layout()
    fig.savefig('figures_square/snapshots.png', dpi=300)





