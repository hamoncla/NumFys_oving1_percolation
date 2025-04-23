# Assignment 1 – Computational Physics: Percolation

This project is organized according to the different exercises. Each group of functions is stored in separate files named after the exercises they support. For example:

- `functions_oppg_123_oving_1.py` – Functions for Exercises 1, 2, and 3  
- `functions_oppg_4_oving_1.py` – Functions for Exercise 4
- `functions_oppg_5_oving_1.py` – Functions for Exercise 5
- `visualization.py` – Contains plotting and visualization utilities  
- `run.py` – Functions for running the main simulations  

latticeSizes is always [10000, 40000, 90000, 160000, 250000, 490000, 810000, 1000000] throughout the whole project, and a latticeSize is sometimes refered to as 'N'. 

When specifying the lattice_type for some of the functions you need to write either 'square', 'triangular' or 'hexagonal'.

Also I am sorry for alternating between camelLike and snake_like notation.

## Exercises 1, 2 and 3

These exercises involve creating lattices and simulating percolation through them. The final step is running simulations and calculating averages of relevant properties.

This is done using the function:

```python
run_N_times_calculate_means(run_number, latticeSizes, lattice_type)
```

You can find this function in run.py. Make sure to use the same input values as specified in the report.

⚠️ Note: Running 1000 simulations takes significant time and memory. Ensure that saving the results works correctly. For very large simulations, you might need to manually adjust the loop range to run specific lattice sizes.

After running the simulations, result files will be stored in the results/ folder, containing the averaged values needed for Exercise 4.

To obtain the figure of the snapshots of the percolation you need to run the function:

```python
run_once_visualize_all_types(N = 1000000, seed = 50)
```
this function is also found in run.py and it has a helper function visualize() in visualization.py. 

## Exercise 4

To obtain the arrays with the convoluted values you run the function:

```python
perform_convolution_all_lattice_sizes(latticeSizes, q_length, lattice_type, num_sims)
```
with q_length = 10000. The function is found in the file with functions for exercise 4 and should be executed in this file. Again this code will take a long time for the larger lattices, so make sure the saving works properly. Note: the saving code in the function is a little crude, so double-check it before running, however it is very easy to make more robust.

The plot of the convoluted values is obtained by running the function:
```python
plot_convolution(latticeSizes, lattice_type, q_length = 10000)
```
which is found in the visualization.py file.

## Exercise 5

In this exercise I've combined all the code needed for the figures and results in the report into the function:

```python
find_all_write_to_file(lattice_sizes)
```
so all that's needed is to run this function in the file where it's found.

