# Reinforcement Learning Experiments

This directory contains data when running a reinforcement learning algorithm on
the servers' behaviour. Each subdirectory contains a number of files that 
correspond to data generated using a spceific utility function, where the
servers' different utility functions are:

$$
    U_k^1 = e (\text{\# served}) + (1-e) (\text{idle time})
$$
$$
    U_k^2 = e \frac{\text{\# served}}{\text{\# all individuals}} + (1 - e) \frac{\text{current time - busy time}}{\text{current time}}
$$
$$
    U_k^3 = e (\text{mean service time})_k + (1 - e) (\text{idle proportion})_k
$$
$$
    U_k^4 = e (\frac{1}{\text{mean service time}_k}) + (1 - e) (\text{idle proportion})_k
$$
$$
    U_k^5 = e \frac{\text{\# served}}{\text{\# all individuals}} + (1 - e) (\text{mean service time})_k
$$
$$
    U_k^6 = e \frac{\text{\# served}}{\text{\# all individuals}} + (1 - e) (\frac{1}{\text{mean service time}_k})
$$
$$
    U_k = e (\text{proportion of inds accepted}) + (1 - e) (\text{proportion of server idle time})
$$

Note that not all utility functions are used here.

## Usage

To run experiments in directory `utility_function_3` see `README.md` in the
`utility_function_3` directory.
To run experiments in the other directories naviagte to that directory and
then to the scripts directory and run:

    $ python main.py [e]

where `[e]` is the value of the `e_parameter` variable. This creates 
the files `utilities.csv` and `rates.csv` that are stored in
`results/[e_parameter]/`.
