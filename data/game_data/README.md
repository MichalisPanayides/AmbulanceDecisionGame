# Generate ambulance game data

This directory contains a script that aims to generate numerous data for the
ambulance game. In particular the data that are of greatest interest are the
routing matrix, and the payoff matrices.

## Usage

To run the experiments run in the command line:

    $ python main.py

This creates the `data` directory that is structured in the following way:

    |-- _parameters/
        |-- main.csv
        |-- README.md
    |-- <set_of_parameters_1>/
        |-- main.csv
        |-- main.npz
        |-- README.md
    |-- <set_of_parameters_2>/
        |-- main.csv
        |-- main.npz
        |-- README.md
    |-- <set_of_parameters_3>/
        |-- main.csv
        |-- main.npz
        |-- README.md

