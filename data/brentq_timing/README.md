# Brent's Algorithm timing

This directory contains experiments looking to understand the effect of the
tolerance measures for Brent's algorithm. This is used to find the routing
strategy of the third player and requires heavy computational effort as the
steady state of a Markov chain needs to be found.

## Usage

To run the experiments:

    $ python main.py

This creates `main.csv` which is of the form:

    repetition, tolerance, problem parameters, time
