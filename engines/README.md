# SparseDNN

Push/Pull Engine for Sparse Deep Neural Network Training.

## Data Conversion

Before we run the Push and Pull Engine, we should convert the input date format. The datasets should be stored in "../code/data/".

    make
    g++ Conversion.cpp -fopenmp -mcmodel=medium -o Conversion
    ./Conversion -n 1024 -l 120

## Build

For coarse-grained engines:

    cd pull
    export OPENMP=1
    make

For fine-grained engines:

    cd pull-fine
    export CILK=1
    make

## Run

    ./Lversion3 -n 1024 -l 120 -t 48
