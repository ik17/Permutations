# Permutations

Comparing the performance of parallelised and non-parallelised implementations of permutations, with or without randomization. 

This repository contains 4 different implementations, and they are:

1. B-S - Basic sequential 
2. B-P - Basic parallel 
3. LV-S - Las Vegas sequential
4. LV-P - Las Vegas parallel 

To test these programs out, it's suggested to install CUDA Toolkit which contains required compiler, along with Visual Studio, which can use said compiler to compile the code.

To make sure you're using correct compiler, you need to link your solution to CUDA Toolkit Custom Directory, which can be found by going to application Properties -> CUDA C/C++ -> CUDA Toolkit Custom Dir.
