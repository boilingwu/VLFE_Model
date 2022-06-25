# VLFE_Model
Codes of very low frequency earthquake models. Datas for the GRL paper "A dynamic rupture source model for very low frequency earthquake signal without detectable non-volcanic tremors"

main_kernel.c is the C program that generates the kernel for BIEM.

main_process_MPIFILE.c is the main C program that calculate the rupture process.

Both program needed to be compiled and run with OpenMPI (command "mpicc", "mpirun"), and support parallel running on multiple cores.
