Installing the Template Alignment Compute in LAMMPS
=========================================

The alignment compute is provided in two folders:

  EXTRA-COMPUTE/   CPU implementation
  KOKKOS/          Kokkos implementation, including an updated Install.sh

Installation
------------

1. Copy the alignment-related files from EXTRA-COMPUTE/ into the
   corresponding directory in the LAMMPS source tree:

     cp EXTRA-COMPUTE/* /path/to/lammps/src/EXTRA-COMPUTE/

2. Copy the alignment-related files and updated Install.sh from KOKKOS/
   into the corresponding LAMMPS source directory:

     cp KOKKOS/* /path/to/lammps/src/KOKKOS/

3. Compile LAMMPS with the EXTRA-COMPUTE and KOKKOS packages enabled,
   using your usual build configuration and procedure.

Note: The supplied KOKKOS/Install.sh should replace the existing file
in the LAMMPS source tree. The Kokkos implementation requires a build
configured for the intended Kokkos backend (for example, CUDA for NVIDIA
GPUs).

