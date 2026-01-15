.. exa-pd documentation master file, created by
   sphinx-quickstart on Fri Nov 14 10:59:39 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
========

.. figure:: images/CuZr-phase-diagram.png
   :align: center

   Phase diagram of the Cu-Zr system predicted by **exa-PD** using an EAM-FS potential  

----

Exa-PD is a highly parallelizable workflow for constructing multi-element phase diagrams (PDs). It uses standard sampling techniques—molecular dynamics (MD) and Monte Carlo (MC)—as implemented in the LAMMPS package, to simultaneously sample multiple phases on a fine temperature–composition mesh for free-energy calculations. The workflow uses `Parsl <https://parsl-project.org>`__  as a global controller to manage the MD/MC jobs to achieve massive parallelization with almost ideal scalability. The resulting free energies of both liquid and solid phases (including solid solutions) are then fed to CALPHAD modeling using the PYCALPHAD package for the construction of a multi-element PD.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   tutorial
   workflow
   parsl_config
