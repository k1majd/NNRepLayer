Welcome to NNRepLayer
=====================

Introduction
-------------

Deployment of Deep Neural Networks (DNNs) in safety-critical settings requires developing new tools and techniques for the certification and analysis of such systems. This paper introduces NNRepLayer, an optimization-based tool in Python for the Deep Neural Network (DNN) Repair problem. NNRepLayer solves the problem of modifying the weights of a single layer in a trained DNN to address unsafe behavior. NNRepLayer formulates this problem as a Mixed-Integer Quadratic Program (MIQP). Given the trained network, a collection of adversarial and correct sample, and target layer, NNRepLayer repairs the network to enforce the safety requirements on the network. Our tool supports output constraints represented in the form of linear and mixed-integer linear equalities(or inequalities). NNRepLayer supports input architectures in both TensorFlow, and PyTorch, and also utilizes the well-known mixed-integer programming solvers including Gurobi and CPLEX. 

NNRepLayer is a python package for Layer-wise Minimal Local Repair of Deep NeuralNetworks. 

Citation
--------

If you choose to use NNRepLayer in your work, please cite the follwing paper:

.. code-block:: bib

   @misc{https://doi.org/10.48550/arxiv.2109.14041,
   doi = {10.48550/ARXIV.2109.14041},

   url = {https://arxiv.org/abs/2109.14041},
   author = {Majd, Keyvan and Zhou, Siyu and Amor, Heni Ben and Fainekos, Georgios and Sankaranarayanan, Sriram}, 
   keywords = {Machine Learning (cs.LG), Systems and Control (eess.SY), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
   title = {Local Repair of Neural Networks Using Optimization},
   publisher = {arXiv},
   year = {2021},
   copyright = {arXiv.org perpetual, non-exclusive license}
   }


Usage
-----
.. toctree::
   :maxdepth: 2

   Getting Started
   Detailed Run
   Documentation
   