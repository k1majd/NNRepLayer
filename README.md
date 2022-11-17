
# Safe Robot Learning in Assistive Devices through Neural Network Repair 
![teaser](teaser.png)

## Introduction
Assistive robotic devices are a particularly promising field of applica- tion for neural networks (NN) due to the need for personalization and hard-to-model human-machine interaction dynamics. However, NN based estimators and controllers may produce potentially unsafe outputs over previously unseen data points. In this paper, we introduce an algorithm for updating NN control policies to satisfy a given set of formal safety constraints, while also optimizing the original loss function. Given a set of mixed-integer linear constraints, we define the NN repair problem as a Mixed Integer Quadratic Program (MIQP). In extensive experiments, we demonstrate the efficacy of our repair method in generating safe policies for a lower-leg prosthesis. This repository contains the code. Please refer to our [publication at CORL 2022](https://openreview.net/pdf?id=X4228W0QpvN) for more details.

If you find our work useful, please consider citing our paper:
```
@inproceedings{majd2022safe,
    title        = {Safe Robot Learning in Assistive Devices through Neural Network Repair},
    author       = {Majd, Keyvan and Clark, Geoffrey Mitchell and Khandait, Tanmay and Zhou, Siyu and Sankaranarayanan, Sriram and Fainekos, Georgios and Amor, Heni},
    booktitle    = {6th Annual Conference on Robot Learning},
    year         = {2022}
    organization = {PMLR}
}
```

![gif](walking_gif.gif)

## Setup
We use the Poetry tool which is a dependency management and packaging tool in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Please follow the installation of poetry [here](https://python-poetry.org/docs/#installation). After you've installed poetry, you can install NNrepLayer by running the following command in the root of the project:

    poetry install

Since we also make of the gurobi optimizer, we need to set it up for python. This can be easily done by installing gurobi, the instructions of which can be found here. In order to set it up for python, run the setup.sh file and pass the python identifier and the complete path to folder that contains the setup.py file in the gurobi folder using the following commands:

    sudo ./setup.sh <Python-Identifier> <Path-To-Folder>

For Example, if Python-Identifier was python3.8 and the path to gurobi folder that contains the *setup.py* file is */home/local/user/gurobi950/linux64/build/*, issue the following command:

    sudo ./setup.sh python3.8 /home/local/user/gurobi950/linux64/build/

As we used [Pyomo](http://www.pyomo.org) for formulating the optimization, for info on other supported solver we refer to [here](https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html#supported-solvers). Note that the solver in NNRepLayer should be specified through passing optimization `options'.

## Experiments
