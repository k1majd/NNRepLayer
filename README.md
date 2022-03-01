## NNRepLayer: A Python package for theLayer-wise Minimal Local Repair of Deep NeuralNetworks

Deployment of Deep Neural Networks (DNNs) in safety-critical settings requires developing new tools and techniques for the certification and analysis of such systems.
This paper introduces *NNRepLayer*, an optimization-based tool in Python for the Deep Neural Network (DNN) Repair problem. NNRepLayer solves the problem of modifying the weights of a single layer in a trained DNN to address unsafe behavior. NNRepLayer formulates this problem as a Mixed-Integer Quadratic Program (MIQP). Given the trained network, a collection of adversarial and correct sample, and target layer, NNRepLayer repairs the network to enforce the safety requirements on the network. Our tool supports output constraints represented in the form of linear and mixed-integer linear equalities\textbackslash inequalities. NNRepLayer supports input architectures in both TensorFlow, and PyTorch, and also utilizes the well-known mixed-integer programming solvers including Gurobi and CPLEX. This paper presents the main components of NNRepLayer and case studies showing the performance of the tool. 

## Installation

We use the Poetry tool which is a dependency management and packaging tool in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Please follow the installation of poetry [here](https://python-poetry.org/docs/#installation).
After you've installed poetry, you can install *NNrepLayer* by running the following command in the root of the project:

```
poetry install
```

Since we also make of the gurobi optimizer, we need to set it up for python. This can be easily done by installing gurobi, the instructions of which can be found [here](https://www.gurobi.com/documentation/9.5/quickstart_windows/software_installation_guid.html). In order to set it up for python, run the setup.sh file and pass the python identifier and the complete path to folder that contains the setup.py file in the gurobi folder using the following commands:

```
sudo ./setup.sh <Python-Identifier> <Path-To-Folder>
```

For Example, if Python-Identifier was *python3.8* and the path to gurobi folder that contains the setup.py file is */home/local/user/gurobi950/linux64/build/*, issue the following command:

```
sudo ./setup.sh python3.8 /home/local/user/gurobi950/linux64/build/
```


## Examples

We have provided one demo which generates a neural network that tries to learn the affine transform relationship. We then repair the neural network to so that the output follows a certain constraint.

We have one notebook that generates the neural network ([Notebook link here](https://github.com/DaitTan/NNRepLayer/blob/master/NNRepLayer/demos/generate_neural_network_inside.ipynb)) and repair this neural network to follow a constraint of the form Ax < b ([Notebook link here](https://github.com/DaitTan/NNRepLayer/blob/master/NNRepLayer/demos/repair_layer_demo.ipynb)).

To run the demos using *Jupyter Notebook*, run the following command:
```
poetry run jupyter notebook
```