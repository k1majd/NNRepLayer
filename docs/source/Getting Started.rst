Getting Started
=====================

Installation
------------
We use the Poetry tool which is a dependency management and packaging tool in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Please follow the installation of poetry `here <https://python-poetry.org/docs/#installation>`_. After you've installed poetry, you can install NNrepLayer by running the following command in the root of the project:

.. code-block:: python
    
    poetry install

Since we also make of the gurobi optimizer, we need to set it up for python. This can be easily done by installing gurobi, the instructions of which can be found here. In order to set it up for python, run the setup.sh file and pass the python identifier and the complete path to folder that contains the setup.py file in the gurobi folder using the following commands:

.. code-block:: python

    sudo ./setup.sh <Python-Identifier> <Path-To-Folder>

For Example, if Python-Identifier was python3.8 and the path to gurobi folder that contains the *setup.py* file is */home/local/user/gurobi950/linux64/build/*, issue the following command:

.. code-block:: python

    sudo ./setup.sh python3.8 /home/local/user/gurobi950/linux64/build/

Usage
------

The repair problem requires 4 components to be defined:

- Neural Network intended for repairs
- Adverserial and Non-Adverserial samples
- Set of constraints
- MIQP Optimizer options

Using some of the pre-build components, a simple repair process might look like:

.. include:: fourComponents.py
   :literal:

In this dummy example, we load the dataset with both adverserial and non-adverserial samples. We then load the trained neural network intended for repair. We then define the constraints such that we want the outputs to stay constrained within the box region defined by Ax < B using the list of *ConstraintClass* objects. After these components are designed, we move on to define the layer to repair and the options for the MIQP solver as weel as additional parameters. We now have all the components ready for repair.

Then we initialize the *Repair* object, compile it by passing the options and run the *repair* methods. And *Voila*, you have teh repaired model ready. We have provided additional experiments in the Demos page. 

For detailed Explanation of running the repair Process, refer to the following pages.


Examples
--------

We have provided one demo which generates a neural network that tries to learn the affine transform relationship. We then repair the neural network to so that the output follows a certain constraint.

We have one notebook that generates the neural network (`Notebook link here <https://www.daittan.com/>`_) and repair this neural network to follow a constraint of the form Ax < b (`Notebook link here <https://www.daittan.com/>`_).

To run the demos using Jupyter Notebook, run the following command:

.. code-block:: python

    poetry run jupyter notebook

