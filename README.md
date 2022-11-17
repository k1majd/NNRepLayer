
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

## Setup


## Experiments
