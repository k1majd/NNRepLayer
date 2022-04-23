from dataclasses import dataclass
from typing import List, Union
import numpy as np
import numpy.typing as npt

def tf2_get_weights(mlp):
    """_summary_

    Args:
        mlp (_type_): _description_

    Returns:
        _type_: _description_
    """
    return mlp.get_weights()

def pt_get_weights(mlp):
    """_summary_

    Args:
        mlp (_type_): _description_

    Returns:
        _type_: _description_
    """
    params = []
    for param_tensor in mlp.state_dict():
        param = mlp.state_dict()[param_tensor].numpy()
        if "weight" in param_tensor.split("."):
            params.append(param.T)
        else:
            params.append(param)
    
    return params

def tf2_get_architecture(model):
    """Extracts the arhitecture of tf model (tf2)

    Args:
        model (_type_): _description_
    """
    if model is None:
        raise TypeError("Model cannot be None")
    else:
        architecture = []
        for lnum, lay in enumerate(model.layers):
            if len(lay.weights)!=0:
                architecture.append(lay.input.shape[1])
                if lnum == len(model.layers) - 1:
                    architecture.append(lay.output.shape[1])
    return architecture

def pt_get_architecture(model):
    """Extracts the arhitecture of tf model (tf2)

    Args:
        model (_type_): _description_
    """
    if model is None:
        raise TypeError("Model cannot be None")
    else:
        architecture = []
        for lay in model.state_dict():
            if "weight" in set(lay.split(".")):
                architecture.append(model.state_dict()[lay].size()[1])
                fin_layer = lay

        architecture.append(model.state_dict()[fin_layer].size()[0])
    return architecture

def generate_inside_constraints(name:str, A:npt.NDArray, b:npt.NDArray):
    if A is None:
        raise ValueError("A cannot be empty")
    if b is None:
        raise ValueError("b cannot be empty")
    if len(b.shape) != 2:
        raise ValueError("Shape of b must be N x 1 dimension")
    A_m, A_n = A.shape
    b_n, b_p = b.shape

    if A_m != b_n:
        raise ValueError("First dimension of A (= {}) should be equal to be first dimesnion of b(= {}).".format(
                        A_m, b_n
        ))
    if b_p != 1:
        raise ValueError("Second dimension of b (={}) should be 1".format(b_p))

    def_string = ""
    add_attr_list = []
    for i in range(A_m):
        single_def_string = ""
        temp_string = []

        for j in range(A_n):
            t_st = "({})*getattr(model, x_l)[i, {}]".format(A[i, j], j)
            
            temp_string.append(t_st)
        add_attr_string = "setattr(self.model, '{}{}'+str(l),pyo.Constraint(range(m), rule={}{}))".format(
            name, i, name, i
        )
        add_attr_list.append(add_attr_string)
        return_string = "(" + " + ".join(temp_string) + "  - ({}) <= 0)".format(b[i, 0])
        single_def_string = (
            "def {}{}(model, i):\n\treturn ".format(name, i) + return_string
        )
        def_string = def_string + single_def_string + "\n\n"

    attr_string = "\n".join(add_attr_list)

    def_string += attr_string

    return def_string


def generate_outside_constraints(name:str, A, B):
    if not A:
        raise ValueError("A cannot be empty")
    if not B:
        raise ValueError("B cannot be empty")
    if len(A) != len(B):
        raise ValueError(f"Length Mismatch betweeb A and B. Length of A is {len(A)} and Length of B is {len(B)}.")
    
    def_string = ""
    add_attr_list = []
    list_alleq = []
    for a, b in zip(A, B):

        list_eq = []
        iterate_1 = 0

        for a_, b_ in zip(a, b):
            str_eq = ""
            for iterate_2, val in enumerate(a_):

                if val != 0:
                    str_eq += (
                        "("
                        + str(val)
                        + " * "
                        + "getattr(model, x_l)[i"
                        + ", "
                        + str(iterate_2)
                        + "]) + "
                    )

            iterate_1 += 1
            str_eq_whole = ""
            if str_eq != "":
                str_eq_whole = str_eq[0:-2] + " <= " + str(b_[0])
            list_eq.append(str_eq_whole)
        list_alleq.append(list_eq)

    list_alleq = np.array(list_alleq).T

    str_eq_all = "["
    for iterate in range(list_alleq.shape[0]):
        str_eq = "["
        for iterate_2 in range(list_alleq.shape[1]):
            if list_alleq[iterate, iterate_2] != "":
                str_eq += list_alleq[iterate, iterate_2] + ","

        str_eq = str_eq[0:-1] + "]"

        if iterate != list_alleq.shape[0] - 1:
            str_eq_all += str_eq + ",\n\t"
        else:
            str_eq_all += str_eq + "]"

    single_def_string = "def " + name + "0(model, i):\n\treturn " + str_eq_all + "\n"
    single_def_string += (
        "setattr(self.model, '"
        + name
        + "0' +str(l), pyg.Disjunction(range(m), rule={}{}))".format(name, str(0))
    )

    return single_def_string


@dataclass
class ConstraintsClass:
    constraint_type: str
    A: Union[List[npt.NDArray], npt.NDArray]
    B: Union[List[npt.NDArray], npt.NDArray]


def generate_output_constraints(constraint):
    outside_count = 0
    inside_count = 0
    out_string = ""
    in_string = ""
    for i in constraint:

        # print(inside_count, outside_count)
        if i.constraint_type == "outside":
            temp_string = generate_outside_constraints(
                i.constraint_type + str(outside_count), i.A, i.B
            )
            # print(temp_string)
            # print("*************************************")
            out_string = out_string + temp_string + "\n\n"
            outside_count = outside_count + 1
        elif i.constraint_type == "inside":
            temp_string = generate_inside_constraints(
                i.constraint_type + str(outside_count), i.A, i.B
            )
            # print(temp_string)
            # print("*************************************")
            in_string = in_string + temp_string + "\n\n"
            inside_count = inside_count + 1

    fin = out_string + "\n\n" + in_string
    return fin



def give_mse_error(data1: npt.NDArray, data2:npt.NDArray):
    """return the mean square error of data1-data2 samples

    Args:
        data1 (ndarray): predicted targets
        data2 (ndarray): original targets

    Returns:
        float: mse error
    """
    if data1 is None or data2 is None:
        raise TypeError("Data cannot be None")
    row_1, col_1 = np.array(data1).shape
    row_2, col_2 = np.array(data2).shape
    if row_1 != row_2 or col_1 != col_2:
        raise ValueError(f"Possible row mismatch. Data 1 has shape {np.array(data1).shape} and Data 2 has shape {np.array(data2).shape}")
        
    _squared_sum = 0
    for i in range(row_1):
        for j in range(col_1):
            _squared_sum += (data1[i, j] - data2[i, j]) ** 2

    return _squared_sum / row_1
