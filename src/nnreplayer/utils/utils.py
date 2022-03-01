
def mlp_get_weights(mlp):
    weights = []
    bias = []
    exec_string = ""
    for iterate in range(len(mlp.layers)):
        w_var = "w"+str(iterate)+"_orig"
        b_var = "b"+str(iterate)+"_orig"
        vars()[w_var] = mlp.layers[iterate].kernel.numpy()
        vars()[b_var] = mlp.layers[iterate].bias.numpy()
        weights.append(vars()[w_var])
        bias.append(vars()[b_var])
        if iterate == len(mlp.layers)-1:
            exec_string = exec_string + "w"+str(iterate)+"_orig, " + "b"+str(iterate)+"_orig"
        else: 
            exec_string = exec_string + "w"+str(iterate)+"_orig, " + "b"+str(iterate)+"_orig, "

    print(list(eval(exec_string)))
    return list(eval(exec_string))


def mlp_set_weights(mlp, mlp_orig_weights):
    iterate = 0
    for j in range(len(mlp.layers)):
        mlp.layers[j].weights = mlp_orig_weights[iterate]
        iterate = iterate+1
        mlp.layers[j].bias = mlp_orig_weights[iterate]
        iterate = iterate+1
    return mlp

# def generate_constraints(A,b):

#     # assert len(b.shape) == 2, 'Shape of b must be N x 1 dimension'
#     # A_m, A_n = A.shape
#     # b_n, b_p = b.shape
#     # assert A_m == b_n, "First dimension of A (= {}) should be equal to be first dimesnion of b(= {}).".format(A_m, b_n)
#     # assert b_p == 1, "Second dimension of b (={}) should be 1".format(b_p)

#     # def_string = ""
#     # add_attr_list = []
#     # for i in range(A_m):
#     #     single_def_string = ""
#     #     temp_string = []
        
#     #     for j in range(A_n):
#     #         t_st = "{}*getattr(model, x_l)[i, {}]".format(A[i,j],j)
#     #         # print(t_st)
#     #         temp_string.append(t_st)
#     #     add_attr_string = "setattr(self.model, 'keep_inside_constraint{}'+str(l),pyo.Constraint(range(m), rule=constraint_inside{}))".format(i,i)
#     #     add_attr_list.append(add_attr_string)
        
#     #     return_string = "(" + " + ".join(temp_string) + "  - {} <= 0)".format(b[i,0])
        
#     #     single_def_string = "def constraint_inside{}(model, i): return ".format(i) + return_string 
#     #     def_string = def_string + single_def_string + "\n\n"
    
#     # attr_string = "\n".join(add_attr_list)
    
#     # def_string += attr_string
#     # print(def_string)

#     assert len(b.shape) == 2, 'Shape of b must be N x 1 dimension'
#     A_m, A_n = A.shape
#     b_n, b_p = b.shape
#     assert A_m == b_n, "First dimension of A (= {}) should be equal to be first dimesnion of b(= {}).".format(A_m, b_n)
#     assert b_p == 1, "Second dimension of b (={}) should be 1".format(b_p)

#     def_string = ""
#     add_attr_list = []
#     for i in range(A_m):
#         single_def_string = ""
#         temp_string = []
        
#         for j in range(A_n):
#             t_st = "A[{},{}]*getattr(model, x_l)[i, {}]".format(i,j,j,i)
#             temp_string.append(t_st)
#         add_attr_string = "setattr(self.model, 'keep_inside_constraint{}'+str(l),pyo.Constraint(range(m), rule=constraint_inside{}))".format(i,i)
#         add_attr_list.append(add_attr_string)
#         return_string = "(" + " + ".join(temp_string) + "  - b[{}] <= 0)".format(i)
#         single_def_string = "def constraint_inside{}(model, i):\n\treturn ".format(i) + return_string 
#         def_string = def_string + single_def_string + "\n\n"
    
#     attr_string = "\n".join(add_attr_list)
    
#     def_string += attr_string

#     return def_string

from dataclasses import dataclass
from typing import Any
import numpy as np





def generate_inside_constraints(name, A,b):
    assert len(b.shape) == 2, 'Shape of b must be N x 1 dimension'
    A_m, A_n = A.shape
    b_n, b_p = b.shape
    assert A_m == b_n, "First dimension of A (= {}) should be equal to be first dimesnion of b(= {}).".format(A_m, b_n)
    assert b_p == 1, "Second dimension of b (={}) should be 1".format(b_p)

    def_string = ""
    add_attr_list = []
    for i in range(A_m):
        single_def_string = ""
        temp_string = []
        
        for j in range(A_n):
            t_st = "({})*getattr(model, x_l)[i, {}]".format(A[i,j],j,i)
            temp_string.append(t_st)
        add_attr_string = "setattr(self.model, '{}{}'+str(l),pyo.Constraint(range(m), rule={}{}))".format(name, i, name, i)
        add_attr_list.append(add_attr_string)
        return_string = "(" + " + ".join(temp_string) + "  - ({}) <= 0)".format(b[i,0])
        single_def_string = "def {}{}(model, i):\n\treturn ".format(name, i) + return_string 
        def_string = def_string + single_def_string + "\n\n"
    
    attr_string = "\n".join(add_attr_list)
    
    def_string += attr_string

    return def_string


def generate_outside_constraints(name, A,B):
    def_string = ""
    add_attr_list = []
    list_alleq = []
    for a,b in zip(A,B):
        
        list_eq = []
        iterate_1 = 0

        for a_, b_ in zip(a,b):
            str_eq = ""
            for iterate_2, val in enumerate(a_):

                if val!= 0:
                    str_eq += "(" + str(val) + " * " + "getattr(model, x_l)[i" + ", " + str(iterate_2) + "]) + " 

            
            iterate_1 += 1
            str_eq_whole = ""
            if str_eq!= "":
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

        if iterate != list_alleq.shape[0]-1:
            str_eq_all += str_eq + ",\n\t"
        else:
            str_eq_all += str_eq + "]"
    
    single_def_string = "def "+ name + "0(model, i):\n\treturn " + str_eq_all + "\n"
    single_def_string += "setattr(self.model, '"+ name +"0' +str(l), pyg.Disjunction(range(m), rule={}{}))".format(name, str(0))

    return single_def_string

@dataclass
class constraints_class:
    constraint_type : str
    A : Any
    B : Any
        

def generate_output_constraints(constraint):
    outside_count = 0
    inside_count = 0
    out_string = ""
    in_string = ""
    for i in constraint:
        
        # print(inside_count, outside_count)
        if i.constraint_type == "outside":
            temp_string = generate_outside_constraints(i.constraint_type+str(outside_count), i.A,i.B)
            # print(temp_string)
            # print("*************************************")
            out_string = out_string + temp_string + "\n\n"
            outside_count = outside_count + 1
        elif i.constraint_type == "inside":
            temp_string = generate_inside_constraints(i.constraint_type+str(outside_count), i.A,i.B)
            # print(temp_string)
            # print("*************************************")
            in_string = in_string + temp_string + "\n\n"
            inside_count = inside_count + 1

    fin = out_string + "\n\n" + in_string
    return fin


