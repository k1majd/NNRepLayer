
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

def generate_constraints(A,b):

    # assert len(b.shape) == 2, 'Shape of b must be N x 1 dimension'
    # A_m, A_n = A.shape
    # b_n, b_p = b.shape
    # assert A_m == b_n, "First dimension of A (= {}) should be equal to be first dimesnion of b(= {}).".format(A_m, b_n)
    # assert b_p == 1, "Second dimension of b (={}) should be 1".format(b_p)

    # def_string = ""
    # add_attr_list = []
    # for i in range(A_m):
    #     single_def_string = ""
    #     temp_string = []
        
    #     for j in range(A_n):
    #         t_st = "{}*getattr(model, x_l)[i, {}]".format(A[i,j],j)
    #         # print(t_st)
    #         temp_string.append(t_st)
    #     add_attr_string = "setattr(self.model, 'keep_inside_constraint{}'+str(l),pyo.Constraint(range(m), rule=constraint_inside{}))".format(i,i)
    #     add_attr_list.append(add_attr_string)
        
    #     return_string = "(" + " + ".join(temp_string) + "  - {} <= 0)".format(b[i,0])
        
    #     single_def_string = "def constraint_inside{}(model, i): return ".format(i) + return_string 
    #     def_string = def_string + single_def_string + "\n\n"
    
    # attr_string = "\n".join(add_attr_list)
    
    # def_string += attr_string
    # print(def_string)

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
            t_st = "A[{},{}]*getattr(model, x_l)[i, {}]".format(i,j,j,i)
            temp_string.append(t_st)
        add_attr_string = "setattr(self.model, 'keep_inside_constraint{}'+str(l),pyo.Constraint(range(m), rule=constraint_inside{}))".format(i,i)
        add_attr_list.append(add_attr_string)
        return_string = "(" + " + ".join(temp_string) + "  - b[{}] <= 0)".format(i)
        single_def_string = "def constraint_inside{}(model, i):\n\treturn ".format(i) + return_string 
        def_string = def_string + single_def_string + "\n\n"
    
    attr_string = "\n".join(add_attr_list)
    
    def_string += attr_string

    return def_string