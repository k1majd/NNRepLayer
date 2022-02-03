from .dense import Dense
class MLP:
    def __init__(self, nin, uout, uhidden):
        self.num_layer = len(uhidden)+1
        prev = nin
        self.layers = []
        for u in uhidden:
            self.layers.append(Dense(prev, u))
            prev = u
        self.layers.append(Dense(prev, uout))
        
    def __call__(self, x, relu=False):
        
        x1 = self.layers[0](x, relu=True)
        for iterate in range(1, self.num_layer-1):
            variable1 = "x"+str(iterate+1)
            variable2 = "x"+str(iterate)
            vars()[variable1] = self.layers[iterate](vars()[variable2], relu=True)
            
        variable_name = "x" + str(self.num_layer-1)
        y = self.layers[self.num_layer-1](vars()[variable_name], relu = True)


        exec_string = ""
        for j in range(1, self.num_layer):
            exec_string = exec_string + "x" + str(j) + ", "
        exec_string = exec_string + "y"
        
        return list(eval(exec_string))