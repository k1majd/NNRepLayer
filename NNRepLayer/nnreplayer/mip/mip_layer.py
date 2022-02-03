import pyomo.environ as pyo
import pyomo.gdp as pyg
from pyomo.gdp import *
from ..utils.utils import generate_constraints

class MIPLayer:
    def __init__(self, model, layer_to_repair, uin, uout, weights, bias, param_bounds=(-1, 1)):

        model.nlayers = getattr(model, 'nlayers', 0)
        # print("Model layers = {}".format(model.nlayers))
        self.layer_num = model.nlayers
        self.uin, self.uout = uin, uout

        if model.nlayers == layer_to_repair:
            w_l, b_l = 'w' + str(model.nlayers), 'b' + str(model.nlayers)
        
            setattr(model, w_l, pyo.Var(range(uin), range(uout), domain=pyo.Reals, bounds=param_bounds))
            setattr(model, b_l, pyo.Var(range(uout), domain=pyo.Reals, bounds=param_bounds))
        
            self.w = getattr(model, w_l)
            self.b = getattr(model, b_l)
            self.w_orig = weights
            self.b_orig = bias
        else:
            self.w = weights
            self.b = bias
            
        model.nlayers += 1

        self.model = model
        self.layer_to_repair = layer_to_repair
        
        
    def __call__(self, x, shape, A,b, relu=False,weightSlack=10, output_bounds=(-1e1, 1e1)):
        
        self.lout = getattr(self, 'layer_num', 0)+1
        if relu:
            return self._relu_constraints(x, shape, self.lout, weightSlack, output_bounds)
        return self._constraints(x, shape, self.lout, A,b, weightSlack, output_bounds)
    
    def _relu_constraints(self, x, shape, l, weightSlack = 10, output_bounds=(-1e1, 1e1)):
        m, n = shape
        assert n == self.uin
        
        x_l, s_l, theta_l = 'x'+str(l), 's'+str(l), 'theta'+str(l)
        w_l = 'w'+str(l-1)
        b_l = 'b'+str(l-1)

        setattr(self.model, x_l, pyo.Var(range(m), range(self.uout), domain=pyo.NonNegativeReals, bounds=output_bounds))
        setattr(self.model, s_l, pyo.Var(range(m), range(self.uout), domain=pyo.NonNegativeReals, bounds=output_bounds))
        setattr(self.model, theta_l, pyo.Var(range(m), range(self.uout), domain=pyo.Binary))
        
        def constraints(model, i, j):
            product = self.b[j]
            for k in range(self.uin):
                product += x[i, k] * self.w[k, j]
            return product == getattr(model, x_l)[i, j] - getattr(model, s_l)[i, j]
        
        setattr(self.model, 'eq_constraint'+str(l),
                pyo.Constraint(range(m), range(self.uout), rule=constraints))
        
        if l==self.layer_to_repair+1:
            print("Activating mid layer")
            
            dw_l = 'dw'
            setattr(self.model, dw_l, pyo.Var(within=pyo.NonNegativeReals, bounds=(0, weightSlack)))
            
            def constraint_bound_w0(model, i, j):
                return(getattr(model, w_l)[i, j]-self.w_orig[i,j] <= getattr(model, dw_l))
        
            def constraint_bound_w1(model, i, j): 
                return(getattr(model, w_l)[i, j]-self.w_orig[i,j] >= - getattr(model, dw_l))
        
            def constraint_bound_b0(model, j):
                return(getattr(model, b_l)[j]-self.b_orig[j] <= getattr(model, dw_l))
        
            def constraint_bound_b1(model, j): 
                return(getattr(model, b_l)[j]-self.b_orig[j] >= - getattr(model, dw_l))
            
            setattr(self.model, 'w_bounded_constraint0'+str(l),
                pyo.Constraint(range(self.uin), range(self.uout), rule=constraint_bound_w0))
            setattr(self.model, 'w_bounded_constraint1'+str(l),
                pyo.Constraint(range(self.uin), range(self.uout), rule=constraint_bound_w1))
            setattr(self.model, 'b_bounded_constraint0'+str(l),
                pyo.Constraint(range(self.uout), rule=constraint_bound_b0))
            setattr(self.model, 'b_bounded_constraint1'+str(l),
                pyo.Constraint(range(self.uout), rule=constraint_bound_b1))
        
        def disjuncts(model, i, j):
            return [(getattr(model, theta_l)[i, j] == 0, getattr(model, x_l)[i, j] <= 0),
                    (getattr(model, theta_l)[i, j] == 1, getattr(model, s_l)[i, j] <= 0)]
        
        setattr(self.model, 'disjunction'+str(l), pyg.Disjunction(range(m), range(self.uout), rule=disjuncts))
        return  getattr(self.model, x_l)
        
    def _constraints(self, x, shape, l, A,b, weightSlack = 10, output_bounds=(-1e1, 1e1)):
        m, n = shape
        assert n == self.uin
        if l==self.layer_to_repair+1:
            w_l = 'w'+str(l-1)
            b_l = 'b'+str(l-1)

        x_l, ind_l = 'x'+str(l), 'ind'+str(l)
        x_l = 'x'+str(l)

        setattr(self.model, ind_l, pyo.Var(range(m), range(1), domain=pyo.Binary))

        #x_l = 'x'+str(l)
        setattr(self.model, x_l, pyo.Var(range(m), range(self.uout), domain=pyo.Reals, bounds=output_bounds))
        
        def constraints(model, i, j):
            product = self.b[j]
            for k in range(self.uin):
                product += x[i, k] * self.w[k, j]
            return product == getattr(model, x_l)[i, j]
        
        setattr(self.model, 'eq_constraint'+str(l),
                pyo.Constraint(range(m), range(self.uout), rule=constraints))
        
        constraint_addition_string = generate_constraints(A,b)
        exec(constraint_addition_string, locals(), globals())
        # def constraint_inside0(model, i):  
        #     return [(getattr(model, ind_l)[i, 0] == 0, getattr(model, x_l)[i, 0] + 0.0001 <= getattr(model, x_l)[i, 1], 
        #             getattr(model, x_l)[i, 0] + 0.0001 <= getattr(model, x_l)[i, 2],
        #             getattr(model, x_l)[i, 0] + 0.0001 <= getattr(model, x_l)[i, 3],
        #             getattr(model, x_l)[i, 0] + 0.0001 <= getattr(model, x_l)[i, 4]),
        #             (getattr(model, ind_l)[i, 0] == 1, getattr(model, x_l)[i, 1] + 0.0001 <= getattr(model, x_l)[i, 0], 
        #             getattr(model, x_l)[i, 1] + 0.0001 <= getattr(model, x_l)[i, 2],
        #             getattr(model, x_l)[i, 1] + 0.0001 <= getattr(model, x_l)[i, 3],
        #             getattr(model, x_l)[i, 1] + 0.0001 <= getattr(model, x_l)[i, 4])]
        
        # setattr(self.model, 'keep_inside_constraint0'+str(l),
        #                 pyg.Disjunction(range(m), rule=constraint_inside0))

        # ########## Define the output constraint here ########################
        # def constraint_inside0(model, i):  
        #     return [[getattr(model, x_l)[i, 0] - 0.45 <= 0],
        #             [0.55 - getattr(model, x_l)[i, 0] <= 0],
        #             [getattr(model, x_l)[i, 0] - 0.55 <= 0, 0.45 - getattr(model, x_l)[i, 0] <= 0, 0.25 - getattr(model, x_l)[i, 1] <= 0],
        #             [getattr(model, x_l)[i, 0] - 0.55 <= 0, 0.45 - getattr(model, x_l)[i, 0] <= 0, getattr(model, x_l)[i, 1] - 0.1 <= 0]]
        
        # setattr(self.model, 'keep_inside_constraint0'+str(l),
        #          pyg.Disjunction(range(m), rule=constraint_inside0))
        # #####################################################################

        if l==self.layer_to_repair+1:
            print("Activating Last layer")
            
            dw_l = 'dw'
            setattr(self.model, dw_l, pyo.Var(within=pyo.NonNegativeReals, bounds=(0, weightSlack)))
            
            def constraint_bound_w0(model, i, j):
                return(getattr(model, w_l)[i, j]-self.w_orig[i,j] <= getattr(model, dw_l))
        
            def constraint_bound_w1(model, i, j): 
                return(getattr(model, w_l)[i, j]-self.w_orig[i,j] >= -getattr(model, dw_l))
        
            def constraint_bound_b0(model, j):
                return(getattr(model, b_l)[j]-self.b_orig[j] <= getattr(model, dw_l))
        
            def constraint_bound_b1(model, j): 
                return(getattr(model, b_l)[j]-self.b_orig[j] >= -getattr(model, dw_l))
            
            setattr(self.model, 'w_bounded_constraint0'+str(l),
                pyo.Constraint(range(self.uin), range(self.uout), rule=constraint_bound_w0))
            setattr(self.model, 'w_bounded_constraint1'+str(l),
                pyo.Constraint(range(self.uin), range(self.uout), rule=constraint_bound_w1))
            setattr(self.model, 'b_bounded_constraint0'+str(l),
                pyo.Constraint(range(self.uout), rule=constraint_bound_b0))
            setattr(self.model, 'b_bounded_constraint1'+str(l),
                pyo.Constraint(range(self.uout), rule=constraint_bound_b1))
        
        
        
        return getattr(self.model, x_l)