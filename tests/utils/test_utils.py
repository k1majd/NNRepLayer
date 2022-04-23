import pickle
from ctypes import util
import unittest
from nnreplayer.utils import utils
import filecmp
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import os
from tensorflow import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
# from nnreplayer.utils.utils import generate_outside_constraints

class TestUtils(unittest.TestCase):
    def test_generate_outside_constraints(self):
        
        A_1 = np.array([[1,0],[-1,0],[1,0],[1,0]])
        b_1 = np.array([[2],[-3],[3],[3]])

        A_2 = np.array([[0, 0], [0, 0],[-1,0],[-1,0]])
        b_2 = np.array([[0],[0],[-2],[-2]])

        A_3 = np.array([[0, 0], [0, 0],[0,-1],[0,1]])
        b_3 = np.array([[0],[0],[-3],[2]])

        A_out = [A_1, A_2, A_3]
        B_out = [b_1,b_2,b_3]
        
        
        fin_string = utils.generate_outside_constraints("outside", A_out,B_out)
        
        with open('./tests/utils/gold_resources/outside_constraint_text.txt', 'w') as f:
            f.write(fin_string)
        
        res = filecmp.cmp("./tests/utils/gold_resources/outside_constraint_gold.txt", './tests/utils/gold_resources/outside_constraint_text.txt')
        assert res == True
        if os.path.exists("./tests/utils/gold_resources/outside_constraint_gold.txt"):
            os.remove("./tests/utils/gold_resources/outside_constraint_text.txt")

    def test1_generate_outside_constraints(self):
        
        A_1 = np.array([[1,0],[-1,0],[1,0],[1,0]])
        b_1 = np.array([[2],[-3],[3],[3]])

        A_2 = np.array([[0, 0], [0, 0],[-1,0],[-1,0]])
        b_2 = np.array([[0],[0],[-2],[-2]])

        A_3 = np.array([[0, 0], [0, 0],[0,-1],[0,1]])
        b_3 = np.array([[0],[0],[-3],[2]])

        A_out = [A_1, A_2]
        B_out = [b_1,b_2,b_3]
        self.assertRaises(ValueError, lambda: utils.generate_outside_constraints("outside", A_out, B_out))

        B_out = [b_1,b_2,b_3]
        self.assertRaises(ValueError, lambda: utils.generate_outside_constraints("outside", [], B_out))

        A_out = [A_1, A_2]
        self.assertRaises(ValueError, lambda: utils.generate_outside_constraints("outside", A_out, []))

        self.assertRaises(ValueError, lambda: utils.generate_outside_constraints("outside", [], []))
        
        self.assertRaises(ValueError, lambda: utils.generate_outside_constraints("outside", [], []))
        

    def test_generate_inside_constraints(self):
        
        poly3 = Polygon([(.45, .1), (.45, .25), (.55, .25), (.55, .1)])

        # get the coordinates of the exterior points of the polytope
        ex_points = np.array(poly3.exterior.coords)
        
        # get A and b matrices: A*x <= b
        hull = ConvexHull(ex_points)
        eqs = np.array(hull.equations)
        A = eqs[0:eqs.shape[0],0:eqs.shape[1]-1]  
        b = -eqs[0:eqs.shape[0],-1]
        b = np.array([b]).T

        fin_string = utils.generate_inside_constraints("inside", A, b)
        with open('./tests/utils/gold_resources/inside_constraint_text.txt', 'w') as f:
            f.write(fin_string)
        
        res = filecmp.cmp("./tests/utils/gold_resources/inside_constraint_gold.txt", './tests/utils/gold_resources/inside_constraint_text.txt')
        assert res == True
        if os.path.exists("./tests/utils/gold_resources/inside_constraint_text.txt"):
            os.remove("./tests/utils/gold_resources/inside_constraint_text.txt")

    def test1_generate_inside_constraints(self):
        
        poly3 = Polygon([(.45, .1), (.45, .25), (.55, .25), (.55, .1)])

        # get the coordinates of the exterior points of the polytope
        ex_points = np.array(poly3.exterior.coords)
        
        # get A and b matrices: A*x <= b
        hull = ConvexHull(ex_points)
        eqs = np.array(hull.equations)
        A = eqs[0:eqs.shape[0],0:eqs.shape[1]-1]  
        b = -eqs[0:eqs.shape[0],-1]
        

        self.assertRaises(ValueError, lambda: utils.generate_inside_constraints("outside", A, b))
        b = np.array([b]).T
        b_ = np.vstack((b,b))
        self.assertRaises(ValueError, lambda: utils.generate_inside_constraints("outside", A, b_))
        A_ = np.vstack((A,A))
        self.assertRaises(ValueError, lambda: utils.generate_inside_constraints("outside", A_, b))
        # A_ = np.vstack((A,A))
        self.assertRaises(ValueError, lambda: utils.generate_inside_constraints("outside", A_, None))
        self.assertRaises(ValueError, lambda: utils.generate_inside_constraints("outside", None, None))
        b1_ = np.hstack((b,b))
        self.assertRaises(ValueError, lambda: utils.generate_inside_constraints("outside", A, b1_))

    def test_generate_output_constraints(self):
        
        poly3 = Polygon([(.45, .1), (.45, .25), (.55, .25), (.55, .1)])

        # get the coordinates of the exterior points of the polytope
        ex_points = np.array(poly3.exterior.coords)
        
        # get A and b matrices: A*x <= b
        hull = ConvexHull(ex_points)
        eqs = np.array(hull.equations)
        A1_in = eqs[0:eqs.shape[0],0:eqs.shape[1]-1]  
        b1_in = -eqs[0:eqs.shape[0],-1]
        b1_in = np.array([b1_in]).T

        poly3 = Polygon([(.434, .121), (.2145, .2435), (.3455, .2115), (.2455, .113)])

        # get the coordinates of the exterior points of the polytope
        ex_points = np.array(poly3.exterior.coords)
        
        # get A and b matrices: A*x <= b
        hull = ConvexHull(ex_points)
        eqs = np.array(hull.equations)
        A2_in = eqs[0:eqs.shape[0],0:eqs.shape[1]-1]  
        b2_in = -eqs[0:eqs.shape[0],-1]
        b2_in = np.array([b2_in]).T
        
        A_1 = np.array([[1,0],[-1,0],[1,0],[1,0]])
        b_1 = np.array([[2],[-3],[3],[3]])

        A_2 = np.array([[0, 0], [0, 0],[-1,0],[-1,0]])
        b_2 = np.array([[0],[0],[-2],[-2]])

        A_3 = np.array([[0, 0], [0, 0],[0,-1],[0,1]])
        b_3 = np.array([[0],[0],[-3],[2]])

        A1_out = [A_1, A_2, A_3]
        B1_out = [b_1,b_2,b_3]

        A_1 = np.array([[13,0],[-1,0],[1,0],[1,0]])
        b_1 = np.array([[21],[-3],[3],[3]])

        A_2 = np.array([[0.7, 0], [0.3, 0],[-1,0],[-1,0]])
        b_2 = np.array([[0.3],[0],[-2],[-2]])

        A_3 = np.array([[0.9, 0], [0, 0],[0,-1],[0,1]])
        b_3 = np.array([[0.5],[0],[-3],[2]])

        A2_out = [A_1, A_2, A_3]
        B2_out = [b_1,b_2,b_3]

        inside_1 = utils.ConstraintsClass("inside", A1_in, b1_in)
        outside_1 = utils.ConstraintsClass("outside", A1_out, B1_out)
        inside_2 = utils.ConstraintsClass("inside", A2_in, b2_in)
        outside_2 = utils.ConstraintsClass("outside", A2_out, B2_out)
        constraint = [inside_1, outside_1, inside_2, outside_2]

        fin_string = utils.generate_output_constraints(constraint)

        with open('./tests/utils/gold_resources/all_constraint_text.txt', 'w') as f:
            f.write(fin_string)
        
        res = filecmp.cmp("./tests/utils/gold_resources/all_constraint_gold.txt", './tests/utils/gold_resources/all_constraint_text.txt')
        assert res == True
        if os.path.exists("./tests/utils/gold_resources/all_constraint_text.txt"):
            os.remove("./tests/utils/gold_resources/all_constraint_text.txt")

    
    def test_tf2_get_architecture(self):
        model_orig = keras.models.load_model("./tests/utils/gold_resources/affine_transform_original_model/")
        model_orig.compile()
        # print(utils.tf2_get_architecture(model_orig))
        assert utils.tf2_get_architecture(model_orig) == [3, 20, 10, 3]
    
    def test1_tf2_get_architecture(self):
        model_orig = None
        # print(utils.tf2_get_architecture(model_orig))
        self.assertRaises(TypeError, utils.tf2_get_architecture, model_orig)
    
    def test_tf2_get_weights(self):
        model_orig = keras.models.load_model("./tests/utils/gold_resources/affine_transform_original_model")
        model_orig.compile()
        # print(utils.tf2_get_architecture(model_orig))
        arch = utils.tf2_get_architecture(model_orig)
        parameters_layers = len(arch) - 1
        parameters = utils.tf2_get_weights(model_orig)
        with open("./tests/utils/gold_resources/affine_transform_weights_gold.pkl", "rb") as f:
            gold_params = pickle.load(f)
            # pickle.dump(parameters, f)

        for x in range(0,2*parameters_layers, 2):
            np.testing.assert_array_equal(parameters[x], gold_params[x])
            np.testing.assert_array_equal(parameters[x+1], gold_params[x+1])
    
    def test_pt_get_architecture(self):
        model_path = "tests/utils/gold_resources/gold_pytorch_model.pt"
        model_orig = torch.torch.jit.load(model_path)
        model_orig.eval()
        
        # print(utils.tf2_get_architecture(model_orig))
        assert utils.pt_get_architecture(model_orig) == [3, 20, 10, 3]
    
    def test1_pt_get_architecture(self):
        model_orig = None
        # print(utils.tf2_get_architecture(model_orig))
        self.assertRaises(TypeError, utils.pt_get_architecture, model_orig)
    
    def test_pt_get_weights(self):
        model_path = "tests/utils/gold_resources/gold_pytorch_model.pt"
        model_orig = torch.torch.jit.load(model_path)
        model_orig.eval()
        
        arch = utils.pt_get_architecture(model_orig)
        parameters_layers = len(arch) - 1
        parameters = utils.pt_get_weights(model_orig)
        with open("./tests/utils/gold_resources/pytorch_affine_transform_weights_gold.pkl", "rb") as f:
            gold_params = pickle.load(f)
            # pickle.dump(parameters, f)
        # print(hekdsnvcsdn)
        for x in range(0,2*parameters_layers, 2):
            np.testing.assert_array_equal(parameters[x], gold_params[x])
            np.testing.assert_array_equal(parameters[x+1], gold_params[x+1])

    def test_give_mse_error(self):
        rng = np.random.default_rng(12345)

        d1 = rng.random((10,20))
        d2 = rng.random((10,20))

        x = utils.give_mse_error(d1, d2)
        
        with open("./tests/utils/gold_resources/goldErrorOutput_seed12345.pkl", "rb") as f:
            gold_output = pickle.load(f)

        np.testing.assert_array_equal(x, gold_output)
        x1 = utils.give_mse_error(d1, d1)
        np.testing.assert_array_equal(np.zeros((10,20)), x1)
    
    def test1_give_mse_error(self):
        rng = np.random.default_rng(12345)

        d1 = rng.random((10,20))
        d2 = rng.random((10,2))

        self.assertRaises(ValueError, utils.give_mse_error, d1,d2)
        d1 = rng.random((1,2))
        d2 = rng.random((10,2))
        self.assertRaises(ValueError, utils.give_mse_error, d1,d2)
        self.assertRaises(TypeError, utils.give_mse_error, d1, None)
        self.assertRaises(TypeError, utils.give_mse_error, None, d1)


if __name__ == '__main__':
    unittest.main()