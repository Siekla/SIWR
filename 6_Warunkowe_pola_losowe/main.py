"""code template"""

import cv2 as cv
import numpy as np
import itertools
from pgmpy.models import MarkovModel, MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Mplp


def create_factor(var_names, var_vals, params, feats, obs):
    """
    Creates factors for given variables using exponential family and provided features.
    :param var_names: list of variable names, e.g. ['A', 'B']
    :param var_vals: list of lists of variable values, e.g. [[1, 2, 3], [3, 4]]
    :param params: list of theta parameters, one for each feature, e.g. [0.4, 5]
    :param feats: list of features (functions that accept variables and observations as arguments),
                    e.g. [feat_fun1, feat_fun2], were feat_fun1 can be defined as 'def feat_fun1(A, B, obs)'
    :param obs: observations that will be passed as the last positional argument to the features
    :return: DiscreteFactor with values computed using provided features
    """
    # shape of the values array
    f_vals_shape = [len(vals) for vals in var_vals]
    # list of values, will be reshaped later
    f_vals = []
    # for all combinations of variables values
    for vals in itertools.product(*var_vals):
        # value for current combination
        cur_f_val = 0
        # for each feature
        for fi, cur_feat in enumerate(feats):
            # value of feature multipled by parameter value
            cur_f_val += params[fi] * cur_feat(*vals, obs)
        f_vals.append(np.exp(cur_f_val))
    # reshape values array
    f_vals = np.array(f_vals)
    f_vals = f_vals.reshape(f_vals_shape)

    return DiscreteFactor(var_names, f_vals_shape, f_vals)

def feat1(x1, x2, obs):
    if x1 == x2:
        return 1
    else:
        return 0
def feat2(x, obs):
    if x == -1:
        return np.log(1-obs)
    elif x == 1:
        return np.log(obs)



def main():
    # create graph and compute MAP assignment
    # TODO PUT YOUR CODE HERE
    # TODO Due to a bug in the MPLP implementation,
    # TODO add unary factors before pairwise ones!

    image =cv.imread("smile_noise_small.png")

    G = MarkovNetwork([('x1', 'x2'), ('x2', 'x3'), ('x3', 'x4'), ('x4', 'x1')])
    u = 1
    z = 0.7
    smile_x, smile_y, smile_c = image.shape
    print(smile_x)
    print(smile_y)
    for w in range(smile_x-1):
        for h in range(smile_y-1):


            current_pixel = "fu_" + str(w) + str(h)
            down_pixel = "fd_" + str(w) + str(h+1)
            right_pixel = "fr_" + str(w+1) + str(h)

            fpixel_current = create_factor([current_pixel], [[-1, 1]], [u], [feat2], z)
            fpixel_down = create_factor([down_pixel], [[-1, 1]], [u], [feat2], z)
            fpixel_right = create_factor([right_pixel], [[-1, 1]], [u], [feat2], z)

            fpixel_CD = create_factor([fpixel_current, fpixel_down], [[-1, 1], [-1, 1]], [1], [feat1], None)
            fpixel_CR = create_factor([fpixel_current, fpixel_right], [[-1, 1], [-1, 1]], [1], [feat1], None)

            G.add_edges_from([(fpixel_current, fpixel_down), (fpixel_current, fpixel_right)])
            G.add_factors(fpixel_current, fpixel_down, fpixel_right, fpixel_CD, fpixel_CR)
            #print(current_pixel)
            #print(image[h][w])


    # G = MarkovModel([('x1', 'x2'), ('x2', 'x3'), ('x3', 'x4'), ('x4', 'x1')])
    #G = MarkovNetwork([('x1', 'x2'), ('x2', 'x3'), ('x3', 'x4'), ('x4', 'x1')])

    # fu1 = create_factor(['x1'], [[-1, 1]], [0.1], [feat2], 0.7)
    # fu2 = create_factor(['x2'], [[-1, 1]], [0.1], [feat2], 0.3)
    # fu3 = create_factor(['x3'], [[-1, 1]], [0.1], [feat2], 0.6)
    # fu4 = create_factor(['x4'], [[-1, 1]], [0.1], [feat2], 0.4)
    #
    # fp1 = create_factor(['x1', 'x2'], [[-1, 1], [-1, 1]], [1], [feat1], None)
    # fp2 = create_factor(['x2', 'x3'], [[-1, 1], [-1, 1]], [1], [feat1], None)
    # fp3 = create_factor(['x3', 'x4'], [[-1, 1], [-1, 1]], [1], [feat1], None)
    # fp4 = create_factor(['x4', 'x1'], [[-1, 1], [-1, 1]], [1], [feat1], None)
    #
    # G.add_nodes_from(['x1', 'x2', 'x3', 'x4'])
    # G.add_edges_from([('x1', 'x2'), ('x2', 'x3'), ('x3', 'x4'), ('x4', 'x1'), ])
    # G.add_factors(fu1, fu2, fu3, fu4, fp1, fp2, fp3, fp4)


    # G.get_factors()
    print(G)
    G.check_model()

    mplp = Mplp(G)

    result = mplp.map_query()

    print(result)


    # ------------------

    # read results of the inference
    # TODO PUT YOUR CODE HERE
    # TODO Due to a bug in the MPLP implementation,
    # TODO add unary factors before pairwise ones!

    # ------------------


if __name__ == '__main__':
    main()