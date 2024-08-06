# !/usr/bin/env python

"""code template"""

import numpy as np

def main():
    P_cav_too_cat = np.array([[[0.108, 0.012],  # too = yes, cav = yes
                               [0.072, 0.008]],  # too = no, cav = yes
                              [[0.016, 0.064],  # too = yes, cav = no
                               [0.144, 0.576]]])  # too = no, cav = no
    print(P_cav_too_cat)

    P_too = P_cav_too_cat.sum(axis=(0, 2))  # P_Toothache
    P_cav = P_cav_too_cat.sum(axis=(1, 2))  # P_Cavity

    print(P_too)
    print(P_cav)

    P_cav_too = P_cav_too_cat.sum(axis=2)
    P_too_giv_cav = P_cav_too.transpose() / P_cav[np.newaxis, :]

    print(P_too_giv_cav)

    # P_cav_giv_too_cat =(
    # [[[0.87096774, 0.15789474],
    #  [0.33333333, 0.01369863]],
    # [[0.12903226, 0.84210526],
    # [0.66666667, 0.98630137]]])

    # zad8
    P_too_cat_giv_cav = (P_cav_too_cat / P_cav[:, np.newaxis, np.newaxis]).transpose((1, 2, 0))
    P_cav_giv_too_cat_nn = (P_too_cat_giv_cav * P_cav[np.newaxis, np.newaxis, :]).transpose((2, 0, 1))
    P_nn = P_cav_giv_too_cat_nn.sum(0)
    P_cav_g_too_cat = P_cav_giv_too_cat_nn / P_nn
    print(P_cav_g_too_cat)


if __name__ == '__main__':
    main()
