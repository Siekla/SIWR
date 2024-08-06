import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from itertools import combinations
from pgmpy.inference import BeliefPropagation

previos_histr = []


class write_kord:
    def __init__(self, n_box):
        self.number_of_bboxes = n_box
        self.bboxes_array = np.zeros((n_box, 4))

    def add_box(self, index, array):
        self.bboxes_array[index] = array


def main():
    # Zdeklarowanie zmiennej globalnej odpowiadającej za przechowywanie wcześniejszego historgramu
    global previos_histr

    bb_file = open("bboxes.txt", "r")
    lines = bb_file.read()

    file_split = lines.split("\n")

    vector_name = []
    box_dict = {}

    img_list = os.listdir("D:\Magisterka_sem_1\SIwR\projekcik/frames")
    img_list.sort()

    for img in img_list:
        for box in range(len(file_split)):
            if img == file_split[box]:
                # print(file_split[box])
                index = int(file_split[box + 1])
                img_class = write_kord(index)
                vector_name.append(file_split[box])

                for idx in range(index):
                    kordy = file_split[box + 1 + idx + 1].split(" ")
                    cords_array = [float(kordy[0]), float(kordy[1]), float(kordy[2]), float(kordy[3])]
                    # print(kordy)
                    img_class.add_box(idx, cords_array)
                    box_dict[file_split[box]] = img_class


    numer_zdjecia = 0
    numer_histogramu = 0
    old_name = []
    # wyświetlanie obrazów wraz z ramkami wokół ludzi
    for name in vector_name:
        img1 = cv.imread("D:\Magisterka_sem_1\SIwR\projekcik/frames/" + name)
        print(('\nNumer zdjecia ' + str(numer_zdjecia + 1)))

        present_histr = []

        G = FactorGraph()

        for index_1 in range(box_dict[name].number_of_bboxes):

            first_point = (int(box_dict[name].bboxes_array[index_1, 0]) + int(box_dict[name].bboxes_array[index_1, 2]),
                           int(box_dict[name].bboxes_array[index_1, 1]))
            last_point = (int(box_dict[name].bboxes_array[index_1, 0]),
                          int(box_dict[name].bboxes_array[index_1, 2]) + int(box_dict[name].bboxes_array[index_1, 3]))
            # cv.rectangle(img1, first_point, last_point, (0, 0, 255), 3) #wyświetlanie głównego BB

            # tworzenie zmniejszonego BB
            point_f_x, point_f_y = first_point
            point_l_x, point_l_y = last_point
            center_l_x = point_l_x + (point_f_x - point_l_x) / 2
            center_l_y = point_f_y + (point_l_y - point_f_y) / 2
            square_x = (center_l_x - point_l_x) / 2
            square_y = (center_l_y - point_f_y) / 2
            left_x = int(center_l_x - square_x)
            right_x = int(center_l_x + square_x)
            down_y = int(center_l_y - square_y)
            up_y = int(center_l_y + square_y)

            # tworzenie histogramu z zmnijeszonego BBoxa
            img_to_hist = img1[down_y:up_y, left_x:right_x]
            # cv.imshow('zdjecie do Histograma' + str(numer_zdjecia), img_to_hist)  #Wyświetlanie BB
            img_to_hist = cv.cvtColor(img_to_hist, cv.COLOR_BGR2GRAY)
            histr = cv.calcHist([img_to_hist], [0], None, [256], [0, 256])
            # plt.plot(histr)
            # plt.show()

            present_histr.append(histr)
            numer_histogramu += 1

            print('Numer BB: ' + str(numer_histogramu), '\nWysokość', up_y - down_y, ' Szerokosc: ', right_x - left_x)
            cv.rectangle(img1, (left_x, down_y), (right_x, up_y), (255, 0, 0), 2)  # wyświetlanie ograniczonego BB

        print("present_histr", len(present_histr))
        print("previos_histr", len(previos_histr))

        cv.imshow('Image' + str(numer_zdjecia + 1), img1) #

        # print(len(present_histr))
        # print(len(previos_histr))

        list_of_compare_hist = []
        wartosc = []
        if numer_zdjecia >= 1:
            DisFact_name = 0
            # for k in range(len(previos_histr)):
            for i in range(len(present_histr)):
                for j in range(len(previos_histr)):
                    compare = cv.compareHist(present_histr[i], previos_histr[j], cv.HISTCMP_CORREL)
                    print('Compare', compare)
                    plt.subplot(1, 2, 1)
                    plt.plot(present_histr[i])
                    plt.subplot(1, 2, 2)
                    plt.plot(previos_histr[j])
                    plt.show()  # !!!!!!!!!!!!!!!!!!!
                    list_of_compare_hist.append(compare)


                    DisFact_name = str(name) + '_' + str(i)
                    wartosc.append(DisFact_name)
                    print("DS NAME: ", DisFact_name)


                    G.add_node(DisFact_name)
                    phi0 = DiscreteFactor([DisFact_name], [len(list_of_compare_hist) + 1], [[0.6] + list_of_compare_hist])
                    G.add_factors(phi0)
                    G.add_edge(DisFact_name, phi0)
                    list_of_compare_hist.clear()

            A = np.ones((len(previos_histr) + 1, len(previos_histr) + 1))
            B = np.eye(len(previos_histr) + 1)
            AB = A - B
            AB[0][0] += 1

            # wartosc = []
            # for k in range(len(previos_histr)):0
            #     name_for_bb = name + '_' + str(k)
            #     wartosc.append(name_for_bb)
            # for k in range(len(present_histr)):
            #     name_for_bb = name + '_' + str(k)
            #     wartosc.append(name_for_bb)

            print("WARTOSC", wartosc)
            print("old name", old_name)

            comb = [x for x in combinations(wartosc, 2)]
            print("comb", comb)
            # for i in range(len(comb)):
            for i in list(comb):
                # G.add_node(DisFact_name)
                print("comb1", i[0])
                print("comb2", i[1])
                # print(comb)
                phi1 = DiscreteFactor([i[0], i[1]], [len(previos_histr) + 1, len(previos_histr) + 1], AB)
                G.add_factors(phi1)
                # G.add_edges_from([(comb[i][0], phi1), (comb[i][1], phi1)])
                G.add_edge(i[0], phi1)
                G.add_edge(i[1], phi1)

            old_name = wartosc
            print("old name2", old_name)

            # Bel_Propag = BeliefPropagation(G)
            # Bel_Propag.calibrate()
            # people = None
            # value = Bel_Propag.map_query(G.get_variable_nodes(), show_progress=False)

            # print(G.get_variable_nodes())
            Bel_Propag = BeliefPropagation(G)
            Bel_Propag.calibrate()

            # que = Bel_Propag.map_query(variables=comb, show_progress=False)
            # que_dic = dict(que)
            # print(que_dic)

            result = list(Bel_Propag.map_query(show_progress=False, variables=G.get_variable_nodes()).values())
            print(result)
            wartosc.clear()

            # phi0 = DiscreteFactor([str(i), [len(previos_histr) + 1], [list_of_compare_hist])

        previos_histr.clear()
        previos_histr = present_histr.copy()
        present_histr.clear()

        numer_zdjecia += 1
        cv.waitKey(0)
        numer_histogramu = 0


if __name__ == '__main__':
    main()
