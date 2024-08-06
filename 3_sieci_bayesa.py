#!/usr/bin/env python

"""code template"""
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def zad_3():
    # Create the model with edges specified as tuples (parent, child)
    dentist_model = BayesianModel([('Cavity', 'Toothache'),
                                   ('Cavity', 'Catch')])
    # Create tabular CPDs, values has to be 2-D array
    cpd_cav = TabularCPD('Cavity', 2, [[0.2], [0.8]], state_names={'Cavity': ['True', 'False']})
    cpd_too = TabularCPD('Toothache', 2, [[0.6, 0.1],
                                          [0.4, 0.9]],
                         evidence=['Cavity'], evidence_card=[2], state_names={'Toothache': ['True', 'False'], 'Cavity': ['True', 'False']})
    cpd_cat = TabularCPD('Catch', 2, [[0.9, 0.2],
                                      [0.1, 0.8]],
                         evidence=['Cavity'], evidence_card=[2], state_names={'Catch': ['True', 'False'], 'Cavity': ['True', 'False']})
    # Add CPDs to model
    dentist_model.add_cpds(cpd_cav, cpd_too, cpd_cat)

    print('Check model :', dentist_model.check_model())

    print('Independencies:\n', dentist_model.get_independencies())

    # Initialize inference algorithm
    dentist_infer = VariableElimination(dentist_model)

    # Some exemplary queries
    q = dentist_infer.query(['Toothache'])
    print('P(Toothache) =\n', q)

    q = dentist_infer.query(['Cavity'])
    print('P(Cavity) =\n', q)

    q = dentist_infer.query(['Toothache'], evidence={'Cavity': 0})
    print('P(Toothache | cavity) =\n', q)

    q = dentist_infer.query(['Toothache'], evidence={'Cavity': 1})
    print('P(Toothache | ~cavity) =\n', q)

    q = dentist_infer.query(['Cavity'], evidence={'Toothache': 'True', 'Catch': 'False'})
    print('P(Cavity | toothache, ~cavity) =\n', q)

    q_1 = dentist_infer.query(['Cavity'], evidence={'Toothache': 'True', 'Catch': 'False'})
    print('P(Cavity | toothache, ~cavity) =\n', q_1)
    q_2 = dentist_infer.query(['Cavity'], evidence={'Toothache': 'True', 'Catch': 'True'})
    print('P(Cavity | toothache, cavity) =\n', q_2)
    q_3 = dentist_infer.query(['Cavity'], evidence={'Toothache': 'False', 'Catch': 'False'})
    print('P(Cavity | ~toothache, ~cavity) =\n', q_3)
    q_4 = dentist_infer.query(['Cavity'], evidence={'Toothache': 'False', 'Catch': 'True'})
    print('P(Cavity | ~toothache, cavity) =\n', q_4)

    last = np.array([[[q_1.values[0], q_2.values[0]], [q_3.values[0], q_4.values[0]]],
                     [[q_1.values[1], q_2.values[1]], [q_3.values[1], q_4.values[1]]]])
    print(last)

def zad_4():
    # Create the model with edges specified as tuples (parent, child)
    auto = BayesianModel([('Battery', 'Radio'),
                          ('Battery', 'Ignition'),
                          ('Ignition', 'Starts'),
                          ('Gas', 'Starts'),
                          ('Starts', 'Moves'),
                          ('NotWe', 'Starts'),
                          ('Battery', 'StMot'),
                          ('StMot', 'Starts')])
    # Create tabular CPDs, values has to be 2-D array
    cpd_bat = TabularCPD('Battery', 2, [[0.7], [0.3]], state_names={'Battery': ['True', 'False']})

    cpd_gas = TabularCPD('Gas', 2, [[0.5], [0.5]], state_names={'Gas': ['True', 'False']})
    cpd_notwe = TabularCPD('NotWe', 2, [[0.9], [0.1]], state_names={'NotWe': ['True', 'False']})

    cpd_rad = TabularCPD('Radio', 2, [[0.9, 0.0],
                                      [0.1, 1.0]],
                         evidence=['Battery'], evidence_card=[2],
                         state_names={'Radio': ['True', 'False'], 'Battery': ['True', 'False']})

    cpd_ign = TabularCPD('Ignition', 2, [[0.97, 0.0],
                                         [0.03, 1.0]],
                         evidence=['Battery'], evidence_card=[2],
                         state_names={'Ignition': ['True', 'False'], 'Battery': ['True', 'False']})

    cpd_mov = TabularCPD('Moves', 2, [[0.8, 0.0],
                                      [0.2, 1.0]],
                         evidence=['Starts'], evidence_card=[2],
                         state_names={'Moves': ['True', 'False'], 'Starts': ['True', 'False']})

    cpd_stmot = TabularCPD('StMot', 2, [[0.95, 0.0],
                                      [0.05, 1.0]],
                         evidence=['Battery'], evidence_card=[2],
                         state_names={'StMot': ['True', 'False'], 'Battery': ['True', 'False']})

    cpd_star = TabularCPD('Starts', 2, [[0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.15, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                         evidence=['Ignition', 'Gas', 'NotWe', 'StMot'], evidence_card=[2, 2, 2, 2],
                         state_names={'Starts': ['True', 'False'], 'Ignition': ['True', 'False'], 'Gas' : ['True', 'False'], 'NotWe' : ['True', 'False'], 'StMot' : ['True', 'False']})
    # Add CPDs to model
    auto.add_cpds(cpd_bat, cpd_gas, cpd_notwe, cpd_rad, cpd_ign, cpd_mov,cpd_stmot, cpd_star)

    print('Check model :', auto.check_model())

    print('Independencies:\n', auto.get_independencies())

    # Initialize inference algorithm
    dentist_infer = VariableElimination(auto)

    q = dentist_infer.query(['Radio'], evidence={'Starts': 'False'})
    print('P(Radio | ~starts) =\n', q)



def zad_5():
    # Create the model with edges specified as tuples (parent, child)
    auto = BayesianModel([('Rain_t-1', 'Rain_t'),
                          ('Rain_t', 'Rain_t+1'),
                          ('Rain_t-1', 'Umbrella_t-1'),
                          ('Rain_t', 'Umbrella_t'),
                          ('Rain_t+1', 'Umbrella_t+1')])

    rain_0 = np.array([[0.6], [0.4]])

    rain_t = np.array([[0.7], [0.3],
                      [0.3], [0.7]])

    umbrella = np.array([[0.9], [0.2],
                      [0.1], [0.8]])


    # Create tabular CPDs, values has to be 2-D array
    cpd_Rain_0 = TabularCPD('Rain_t-1', 2, [[0.6], [0.4]], state_names={'Rain_t-1': ['True', 'False']})

    cpd_Rain_1 = TabularCPD('Rain_t', 2, [[0.7, 0.3],
                                      [0.3, 0,7]],
                         evidence=['Rain_t-1'], evidence_card=[2],
                         state_names={'Rain_t': ['True', 'False'], 'Rain_t-1': ['True', 'False']})

    cpd_Rain_2 = TabularCPD('Rain_t+1', 2, [[0.7, 0.3],
                                         [0.3, 0.7]],
                         evidence=['Rain_t'], evidence_card=[2],
                         state_names={'Rain_t+1': ['True', 'False'], 'Rain_t': ['True', 'False']})

    cpd_Umbrel_0 = TabularCPD('Rain_t-1', 2, [[0.9, 0.2],
                                      [0.1, 0.8]],
                         evidence=['Umbrella_t-1'], evidence_card=[2],
                         state_names={'Rain_t-1': ['True', 'False'], 'Umbrella_t-1': ['True', 'False']})


    cpd_Umbrel_1 = TabularCPD('Rain_t', 2, [[0.9, 0.2],
                                      [0.1, 0.8]],
                         evidence=['Umbrella_t'], evidence_card=[2],
                         state_names={'Rain_t': ['True', 'False'], 'Umbrella_t': ['True', 'False']})


    cpd_Umbrel_2 = TabularCPD('Rain_t+1', 2, [[0.9, 0.2],
                                      [0.1, 0.8]],
                         evidence=['Umbrella_t+1'], evidence_card=[2],
                         state_names={'Rain_t+1': ['True', 'False'], 'Umbrella_t+1': ['True', 'False']})

    # Add CPDs to model
    auto.add_cpds(cpd_Rain_0, cpd_Rain_1, cpd_Rain_2, cpd_Umbrel_0, cpd_Umbrel_1, cpd_Umbrel_2)

    print('Check model :', auto.check_model())

    print('Independencies:\n', auto.get_independencies())

    # Initialize inference algorithm
    dentist_infer = VariableElimination(auto)

    q = dentist_infer.query(['Umbrella_t-1'], evidence={'Rain_t-1': 'True', 'Gas': 'True'})
    print('P(Starts | radio, gas) =\n', q)



if __name__ == '__main__':
    #zad_3()
    #zad_4()
    zad_5()