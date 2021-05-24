# for planner
import sys
import time
from math import isclose
import numpy as np
import copy
import itertools
from scipy.special import comb
from scipy.stats import entropy
from scipy.spatial import distance
import random
import gurobipy as grb
import pandas as pd
import matplotlib.pyplot as plt
from partial_semantics import *


##############################################################################
# Airsim initialization
grid_space = 5
x_state_num = 7
y_state_num = 7

target = 0 # At the beginning reaching the target is FALSE
agent_x = 0
agent_y = 0
##############################################################################

# define simulation parameters
n_iter = 10
infqual_hist_all = []
risk_hist_all = []
timestep_all = []
plan_count_all = []
task_flag_all = []

for iter in range(n_iter):

    # create problem setting
    model = MDP('gridworld')
    model.semantic_representation() # changed for scenario 2
    perc_flag = True
    bayes_flag = True
    replan_flag = True
    div_test_flag = False
    act_info_flag = False
    spec_true = [[],[]]
    for s in range(len(model.states)):
        if model.label_true[s,0] == True:
            spec_true[0].append(s)
        if model.label_true[s,1] == True:
            spec_true[1].append(s)

    visit_freq = np.ones(len(model.states))

##############################################################################

    # simulation results
    term_flag = False
    task_flag = False
    timestep = 0
    max_timestep = 250
    plan_count = 0
    div_thresh = 0.001
    n_sample = 10
    risk_thresh = 0.1
    state_hist = []
    state_hist.append(model.init_state)
    action_hist = [[],[]] # [[chosen action],[taken action]]
    infqual_hist = []
    infqual_hist.append(info_qual(model.label_belief))
    risk_hist = []

    #f1 = client.moveToPositionAsync(agent_x*grid_space, agent_y*grid_space, -30, 8, vehicle_name="Drone1")
    #f1.join()
    spec_est = [[],[]]

    while not term_flag:

        if perc_flag:
            # estimate map
            label_est = estimate_AP(model.label_belief, method='risk-averse')
            spec_est = [[],[]]
            for s in range(len(model.states)):
                if label_est[s,0] == True:
                    spec_est[0].append(s)
                if label_est[s,1] == True:
                    spec_est[1].append(s)
            print("obstacle:   ",spec_est[0])
            print("target:     ",spec_est[1])

        if replan_flag or (not replan_flag and plan_count==0):
            # find an optimal policy
            (vars_val, opt_policy) = verifier(copy.deepcopy(model), spec_est)
            print(opt_policy[0:20])
            plan_count += 1

        if act_info_flag:
            # risk evaluation
            prob_sat = stat_verifier(model,state_hist[-1],opt_policy,spec_est,n_sample)
            risk = np.abs(vars_val[state_hist[-1]] - prob_sat); print(vars_val[state_hist[-1]],prob_sat)
            risk_hist.append(risk)
            print("Risk due to Perception Uncertainty:   ",risk)

            # perception policy
            if risk > risk_thresh:
                # implement perception policy
                timestep += 1
                state = state_hist[-1]
                action = 0
            else:
                pass
        timestep += 1
        #print("Timestep:   ",timestep)
        state = state_hist[-1]
        opt_act = opt_policy[state]
        if 0 in opt_act and len(opt_act)>1:
            opt_act = opt_act[1:]

        action = np.random.choice(opt_act)

        action_hist[0].append(action)
        next_state = np.random.choice(model.states, p=model.transitions[state,action])
        # identify taken action
        for a in model.actions[model.enabled_actions[state]]:
            if model.action_effect(state,a)[0] == next_state:
                action_taken = a
        action_hist[1].append(action_taken)
        state_hist.append(next_state)

############################################################################### commented for scenario 2
        
        #gridworld gen stuff here 

        gridworld = model.get_current_gridworld(x_state_num,y_state_num)
        print(gridworld.shape)
        # compute visibility for each state

        # update belief
        next_label_belief = copy.deepcopy(model.label_belief)
        visit_freq_next = copy.deepcopy(visit_freq) + 1
        for s in model.states:
            gridworld = gridworld.reshape((-1,3))
            # update for 'obstacle'
            next_label_belief[s,0] = (next_label_belief[s,0]*visit_freq[s] + gridworld[s,1]) / visit_freq_next[s]
            # update for 'target'
            next_label_belief[s,1] = (next_label_belief[s,1]*visit_freq[s] + gridworld[s,2]) / visit_freq_next[s]
        visit_freq = copy.deepcopy(visit_freq_next)

##############################################################################
        # move to next state
        if len(state_hist) > 1:
            if next_state != state_hist[-2]:
                next_state = model.state_mapping[next_state]
                print(next_state)
                #f1 = client.moveToPositionAsync(next_state[1]*grid_space, next_state[0]*grid_space, -30, 2, vehicle_name="Drone1")
        else:
            next_state = model.state_mapping[next_state]
            print(next_state)
            #f1 = client.moveToPositionAsync(next_state[1]*grid_space, next_state[0]*grid_space, -30, 2, vehicle_name="Drone1")
        # f1.join()

        # divergence test on belief
        if div_test_flag:
            div = info_div(model.label_belief,next_label_belief)
            print("Belief Divergence:   ",div)
            if info_div(model.label_belief,next_label_belief) > div_thresh:
                replan_flag = True
            else:
                replan_flag = False
        model.label_belief = np.copy(next_label_belief)
        infqual_hist.append(info_qual(model.label_belief))

        # check task realization
        if model.label_true[state_hist[-1],0] == True:
            term_flag = True
            print("at a state with an obstacle")

        if model.label_true[state_hist[-1],1] == True:
            task_flag = True
            term_flag = True
            print("at a target state")

        if timestep == max_timestep:
            term_flag = True
            print("timestep exceeded the maximum limit")

    print("Number of Time Steps:   ",timestep)
    print("Number of Replanning:   ",plan_count)

##############################################################################
    # exit AirSim
   # airsim.wait_key('Press any key to reset to original state')

    #client.armDisarm(False)
    #client.reset()

    # that's enough fun for now. let's quit cleanly
    #client.enableApiControl(False)

##############################################################################
    infqual_hist_all.append(infqual_hist)
    risk_hist_all.append(risk_hist)
    timestep_all.append(timestep)
    plan_count_all.append(plan_count)
    task_flag_all.append(int(task_flag))

task_rate = np.mean(task_flag_all)

print(infqual_hist_all)
print(risk_hist_all)
print(timestep_all)
print(plan_count_all)
print(task_flag_all)

print("success rate: ", np.average(task_flag_all) * 100, "%")
print("average step: ", np.average(timestep_all))
print("number replan:", np.average(plan_count_all))