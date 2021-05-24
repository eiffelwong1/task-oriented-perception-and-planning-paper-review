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


# define simulation parameters
n_iter = 100
infqual_hist_all = []
risk_hist_all = []
timestep_all = []
plan_count_all = []
task_flag_all = []
time_all = []

for iter in range(n_iter):
    print("iter: ", iter)
    # create problem setting
    model = MDP('gridworld')
    model.semantic_representation(prior_belief='random')
    perc_flag = False
    bayes_flag = False
    replan_flag = False
    div_test_flag = False
    act_info_flag = False
    spec_true = [[],[]]
    for s in range(len(model.states)):
        if model.label_true[s,0] == True:
            spec_true[0].append(s)
        if model.label_true[s,1] == True:
            spec_true[1].append(s)

    time_start = time.time()
    # simulate: plan, exploit
    (state_hist,action_hist,infqual_hist,risk_hist,timestep,plan_count,task_flag,l_true, l_belief) = simulation(
            model,spec_true,
            perc_flag,bayes_flag,replan_flag,div_test_flag,act_info_flag)

    infqual_hist_all.append(infqual_hist)
    risk_hist_all.append(risk_hist)
    timestep_all.append(timestep)
    plan_count_all.append(plan_count)
    task_flag_all.append(task_flag)
    time_all.append(time.time() - time_start)

task_rate = np.mean(task_flag_all)
timestep_all = np.array(timestep_all)
task_flag_all = np.array(task_flag_all)
title_posfix = " in obstacle dense environment"
title = f"random map sample for perc with {'no' if not bayes_flag else ''} update{' + replan' if replan_flag else''}{' + div' if div_test_flag else''}{' + info' if act_info_flag else''}" + title_posfix
if not perc_flag:
    title = f"random map sample for no perc" + title_posfix

print(title)
#print("average risk: ", np.average(risk_hist_all[-1]))
print("success rate: ", np.average(task_flag_all) * 100, "%")
print("average step: ", np.average(timestep_all))
succ = timestep_all[task_flag_all]
fail = timestep_all[np.invert(task_flag_all)]
print("average succ step, ", np.mean(succ))
print("average fail step, ", np.mean(fail))
print("number replan:", np.average(plan_count_all))
print("average time taken: ", np.average(time_all))

print(title)
print(np.average(task_flag_all) * 100, "\%", np.average(timestep_all), np.mean(succ), np.mean(fail), np.average(plan_count_all), np.average(time_all), sep=" & ")

map_shape = (10,10)
total = map_shape[0] * map_shape[1]
ground_true = np.zeros((map_shape[0], map_shape[1],3))
ground_true[:,:,0:2] = np.array(l_true).reshape((map_shape[0], map_shape[1],2))

l_belief_ob = l_belief[:,0].reshape(map_shape)
l_belief_ta = l_belief[:,1].reshape(map_shape)

fig, ax = plt.subplots( nrows=1, ncols=3 , figsize=(7,3))
ax[0].title.set_text('ground truth')
ax[0].imshow(ground_true)
ax[1].title.set_text('belief for obstical')
ax[1].imshow(l_belief_ob)
ax[2].title.set_text('belief for target')
ax[2].imshow(l_belief_ta)

fig.suptitle(title)
plt.savefig(f"map_h2_{int(perc_flag)}{int(bayes_flag)}{int(replan_flag)}{int(div_test_flag)}{int(act_info_flag)}.png")