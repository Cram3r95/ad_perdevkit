#!/usr/bin/env python3.8

# -*- coding: utf-8 -*-


"""

Created on Mon May 15 13:47:58 2023

@author: Rodrigo Gutiérrez-Moreno & Carlos Gómez-Huélamo

"""


# General purpose imports


from smarts.env.wrappers.format_obs import _set_timestamp
from gym.spaces import Box
from plot_smarts import plot_actor_tracks, plot_predictions, get_object_type
from data import from_numpy
from av2.datasets.motion_forecasting.data_schema import ObjectType, TrackCategory
from av2.utils.typing import NDArrayFloat
import gym
import utils
import pdb

import sys

import pandas as pd

import git

import os

import csv

import argparse

import copy


from typing import Tuple, Optional, List, Set, Dict

from importlib import import_module


# DL & Math imports


import torch

import numpy as np


# Plot imports


import matplotlib.pyplot as plt


from pathlib import Path

from matplotlib.axes import Axes

from matplotlib.patches import Rectangle

from PIL import Image


# Custom imports


BASE_DIR = "/home/argo2goalmp"

sys.path.append(BASE_DIR)


# Global variables


PREDICT_AGENTS = True

TINY_PREDICTION = True


PLOT_OBSERVATIONS = False

PLOT_PREDICTIONS = True


STEP = 10  # To obtain predictions every nth STEP

NUM_PREDICTIONS = 4  # t+0, t+STEP, t+2*STEP, t+3*STEP


OBS_LEN = 50


RELATIVE_ROUTE = "smarts_simulator_scenarios/scenario_poses_episode_0"

SCENARIO_ROUTE = os.path.join(BASE_DIR, RELATIVE_ROUTE)

ADDITIONAL_STRING = "poses_"


SAVE_DIR = os.path.join(BASE_DIR, RELATIVE_ROUTE+"_plots")

os.makedirs(SAVE_DIR, exist_ok=True)


_STATIC_OBJECT_TYPES: Set[ObjectType] = {

    ObjectType.STATIC,

    ObjectType.BACKGROUND,

    ObjectType.CONSTRUCTION,

    ObjectType.RIDERLESS_BICYCLE,

}


_PlotBounds = Tuple[float, float, float, float]


#######################################


class Observation(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):

        super().__init__(env)

        self.n_features = 3

        self.n_features_pred = 2

        self.n_vehicles = 11

        self.n_states = 50

        self.n_states_pred = 4

        self.empty = gym.spaces.Dict({'ego': Box(0, 1, shape=(self.n_states, self.n_features)),

                                      'vehicle A': Box(0, 1, shape=(self.n_states, self.n_features)),

                                      'vehicle B': Box(0, 1, shape=(self.n_states, self.n_features)),

                                      'vehicle C': Box(0, 1, shape=(self.n_states, self.n_features)),

                                      'vehicle D': Box(0, 1, shape=(self.n_states, self.n_features)),

                                      'vehicle E': Box(0, 1, shape=(self.n_states, self.n_features)),

                                      'vehicle F': Box(0, 1, shape=(self.n_states, self.n_features)),

                                      'vehicle G': Box(0, 1, shape=(self.n_states, self.n_features)),

                                      'vehicle H': Box(0, 1, shape=(self.n_states, self.n_features)),

                                      'vehicle I': Box(0, 1, shape=(self.n_states, self.n_features)),

                                      'vehicle J': Box(0, 1, shape=(self.n_states, self.n_features))})

        self.observation_space = gym.spaces.Dict({'ego': Box(0, 1, shape=(self.n_states_pred, self.n_features_pred)),

                                                  'vehicle A': Box(0, 1, shape=(self.n_states_pred, self.n_features_pred)),

                                                  'vehicle B': Box(0, 1, shape=(self.n_states_pred, self.n_features_pred)),

                                                  'vehicle C': Box(0, 1, shape=(self.n_states_pred, self.n_features_pred)),

                                                  'vehicle D': Box(0, 1, shape=(self.n_states_pred, self.n_features_pred)),

                                                  'vehicle E': Box(0, 1, shape=(self.n_states_pred, self.n_features_pred)),

                                                  'vehicle F': Box(0, 1, shape=(self.n_states_pred, self.n_features_pred)),

                                                  'vehicle G': Box(0, 1, shape=(self.n_states_pred, self.n_features_pred)),

                                                  'vehicle H': Box(0, 1, shape=(self.n_states_pred, self.n_features_pred)),

                                                  'vehicle I': Box(0, 1, shape=(self.n_states_pred, self.n_features_pred)),

                                                  'vehicle J': Box(0, 1, shape=(self.n_states_pred, self.n_features_pred))})

        self.list_of_keys = [key for key in self.empty.keys()]

        self.list_of_ids = {}

        for num_adv in range(self.n_vehicles):

            current_adv = self.list_of_keys[num_adv]

            self.empty[current_adv] = np.zeros(
                (self.n_states, self.n_features))

            self.empty[current_adv][:, 2] = 1

        self.timestamp = 0

        self.episode_index = -1

        # Create prediction network

        # TODO: Load this dict from a file

        args = dict()

        args["model"] = "CGHNet"

        args["weight"] = "/home/argo2goalmp/stable_ckpts/results_student/50.000.ckpt"

        args["exp_name"] = "results_student"

        args["use_map"] = False

        args["use_goal_areas"] = False

        args["map_in_decoder"] = False

        args["motion_refinement"] = False

        args["distill"] = False

        model = import_module("src.%s" % args["model"])

        config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model(exp_name=args["exp_name"],

                                                                                    distill=args["distill"],

                                                                                    use_map=args["use_map"],

                                                                                    use_goal_areas=args["use_goal_areas"],

                                                                                    map_in_decoder=args["map_in_decoder"])

        # Load pretrained model

        ckpt_path = args["weight"]

        if not os.path.isabs(ckpt_path):

            ckpt_path = os.path.join(config["save_dir"], ckpt_path)

        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

        utils.load_pretrain(net, ckpt["state_dict"])

        net.eval()

        self.net = net

    def observation(self, obs: Dict[str, gym.Space]) -> np.ndarray:

        SAVE_CSV = False

        self.timestamp = self.env.get_timestamp(self.timestamp)

        file_id = f"{self.episode_index}_{self.timestamp}"

        if self.timestamp == 0:

            self.reset_obs()

        ego = obs["ego_vehicle_state"]["pos"]

        if "ego" in self.list_of_ids:

            self.list_of_ids["ego"].append([ego[0], ego[1], 0])

        else:

            self.list_of_ids["ego"] = [[0, 0, 1] for _ in range(self.n_states)]

            self.list_of_ids["ego"].append([ego[0], ego[1], 0])

        adversaries = obs["neighborhood_vehicle_states"]

        for adv in range(len(adversaries)):

            if np.any(adversaries["nghb_id"][adv]):

                adv_id = adversaries["nghb_id"][adv]

                x_adv = adversaries["pos"][adv][0]

                y_adv = adversaries["pos"][adv][1]

                if adv_id in self.list_of_ids:

                    self.list_of_ids[adv_id].append([x_adv, y_adv, 0])

                else:

                    self.list_of_ids[adv_id] = [[0, 0, 1]
                                                for _ in range(self.n_states)]

                    self.list_of_ids[adv_id].append([x_adv, y_adv, 0])

        self.state = copy.deepcopy(self.empty)

        current_adv = self.list_of_keys[0]

        self.state[current_adv] = np.array(self.list_of_ids["ego"][-50:])

        adv_index = 1

        for id_key in self.list_of_ids:

            if id_key is not "ego" and int(id_key) in adversaries["nghb_id"]:

                current_adv = self.list_of_keys[adv_index]

                self.state[current_adv] = np.array(
                    self.list_of_ids[id_key][-50:])

                adv_index += 1

        # Save raw observations in .csv

        if SAVE_CSV:
            self.write_csv(obs)

        # Process raw observations to filter full-padded agents

        agents_info_array = self.trajectories(obs)

        agents_id = np.unique(agents_info_array[:, 0], axis=0)

        valid_agents_info = []

        for agent_id in agents_id:

            agent_info = agents_info_array[agents_info_array[:, 0] == agent_id]

            if not (agent_info[:, -1] == 1).all():  # Avoid storing full-padded agents

                valid_agents_info.append(agent_info)

        predictions, confidences = self.predict_agents(
            valid_agents_info, file_id)

        # Plot trajectories

        if PLOT_OBSERVATIONS:

            # Create figure

            fig = plt.figure(1, figsize=(8, 7))

            ax = fig.add_subplot(111)

            plot_actor_tracks(ax, valid_agents_info)

            if PLOT_PREDICTIONS and predictions:

                for agent_index in range(len(predictions)):

                    if agent_index == 0:

                        color = "#FAC205"

                    else:

                        color = "#15B01A"

                    if TINY_PREDICTION:  # Unimodal, best score

                        agent_um_pred = predictions[agent_index]

                        plot_predictions(
                            agent_um_pred, purpose="unimodal", color=color)

                    else:

                        agent_mm_pred = predictions[agent_index]

                        agent_cls = confidences[agent_index]

                        plot_predictions(agent_mm_pred, agent_cls, color=color)

            plt.xlim([-30, 30], auto=False)

            plt.ylim([0, 120], auto=False)

            filename = os.path.join(
                SAVE_DIR, ADDITIONAL_STRING+str(file_id)+".png")

            plt.savefig(filename, bbox_inches='tight',
                        facecolor="white", edgecolor='none', pad_inches=0)

            plt.close('all')

        state = self.observation_space.sample()

        for num_adv in range(self.n_vehicles):

            current_adv = self.list_of_keys[num_adv]

            state[current_adv] = np.zeros(
                (self.n_states_pred, self.n_features_pred))

        if predictions:

            for num_adv in range(self.n_vehicles):

                current_adv = self.list_of_keys[num_adv]

                if num_adv < len(predictions)-1:

                    state[current_adv] = predictions[num_adv]

                else:

                    break

        self.timestamp += 1

        _set_timestamp(self.timestamp)

        return (state)

    def reset_obs(self):

        # print("------------------ New Episode ------------------")

        self.list_of_ids.clear()

        self.episode_index += 1

    def write_csv(self, obs):

        # results_path = os.path.join(

        #     os.getcwd(),"intersection","scenario_poses"

        # )

        results_path = f"/home/argo2goalmp/smarts_simulator_scenarios/scenario_poses_episode_{self.episode_index}/"

        if not os.path.exists(results_path):

            print("Create results path folder: ", results_path)

            os.makedirs(results_path)

        with open(f'{results_path}/poses_{self.timestamp}.csv', 'w', newline='') as file:

            writer = csv.writer(file, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

            for num_adv in range(self.n_vehicles):

                current_adv = self.list_of_keys[num_adv]

                for num_obs in range(self.n_states):

                    if num_adv == 0:

                        writer.writerow([0,

                                         num_obs,

                                         self.state[current_adv][num_obs][0],

                                         self.state[current_adv][num_obs][1],

                                         self.state[current_adv][num_obs][2]])

                    else:

                        if obs["neighborhood_vehicle_states"]["nghb_id"][num_adv-1] == 0:

                            id_ = -1

                        else:

                            id_ = obs["neighborhood_vehicle_states"]["nghb_id"][num_adv-1]

                        writer.writerow([id_,

                                        num_obs,

                                        self.state[current_adv][num_obs][0],

                                        self.state[current_adv][num_obs][1],

                                        self.state[current_adv][num_obs][2]])

    def trajectories(self, obs):

        agents_info_array = np.zeros([0, 5])

        for num_adv in range(self.n_vehicles):

            current_adv = self.list_of_keys[num_adv]

            for num_obs in range(self.n_states):

                if num_adv == 0:

                    agent_info = np.array([0,

                                           num_obs,

                                           self.state[current_adv][num_obs][0],

                                           self.state[current_adv][num_obs][1],

                                           self.state[current_adv][num_obs][2]])

                else:

                    if obs["neighborhood_vehicle_states"]["nghb_id"][num_adv-1] == 0:

                        id_ = -1

                    else:

                        id_ = obs["neighborhood_vehicle_states"]["nghb_id"][num_adv-1]

                    agent_info = np.array([id_,

                                           num_obs,

                                           self.state[current_adv][num_obs][0],

                                           self.state[current_adv][num_obs][1],

                                           self.state[current_adv][num_obs][2]])

                agents_info_array = np.vstack((agents_info_array, agent_info))

        return agents_info_array

    def predict_agents(self, valid_agents_info, file_id):

        # Get agents of the scene (we assume the first one represents our ego-vehicle)

        # 550 (11 agents = ego + 10 adversaries) x 5 (track_id, timestep, x, y, padding)

        # Preprocess agents (relative displacements, orientation, etc.)

        trajs, steps, track_ids, object_types = [], [], [], []

        final_predictions, final_confidences = [], []

        # All agents in the Smarts simulator are assumed to be vehicles
        object_type = ObjectType.VEHICLE

        # agent_info: 0 = track_id, 1 = timestep, 2 = x, 3 = y, 4 = padding

        for agent_info in valid_agents_info:

            non_padded_agent_info = agent_info[agent_info[:, -1] == 0]

            trajs.append(non_padded_agent_info[:, 2:4])

            steps.append(non_padded_agent_info[:, 1].astype(np.int64))

            # Our ego-vehicle is always the first agent of the scenario
            track_ids.append(non_padded_agent_info[0, 0])

            object_types.append(get_object_type(object_type))

        if trajs[0].shape[0] > 1:  # Our ego-vehicle must have at least two observations

            current_step_index = steps[0].tolist().index(OBS_LEN-1)

            pre_current_step_index = current_step_index-1

            orig = trajs[0][current_step_index][:2].copy().astype(np.float32)

            pre = trajs[0][pre_current_step_index][:2] - orig

            theta = np.arctan2(pre[1], pre[0])

            rot = np.asarray([[np.cos(theta), -np.sin(theta)],

                              [np.sin(theta), np.cos(theta)]], np.float32)

            feats, ctrs, valid_track_ids, valid_object_types = [], [], [], []

            for traj, step, track_id, object_type in zip(trajs, steps, track_ids, object_types):

                if OBS_LEN-1 not in step:

                    continue

                valid_track_ids.append(track_id)

                valid_object_types.append(object_type)

                obs_mask = step < 50

                step = step[obs_mask]

                traj = traj[obs_mask]

                idcs = step.argsort()

                step = step[idcs]

                traj = traj[idcs]

                feat = np.zeros((50, 3), np.float32)

                feat[step, :2] = np.matmul(
                    rot, (traj[:, :2] - orig.reshape(-1, 2)).T).T

                feat[step, 2] = 1.0

                ctrs.append(feat[-1, :2].copy())

                feat[1:, :2] -= feat[:-1, :2]

                feat[step[0], :2] = 0

                feats.append(feat)

            feats = np.asarray(feats, np.float32)

            ctrs = np.asarray(ctrs, np.float32)

            data = dict()

            # OBS: Our network must receive a list (batch) per value of the dictionary. In this case, we only want

            # to analyze a single scenario, so the values must be introduced as lists of 1 element, indicating

            # batch_size = 1

            data['scenario_id'] = [file_id]

            data['track_ids'] = [valid_track_ids]

            data['object_types'] = [np.asarray(valid_object_types, np.float32)]

            data['feats'] = [feats]

            data['ctrs'] = [ctrs]

            data['orig'] = [orig]

            data['theta'] = [theta]

            data['rot'] = [rot]

            # Recursively transform numpy.ndarray to torch.Tensor
            data = from_numpy(data)

            output = self.net(data)

            for agent_index in range(feats.shape[0]):

                agent_mm_pred = output["reg"][0][agent_index,
                                                 :, :, :].cpu().data.numpy()

                agent_cls = output["cls"][0][agent_index, :].cpu().data.numpy()

                if TINY_PREDICTION:

                    most_probable_mode_index = np.argmax(agent_cls)

                    # 60 x 2
                    agent_um_pred = agent_mm_pred[most_probable_mode_index, :, :]

                    agent_um_pred = agent_um_pred[::STEP,
                                                  :][:NUM_PREDICTIONS, :]

                    final_predictions.append(agent_um_pred)

                else:

                    final_predictions.append(agent_mm_pred)

                    final_confidences.append(agent_cls)

        return final_predictions, final_confidences
