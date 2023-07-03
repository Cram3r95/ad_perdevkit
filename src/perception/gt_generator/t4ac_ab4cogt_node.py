#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

"""
Created on Mon June 26 14:26:13 2023
@author: Carlos Gómez Huélamo

AB4COGT (A Baseline for CARLA Online Ground-truth generation) ROS node

Get online GT generation without using CARLA ROS-Bridge

It can be the whole GT within a particular range (including occluded obstacles, so, non-realistic GT) or 
filtered GT using ray-casting from the LiDAR sensor, evaluating if the object is within the Field-of-View 
of our sensors (so, realistic GT)
"""

# General purpose imports

import csv
import os
import time
import pdb
import sys

# DL & Math imports

import numpy as np
import math

# ROS imports

import rospy

from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleInfo
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from derived_object_msgs.msg import Object, ObjectArray
from ad_perdevkit.msg import GT_3D_Object_list

# Custom imports

import carla
import carla_common.transforms as trans
import visualization_msgs.msg

from objects2gt import Objects2GT
from gt_publisher import GTPublisher
from gt2csv import GT2CSV
from store_data import StoreData
from av2.datasets.motion_forecasting.data_schema import ObjectType

sys.path.append("/workspace/team_code")
from catkin_ws.src.t4ac_unified_perception_layer.src.t4ac_prediction_module.get_prediction_model import get_prediction_model
from catkin_ws.src.t4ac_unified_perception_layer.src.t4ac_prediction_module.plot_qualitative_results_simulation import plot_actor_tracks, plot_predictions, get_object_type
from catkin_ws.src.t4ac_unified_perception_layer.src.t4ac_prediction_module.data import from_numpy

#######################################

# Aux functions

def get_marker(actor, color):
    """
    """
    
    marker = Marker()
    marker.header.frame_id = "/map"
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.scale.x = 10
    marker.scale.y = 10
    marker.scale.z = 10
    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.pose.orientation.w = 1.0
    marker.lifetime = rospy.Duration.from_sec(0.1)
    marker.pose.position.x = actor.get_transform().location.x
    marker.pose.position.y = actor.get_transform().location.y
    marker.pose.position.z = 0
    marker.id = actor.id
    
    return marker

def get_current_ros_pose(carla_actor):
    """
    Function to provide the current ROS pose

    :return: the ROS pose of this actor
    :rtype: geometry_msgs.msg.Pose
    """
    return trans.carla_transform_to_ros_pose(
        carla_actor.get_transform())
    
def get_current_ros_twist(carla_actor):
    """
    Function to provide the current ROS twist

    :return: the ROS twist of this actor
    :rtype: geometry_msgs.msg.Twist
    """
    return trans.carla_velocity_to_ros_twist(
        carla_actor.get_velocity(),
        carla_actor.get_angular_velocity())

def get_current_ros_accel(carla_actor):
    """
    Function to provide the current ROS accel

    :return: the ROS twist of this actor
    :rtype: geometry_msgs.msg.Twist
    """
    return trans.carla_acceleration_to_ros_accel(
        carla_actor.get_acceleration())

def stop_callback(event):
    print("Kill node. Time finished.")
    rospy.signal_shutdown("Just stopping publishing...")

# Main class (Perform detection and tracking using LiDAR-based ray-tracing to obtain the ground-truth -> prediction)

class AB4COGT():
    def __init__(self):
        """
        """
        
        self.DEBUG = False
        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        
        retry_count = 10
        while retry_count > 0:
            if len(self.world.get_actors()) > 0:
                self.ego_vehicle = self.world.get_actors().filter('vehicle.*.*')[0]
                break
            else:
                time.sleep(1.0)

        if self.DEBUG: print(">>>>>>>>>>>>>>>>> Ego-vehicle: ", self.ego_vehicle)
        
        if retry_count == 0 and len(self.world.get_actors()) == 0:
            raise ValueError('Actors not populated in time')
        
        self.list_of_ids = {}
        self.state = {}
        self.timestamp = 0
        self.safe_csv = False
        self.range = 150
        self.max_agents = 11 # including ego-vehicle (only for Decision-Making module)
        
        self.lidar_pcl = None
        self.ego_vehicle_location = None
        
        # Motion Prediction
        
        self.OBS_LEN = 50
        self.PRED_LEN = 60
        self.required_variables = 5 # id, obs_num, x, y, padding
        self.TINY_PREDICTION = False
        self.NUM_STEPS = 10 # To obtain predictions every n-th STEP
        self.NUM_PREDICTED_POSES = 4 # e.g. t+0, t+STEP, t+2*STEP, t+3*STEP
        
        self.prediction_network = get_prediction_model()

        # ROS communications
        
        self.use_filtering_as_ros_callback = False
        
        node_name = rospy.get_param("/ad_devkit/generate_perception_groundtruth_node/node_name")
        rospy.init_node(node_name, anonymous=True)
        self.rate = rospy.Rate(10)
        
        ## Subscribers
        
        camera_info_topic = rospy.get_param("ad_devkit/generate_perception_groundtruth_node/camera_info")
        self.camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
        
        self.lidar_topic = rospy.get_param("/ad_devkit/generate_perception_groundtruth_node/lidar")
        
        self.sub_location_ego = rospy.Subscriber("/t4ac/localization/pose", Odometry, self.localization_callback)
        self.sub_lidar_pcl = rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback)
        
        ## Publishers
        
        self.pub_gt = rospy.Publisher("/gt_detections", MarkerArray, queue_size=20)
        
        groundtruth = rospy.get_param("/ad_devkit/generate_perception_groundtruth_node/groundtruth")
        self.pub_groundtruth = rospy.Publisher(groundtruth, GT_3D_Object_list, queue_size=10)
        
        self.pub_predictions_marker = rospy.Publisher("/t4ac/perception/prediction/prediction_markers", MarkerArray, queue_size=10)
    
    def lidar_callback(self, lidar_msg):
        """
        """
    
        self.lidar_pcl = lidar_msg
           
    def localization_callback(self, location_msg):
        """
        """
    
        self.ego_vehicle_location = location_msg
        
    def generate_gt(self):
        """
        """
        if self.DEBUG: print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        if self.lidar_pcl and self.ego_vehicle_location:
            # Create GT generator
            
            gt_publisher = GTPublisher(self.pub_groundtruth)
            gt_generator = Objects2GT(gt_publisher, self.camera_info, self.ego_vehicle, self.use_filtering_as_ros_callback)

            # Get all objects (only walkers and vehicles) in the world within a specific range
            # More info: https://carla.readthedocs.io/en/latest/core_actors/
            
            vehicles = self.world.get_actors().filter('vehicle.*.*')
            walkers = self.world.get_actors().filter('walker.*.*') 

            aux_carla_actor_list = [vehicles, walkers]
            carla_actor_list = []
            
            for carla_actor_type_list in aux_carla_actor_list:
                for carla_actor in carla_actor_type_list:
                    if gt_generator.object_in_range(self.ego_vehicle, carla_actor, self.range):
                        carla_actor_list.append(carla_actor)
            
            # if self.DEBUG: print("CARLA actor list: ", carla_actor_list) 
            
            # Transform actor list to ObjectArray ROS message format
            
            actor_list_ros = ObjectArray()

            try:
                actor_list_ros.header = self.ego_vehicle_location.header   
            except:
                actor_list_ros.header.stamp = rospy.Time.now()

            for carla_actor in carla_actor_list:
                actor_ros = Object()
                
                actor_ros.header.frame_id = "map" # We always receive the CARLA objects in map (world) coordinates
                
                # ID
                
                actor_ros.id = carla_actor.id
                
                # Position, Velocity (linear and angular) and acceleration (linear and angular)
                
                actor_ros.pose = get_current_ros_pose(carla_actor)
                actor_ros.twist = get_current_ros_twist(carla_actor)
                actor_ros.accel = get_current_ros_accel(carla_actor)

                # Dimensions
                
                actor_ros.shape.type = SolidPrimitive.BOX
                actor_ros.shape.dimensions.extend([
                    carla_actor.bounding_box.extent.x * 2.0,
                    carla_actor.bounding_box.extent.y * 2.0,
                    carla_actor.bounding_box.extent.z * 2.0])
                
                # Classification if available in attributes (here we assume that the actor is either
                # a pedestrian or a vehicle (truck, motorcycle, bike and car are included))

                actor_ros.object_classified = True

                if carla_actor.type_id.split('.')[0] == "vehicle":
                    actor_ros.classification = Object.CLASSIFICATION_CAR
                else:
                    actor_ros.classification = Object.CLASSIFICATION_PEDESTRIAN

                actor_ros.classification_certainty = 255
                
                # Append object
                
                actor_list_ros.objects.append(actor_ros)
            
            # Store current point-cloud

            gt_generator.store_pointcloud(self.lidar_pcl)

            # Filter objects according to LiDAR ray-tracing (get realistic GT)

            filtered_objects = gt_generator.callback(actor_list_ros)
            if self.DEBUG: print("Number of filtered objects: ", len(filtered_objects.gt_3d_object_list))

            # Preprocess filtered objects as input for the Motion Prediction algorithm

            self.state.clear()
            
            # We assume that the ego-vehicle is the first object since we have previously sorted from nearest to furthest
   
            for i in range(len(filtered_objects.gt_3d_object_list)):
                filtered_obj = filtered_objects.gt_3d_object_list[i]

                if filtered_obj.type == "ego_vehicle":
                    if not "ego" in self.list_of_ids:
                        self.list_of_ids["ego"] = [[0, 0, 1] for _ in range(self.OBS_LEN)] # Initialize buffer

                    self.list_of_ids["ego"].append([filtered_obj.global_position.x, 
                                                    - filtered_obj.global_position.y,
                                                    0])
                    
                    self.state[filtered_obj.object_id] = np.array(self.list_of_ids["ego"][-50:])

                else: # Other agents
                    adv_id = filtered_obj.object_id
                    if self.DEBUG: print("Adversary ID: ", adv_id)
                    x_adv = filtered_obj.global_position.x
                    y_adv = - filtered_obj.global_position.y
                    
                    if adv_id in self.list_of_ids:
                        self.list_of_ids[adv_id].append([x_adv, y_adv, 0])
                    else:
                        self.list_of_ids[adv_id] = [[0, 0, 1] for _ in range(self.OBS_LEN)]
                        self.list_of_ids[adv_id].append([x_adv, y_adv, 0])

                    self.state[adv_id] = np.array(self.list_of_ids[adv_id][-50:])
                    if self.DEBUG: print(self.state[adv_id])
            
            # Save current observations into .csv to be predicted offline
            
            if self.safe_csv:
                self.write_csv()
            
            # Online prediction
              
            agents_info = self.preprocess_trackers(self.state)
            predictions, confidences = self.predict_agents(agents_info, self.timestamp)
            
            # Plot ROS markers
            
            if len(predictions) > 0:
                self.plot_predictions_ros_markers(predictions) 
                
            self.timestamp += 1 
    
    def plot_predictions_ros_markers(self, scenario_predictions):
        """
        """
        
        predictions_markers_list = visualization_msgs.msg.MarkerArray()
        
        for i, agent_predictions in enumerate(scenario_predictions):
            agent_predictions_marker = visualization_msgs.msg.Marker()
            agent_predictions_marker.header.frame_id = "/map"
            agent_predictions_marker.header.stamp = self.ego_vehicle_location.header.stamp
        
            agent_predictions_marker.ns = "motion_prediction_output"
            
            agent_predictions_marker.action = agent_predictions_marker.ADD
            agent_predictions_marker.lifetime = rospy.Duration.from_sec(0.2)

            agent_predictions_marker.id = i
            agent_predictions_marker.type = visualization_msgs.msg.Marker.LINE_LIST

            agent_predictions_marker.color.r = 1.0
            agent_predictions_marker.color.a = 1.0
            agent_predictions_marker.scale.x = 0.3
            agent_predictions_marker.pose.orientation.w = 1.0
            
            
        particular_monitorized_area_marker.pose.orientation.w = 1.0

        for p in area:
            point = geometry_msgs.msg.Point()

            point.x = p.x
            point.y = p.y
            point.z = 0.2

            particular_monitorized_area_marker.points.append(point)

        point = geometry_msgs.msg.Point()

        point.x = area[0].x
        point.y = area[0].y

        particular_monitorized_area_marker.points.append(point) # To close the polygon
        predictions_markers_list.markers.append(particular_monitorized_area_marker)

        self.pub_predictions_marker.publish(predictions_markers_list)
        
    def preprocess_trackers(self, trajectories):
        """
        """
        
        agents_info_array = np.zeros([0, self.required_variables])

        for key, value in trajectories.items():
            for num_obs in range(self.OBS_LEN):
                agent_info = np.array([key,
                                        num_obs,
                                        value[num_obs,0],
                                        value[num_obs,1],
                                        value[num_obs,2]])

                agents_info_array = np.vstack((agents_info_array, agent_info))

        return agents_info_array

    def predict_agents(self, agents_info_array, file_id):
        """
        """
        
        agents_id = np.unique(agents_info_array[:, 0], axis=0)

        valid_agents_info = []

        for agent_id in agents_id:
            agent_info = agents_info_array[agents_info_array[:, 0] == agent_id]

            if not (agent_info[:, -1] == 1).all():  # Avoid storing full-padded agents
                valid_agents_info.append(agent_info)
                
        # Get agents of the scene (we assume the first one represents our ego-vehicle)
        # (N agents * 50) x 5 (track_id, timestep, x, y, padding)
        # Preprocess agents (relative displacements, orientation, etc.)

        trajs, steps, track_ids, object_types = [], [], [], []
        final_predictions, final_confidences = [], []

        # TODO: Write here the corresponding object type, though at this moment it is not used
        
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

            current_step_index = steps[0].tolist().index(self.OBS_LEN-1)
            pre_current_step_index = current_step_index-1

            orig = trajs[0][current_step_index][:2].copy().astype(np.float32)
            pre = trajs[0][pre_current_step_index][:2] - orig

            theta = np.arctan2(pre[1], pre[0])
            rot = np.asarray([[np.cos(theta), -np.sin(theta)],

                              [np.sin(theta), np.cos(theta)]], np.float32)

            feats, ctrs, valid_track_ids, valid_object_types = [], [], [], []

            for traj, step, track_id, object_type in zip(trajs, steps, track_ids, object_types):

                if self.OBS_LEN-1 not in step:

                    continue

                valid_track_ids.append(track_id)
                valid_object_types.append(object_type)

                obs_mask = step < self.OBS_LEN
                step = step[obs_mask]
                traj = traj[obs_mask]
                idcs = step.argsort()
                step = step[idcs]
                traj = traj[idcs]
                feat = np.zeros((self.OBS_LEN, 3), np.float32)

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

            output = self.prediction_network(data)

            for agent_index in range(feats.shape[0]):
                agent_mm_pred = output["reg"][0][agent_index,
                                                 :, :, :].cpu().data.numpy()
                agent_cls = output["cls"][0][agent_index, :].cpu().data.numpy()

                if self.TINY_PREDICTION: # Unimodal prediction (60 x 2)
                    most_probable_mode_index = np.argmax(agent_cls)
                    agent_um_pred = agent_mm_pred[most_probable_mode_index, :, :]
                    agent_um_pred = agent_um_pred[::self.STEP,
                                                  :][:self.NUM_PREDICTED_POSES, :]
                    final_predictions.append(agent_um_pred)

                else:
                    final_predictions.append(agent_mm_pred)
                    final_confidences.append(agent_cls)

        return final_predictions, final_confidences
            
    def write_csv(self):
        """
        """
        
        results_path = "/workspace/team_code/catkin_ws/src/t4ac_unified_perception_layer/src/t4ac_prediction_module/poses"

        if not os.path.exists(results_path):
            print("Create results path folder: ", results_path)
            os.makedirs(results_path)

        with open(f'{results_path}/poses_{self.timestamp}.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            for key in self.state.keys():
                for num_obs in range(self.OBS_LEN):
                    writer.writerow([key, 
                                    num_obs, 
                                    self.state[key][num_obs][0],
                                    self.state[key][num_obs][1],
                                    self.state[key][num_obs][2]])
        
if __name__=="__main__":
    """
    """
    
    ab4cogt = AB4COGT() 
    
    time = 20 # seconds
    rospy.Timer(rospy.Duration(time), stop_callback)
    
    while not rospy.is_shutdown():
        ab4cogt.generate_gt()
        ab4cogt.rate.sleep()
        
    
    
# # Plot trajectories

# def plot_trajectories():
#     if PLOT_OBSERVATIONS:

#         # Create figure

#         fig = plt.figure(1, figsize=(8, 7))

#         ax = fig.add_subplot(111)

#         plot_actor_tracks(ax, valid_agents_info)

#         if PLOT_PREDICTIONS and predictions:

#             for agent_index in range(len(predictions)):

#                 if agent_index == 0:

#                     color = "#FAC205"

#                 else:

#                     color = "#15B01A"

#                 if TINY_PREDICTION:  # Unimodal, best score

#                     agent_um_pred = predictions[agent_index]

#                     plot_predictions(
#                         agent_um_pred, purpose="unimodal", color=color)

#                 else:

#                     agent_mm_pred = predictions[agent_index]

#                     agent_cls = confidences[agent_index]

#                     plot_predictions(agent_mm_pred, agent_cls, color=color)

#         plt.xlim([-30, 30], auto=False)

#         plt.ylim([0, 120], auto=False)

#         filename = os.path.join(
#             SAVE_DIR, ADDITIONAL_STRING+str(file_id)+".png")

#         plt.savefig(filename, bbox_inches='tight',
#                     facecolor="white", edgecolor='none', pad_inches=0)

#         plt.close('all')