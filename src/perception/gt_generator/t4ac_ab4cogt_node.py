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
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from derived_object_msgs.msg import Object, ObjectArray
from ad_perdevkit.msg import GT_3D_Object_list
from std_msgs.msg import Header
from t4ac_msgs.msg import BEV_detections_list

# Custom imports

import carla
import carla_common.transforms as trans

from objects2gt import Objects2GT
from gt_publisher import GTPublisher
from gt2csv import GT2CSV
from store_data import StoreData
from ab4cogt2sort_functions import get_gtlist_as_BEV
from av2.datasets.motion_forecasting.data_schema import ObjectType

sys.path.append("/workspace/team_code")

from generic_modules.utils import str2bool
from catkin_ws.src.t4ac_unified_perception_layer.src.t4ac_prediction_module.motion_predictor import Motion_Predictor 
from tf.transformations import euler_from_quaternion, quaternion_from_euler

#######################################

# IMPORTANT

# TODO: Check additional padding when the object is not observed. An additional
# study should be done to know how many observations steps had the relevant actors
# (those with observation in t=0)

# Aux functions

LIFETIME = 0.5
TIME_SCENARIO = 120 # Shutdown after these seconds once the node has started to collect
                  # world data. If 0, it is assumed that the user will indicate when
                  # data collection is finished
NUM_COLOURS = 32
COLOURS = np.random.rand(NUM_COLOURS,3) / 1.25
APPLY_RANDOM_COLOUR = False

colour_classes = {"Car":[0,0.2,1],
                  "Truck":[0,0.2,1],
                  "Other_Vehicle":[0,0.2,1],
                  "Pedestrian":[0.6,0,0.1],
                  "Motorcycle":[0.6,0,0.1],
                  "Bike":[0.6,0,0.1],
                  "Cyclist":[0.6,0,0.1]
                  }

def get_observations_marker(actor_info, id_types):
    """
    """

    marker = Marker()
    marker.header.frame_id = "/map"
    # marker.type = marker.LINE_STRIP
    marker.type = marker.POINTS
    marker.action = marker.ADD
    marker.ns = "agents_observations"
    marker.id = int(actor_info[0,0])
    
    marker.scale.x = 0.4
    marker.pose.orientation.w = 1.0
    
    marker.color.a = 0.5
    
    if APPLY_RANDOM_COLOUR:
        colour = COLOURS[int(actor_info[0,0])%NUM_COLOURS]
    else:
        colour = colour_classes[id_types[int(actor_info[0,0])]]
        
    marker.color.r = colour[0]
    marker.color.g = colour[1]
    marker.color.b = colour[2]
    
    marker.lifetime = rospy.Duration.from_sec(LIFETIME)
    
    observations = actor_info[actor_info[:,-1] == 1.0] # get non-padded observations
    
    for i in range(observations.shape[0]):
        point = Point()

        point.x = observations[i,2]
        point.y = observations[i,3]
        point.z = 0

        marker.points.append(point)
    
    return marker

def get_detection_marker(actor):
    """
    """
                                
    # Current position

    marker_position = Marker()
    marker_position.header.frame_id = "/map"
    marker_position.type = marker_position.CUBE
    marker_position.action = marker_position.ADD
    marker_position.ns = "agents_detection"
    marker_position.id = actor.object_id
    
    if actor.type == "Pedestrian" or actor.type == "Bike" or actor.type == "Cyclist" or actor.type == "Motorcycle":
        marker_position.scale.x = actor.dimensions.x * 1.5
        marker_position.scale.y = actor.dimensions.y * 1.5
    else:
        marker_position.scale.x = actor.dimensions.x
        marker_position.scale.y = actor.dimensions.y
        
    marker_position.scale.z = actor.dimensions.z
    
    marker_position.color.a = 1.0
    
    if APPLY_RANDOM_COLOUR:
        colour = COLOURS[marker_position.id%NUM_COLOURS]
    else:
        colour = colour_classes[actor.type]
        
    marker_position.color.r = colour[0]
    marker_position.color.g = colour[1]
    marker_position.color.b = colour[2]
    
    quaternion = quaternion_from_euler(0,0,actor.rotation_z)
    marker_position.pose.orientation.x = quaternion[0]
    marker_position.pose.orientation.y = quaternion[1]
    marker_position.pose.orientation.z = quaternion[2]
    marker_position.pose.orientation.w = quaternion[3]
    
    marker_position.lifetime = rospy.Duration.from_sec(LIFETIME)
    
    marker_position.pose.position.x = actor.global_position.x
    marker_position.pose.position.y = actor.global_position.y
    marker_position.pose.position.z = 0
    
    # Identification
    
    marker_id_text = Marker()
    marker_id_text.header.frame_id = "/map"
    marker_id_text.type = marker_id_text.TEXT_VIEW_FACING
    marker_id_text.action = marker_id_text.ADD
    marker_id_text.ns = "agents_detection_id"
    marker_id_text.id = actor.object_id
    
    if actor.type == "Pedestrian" or actor.type == "Bike" or actor.type == "Cyclist" or actor.type == "Motorcycle":
        marker_id_text.scale.z = 0.9
    else:
        marker_id_text.scale.z = 1.6
    
    marker_id_text.text = str(actor.object_id)
    
    marker_id_text.color.a = 1.0

    # White identifier
    
    marker_id_text.color.r = 1
    marker_id_text.color.g = 1
    marker_id_text.color.b = 1
    
    quaternion = quaternion_from_euler(0,0,actor.rotation_z)
    marker_id_text.pose.orientation.x = quaternion[0]
    marker_id_text.pose.orientation.y = quaternion[1]
    marker_id_text.pose.orientation.z = quaternion[2]
    marker_id_text.pose.orientation.w = quaternion[3]
    
    marker_id_text.lifetime = rospy.Duration.from_sec(LIFETIME)
    
    marker_id_text.pose.position.x = actor.global_position.x
    marker_id_text.pose.position.y = actor.global_position.y
    marker_id_text.pose.position.z = 2
    
    return marker_position, marker_id_text

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
    """
    """
    print("AB4COGT: Stop collecting data")
    rospy.signal_shutdown("Just stopping publishing...")

# Main class (Perform detection and tracking using LiDAR-based ray-tracing to obtain the ground-truth -> prediction)

class AB4COGT():
    def __init__(self):
        """
        """
        
        # Aux variables
        
        self.DEBUG = False
        self.RAY_TRACING_FILTER = True
        self.PREPROCESS_TRACKERS = True
        self.init_time = time.time()
        self.SCENARIO_INITIALIZATION_TIME = 0 # seconds. TODO: Is there a flag that indicates
        # when the star of the scenario? (i.e. When the CARLA world starts moving)
        self.init_stop_callback = False
        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        
        self.list_of_ids = {}
        self.state = {}
        self.timestamp = 0
        self.range = 150
        self.max_agents = 11 # including ego-vehicle (only for Decision-Making module)
        
        self.lidar_pcl = None
        self.ego_vehicle_location = None
        self.ego_vehicle = None
        
        self.safe_csv = False
   
        routes = dict()

        routes[1] = "route22_town03_training"
        routes[2] = "route15_town04_testing"
        routes[3] = "route1_town01_training_threshold_3"
        routes[4] = "route27_town03_training"

        SCENARIO_ID = 3
        
        root_path = "/workspace/team_code/catkin_ws/src/t4ac_unified_perception_layer/src/t4ac_prediction_module/data/datasets/CARLA"
        self.results_path = f"{root_path}/scenario_{routes[SCENARIO_ID]}/poses"
        
        # Motion Prediction
        
        self.USE_PREDICTION = False
        self.motion_predictor = Motion_Predictor(self.USE_PREDICTION)

        # ROS communications
        
        node_name = rospy.get_param("/ad_devkit/generate_perception_groundtruth_node/node_name")
        rospy.init_node(node_name, anonymous=True)
        # self.rate = rospy.Rate(10)
        
        ## Publishers
        
        # Aux GT for SmartMOT. There should be a single GT for the whole architecture
        
        self.pub_BEV_groundtruth_obstacles = rospy.Publisher("/t4ac/perception/detection/BEV_groundtruth_obstacles",\
                                                             BEV_detections_list, queue_size=10)

        self.pub_non_filtered_groundtruth = rospy.Publisher("/ad_devkit/generate_perception_groundtruth_node/perception_non_filtered_groundtruth", 
                                                            ObjectArray, 
                                                            queue_size=10)

        groundtruth_topic = rospy.get_param("/ad_devkit/generate_perception_groundtruth_node/pub_groundtruth")
        self.pub_groundtruth = rospy.Publisher(groundtruth_topic, GT_3D_Object_list, queue_size=10)
        
        groundtruth_markers_topic = rospy.get_param("/ad_devkit/generate_perception_groundtruth_node/pub_groundtruth_markers")
        self.pub_gt_marker = rospy.Publisher(groundtruth_markers_topic, MarkerArray, queue_size=10)
        
        ## Subscribers
        
        camera_info_topic = rospy.get_param("ad_devkit/generate_perception_groundtruth_node/camera_info")
        self.camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
        
        self.lidar_topic = rospy.get_param("/ad_devkit/generate_perception_groundtruth_node/lidar")
        
        self.sub_location_ego = rospy.Subscriber("/t4ac/localization/pose", Odometry, self.localization_callback)
        self.sub_lidar_pcl = rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback)
    
        # Aux synchronization
        
        flag_processed_data_topic = "/t4ac/perception/flag_processed_data"
        self.sub_flag_processed_data = rospy.Subscriber(flag_processed_data_topic, Header, self.flag_processed_data_callback)
        
        self.previous_simulation_iteration_stamp = rospy.Time.now()
        time.sleep(1)
        self.current_simulation_iteration_stamp = rospy.Time.now() # Current simulation iteration stamp must be 
                                                                   # higher to previous simulation iteration
                                                                   # to run step
                                                                   
    # Class functions
    
    def write_csv(self, obs, timestamp):
        """
        """

        if not os.path.exists(self.results_path):
            print("Create results path folder: ", self.results_path)
            os.makedirs(self.results_path)

        with open(f'{self.results_path}/poses_{timestamp}.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            for key in obs.keys():
                for num_obs in range(len(obs[key])):
                    writer.writerow([key, 
                                     num_obs, 
                                     obs[key][num_obs][0],
                                     obs[key][num_obs][1],
                                     obs[key][num_obs][2]])
                        
    # Callbacks
    
    def flag_processed_data_callback(self, flag_processed_data_msg):
        """
        """

        self.current_simulation_iteration_stamp = flag_processed_data_msg.stamp
        
        if (self.current_simulation_iteration_stamp > self.previous_simulation_iteration_stamp):
            self.previous_simulation_iteration_stamp = self.current_simulation_iteration_stamp
            self.generate_gt()
            
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
        
        # while (self.current_simulation_iteration_stamp <= self.previous_simulation_iteration_stamp):
        #     continue

        # self.previous_simulation_iteration_stamp = self.current_simulation_iteration_stamp
        
        # ###########
        
        # Set traffic lights to green

        # traffic_lights = self.world.get_actors().filter('traffic.traffic_light*')
        # for traffic_light in traffic_lights:
        #     traffic_light.set_state(carla.TrafficLightState.Green)
        
        # ###########
        
        start_adperdevkit = rospy.get_param('/t4ac/operation_modes/start_adperdevkit')
        if type(start_adperdevkit) != bool:
            start_adperdevkit = str2bool(start_adperdevkit)
            
        if self.DEBUG: print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        if len(self.world.get_actors().filter('vehicle.*.*')) > 0:
            # We assume that the ego-vehicle is the first element of the vehicles list 
            self.ego_vehicle = self.world.get_actors().filter('vehicle.*.*')[0]
                              
        if (self.lidar_pcl 
           and self.ego_vehicle_location 
           and self.ego_vehicle
           and (start_adperdevkit
                or (self.SCENARIO_INITIALIZATION_TIME > 0 
                                 and time.time() - self.init_time >= self.SCENARIO_INITIALIZATION_TIME))):

            # List current objects in the scenario buffer
            
            actors_scenario = list(self.list_of_ids.keys())

            if self.DEBUG: print(">>>>>>>>>>>>>>>>> Ego-vehicle: ", self.ego_vehicle)
           
            # Create GT generator
            # TODO: Is it required to create these objects per timestamp?
            
            gt_publisher = GTPublisher(self.pub_groundtruth)
            gt_generator = Objects2GT(gt_publisher, self.camera_info, self.ego_vehicle)

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
            
            self.pub_non_filtered_groundtruth.publish(actor_list_ros)
            
            # Aux for SmartMOT
            
            groundtruth_BEV_list = get_gtlist_as_BEV(actor_list_ros, self.ego_vehicle.id)
            self.pub_BEV_groundtruth_obstacles.publish(groundtruth_BEV_list)
            
            # Ray-tracing-based filter
            
            if self.RAY_TRACING_FILTER:
                # Store current point-cloud
                
                gt_generator.store_pointcloud(self.lidar_pcl)

                # Filter objects according to LiDAR ray-tracing (get realistic GT)

                filtered_objects = gt_generator.callback(actor_list_ros)
                if self.DEBUG: print("Number of filtered objects: ", len(filtered_objects.gt_3d_object_list))

                ## Publish filtered object as ROS markers
                
                gt_detections_marker_list = MarkerArray()
                
                for num_object, filtered_object in enumerate(filtered_objects.gt_3d_object_list):
                    if num_object > 0: # Avoid plotting the ego-vehicle, we already have the URDF marker
                        orientation_q = self.ego_vehicle_location.pose.pose.orientation
                        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                        (ego_roll, ego_pitch, ego_yaw) = euler_from_quaternion(orientation_list)
                        
                        filtered_object.rotation_z = ego_yaw - filtered_object.rotation_z 
                        
                        marker_position, marker_id_text = get_detection_marker(filtered_object)
                        gt_detections_marker_list.markers.append(marker_position)
                        gt_detections_marker_list.markers.append(marker_id_text)
                
                # Preprocess filtered objects as input for the Motion Prediction algorithm

                if self.PREPROCESS_TRACKERS:
                    self.state.clear()
                    id_type = {}
                    
                    # We assume that the ego-vehicle is the first object since we have previously sorted from nearest to furthest
                                        
                    for i in range(len(filtered_objects.gt_3d_object_list)):
                        filtered_obj = filtered_objects.gt_3d_object_list[i]

                        # OBS: If a timestep i-th has not been truly observed, that particular observation (x,y,binary_flag) 
                        # is padded (that is, third dimension set to 0). Otherwise, set to 1
                        
                        # TODO: Is this required? You know the identifier of the ego
                        
                        if filtered_obj.type == "ego_vehicle":
                            if not "ego" in self.list_of_ids:
                                self.list_of_ids["ego"] = [[0, 0, 0] for _ in range(self.motion_predictor.OBS_LEN)] # Initialize buffer

                            self.list_of_ids["ego"].append([filtered_obj.global_position.x, 
                                                            filtered_obj.global_position.y,
                                                            1])
                            
                            self.state[filtered_obj.object_id] = np.array(self.list_of_ids["ego"][-self.motion_predictor.OBS_LEN:])
                            id_type[filtered_obj.object_id] = filtered_obj.type
                            
                        else: # Other agents
                            adv_id = filtered_obj.object_id
                            if self.DEBUG: print("Adversary ID: ", adv_id)
                            x_adv = filtered_obj.global_position.x
                            y_adv = filtered_obj.global_position.y
                            
                            if adv_id in self.list_of_ids:
                                self.list_of_ids[adv_id].append([x_adv, y_adv, 1])
                            else:
                                self.list_of_ids[adv_id] = [[0, 0, 0] for _ in range(self.motion_predictor.OBS_LEN)]
                                self.list_of_ids[adv_id].append([x_adv, y_adv, 1])

                            self.state[adv_id] = np.array(self.list_of_ids[adv_id][-self.motion_predictor.OBS_LEN:])
                            id_type[filtered_obj.object_id] = filtered_obj.type
                            
                            if self.DEBUG: print("Agents state: ", self.state[adv_id])

                        if (self.timestamp > 0 
                           and (filtered_obj.object_id in actors_scenario
                                or "ego" in actors_scenario)):
                            if filtered_obj.type == "ego_vehicle":
                                agent_to_remove = actors_scenario.index("ego")
                            else:
                                agent_to_remove = actors_scenario.index(filtered_obj.object_id)
                            actors_scenario.pop(agent_to_remove)

                    # Set 0,0,0 (padding) for actors that are in the list_of_ids buffer but 
                    # they have not been observed in the current timestamp

                    # TODO: Is this correct?
                    
                    # if self.timestamp > 0:
                    #     for non_observed_actor_id in actors_scenario:
                    #         self.list_of_ids[non_observed_actor_id].append([0, 0, 0])
                        
                    # Save current observations into .csv to be predicted offline
                    
                    if self.safe_csv:
                        self.write_csv(self.state, self.timestamp)
                        
                        # if TIME_SCENARIO > 0 and not self.init_stop_callback: 
                        #     print("AB4COGT: Start collection data")
                        #     rospy.Timer(rospy.Duration(TIME_SCENARIO), stop_callback)
                        #     self.init_stop_callback = True
            
                    # Preprocess trackers
                    
                    valid_agents_info, valid_agents_id = self.motion_predictor.preprocess_trackers(self.state)
                        
                    if valid_agents_info: # Agents with more than a certain number of observations
                        # Plot observations ROS markers
                        
                        for num_object, valid_agent_info in enumerate(valid_agents_info):
                            if num_object > 0: # Avoid plotting the ego-vehicle, we already have the URDF marker
                                marker = get_observations_marker(valid_agent_info, 
                                                                 id_type)
                                gt_detections_marker_list.markers.append(marker)

                        self.pub_gt_marker.publish(gt_detections_marker_list)
                        
                    # Online prediction

                    if self.USE_PREDICTION and valid_agents_info: 
                        # Predict agents
                        
                        predictions, confidences = self.motion_predictor.predict_agents(valid_agents_info, self.timestamp)

                        # Plot predictions ROS markers

                        if len(predictions) > 0:
                            self.motion_predictor.plot_predictions_ros_markers(predictions, 
                                                                            confidences, 
                                                                            valid_agents_id, 
                                                                            self.ego_vehicle_location.header.stamp,
                                                                            COLOURS,
                                                                            apply_colour=APPLY_RANDOM_COLOUR,
                                                                            lifetime=LIFETIME)         
                            
            self.timestamp += 1 
             
if __name__=="__main__":
    """
    """
    
    ab4cogt = AB4COGT() 
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down AD-PerDevKit node")