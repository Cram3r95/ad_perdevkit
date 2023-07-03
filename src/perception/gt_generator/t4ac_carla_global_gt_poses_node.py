#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import rospy
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Odometry

import carla
import numpy as np
import csv
import os
import math
import time

class CarlaGT():
    def __init__(self):
        # Carla Config
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

        if retry_count == 0 and len(self.world.get_actors()) == 0:
            raise ValueError('Actors not populated in time')
        
        self.list_of_ids = {}
        self.state = {}
        self.timestamp = 0
        self.safe_csv = False
        self.range = 150
        self.n_states = 50
        
        rospy.init_node('gt_node', anonymous=True)
        self.rate = rospy.Rate(10) # 20hz
        self.pub_gt = rospy.Publisher("/gt_detections", MarkerArray, queue_size=20)
        self.localization_sub = rospy.Subscriber("/t4ac/localization/pose", Odometry, self.localization_callback)
            
    def publish_detections(self):
        print("detections")
        markerArray = MarkerArray()

        vehicles = self.world.get_actors().filter('vehicle.*.*')
        for vehicle in vehicles:
            marker = self.get_marker(vehicle, (1,1,0))
            markerArray.markers.append(marker)

        walkers = self.world.get_actors().filter('walker.*.*') 
        for walker in walkers:
            marker = self.get_marker(walker, (1,0,0))
            markerArray.markers.append(marker)
            
        self.pub_gt.publish(markerArray)
    
    def get_marker(self, actor, color):
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
    
    def localization_callback(self, pose):
        # Ego Vehicle
        if "ego" in self.list_of_ids:
            self.list_of_ids["ego"].append([self.ego_vehicle.get_transform().location.x, 
                                            - self.ego_vehicle.get_transform().location.y, 
                                            0])
        else:
            self.list_of_ids["ego"] = [[0, 0, 1] for _ in range(self.n_states)]
            self.list_of_ids["ego"].append([self.ego_vehicle.get_transform().location.x, 
                                            - self.ego_vehicle.get_transform().location.y, 
                                            0]) 
        
        self.state.clear()
        self.state["0"] = np.array(self.list_of_ids["ego"][-50:])
        
        # Adversaries]
        vehicles = self.world.get_actors().filter('vehicle.*.*')
        for vehicle in vehicles:
            if(vehicle.id != self.ego_vehicle.id and self.vehicle_in_range(vehicle)):
                adv_id = vehicle.id
                x_adv = vehicle.get_transform().location.x
                y_adv = - vehicle.get_transform().location.y
                
                if adv_id in self.list_of_ids:
                    self.list_of_ids[adv_id].append([x_adv, y_adv, 0])
                else:
                    self.list_of_ids[adv_id] = [[0, 0, 1] for _ in range(self.n_states)]
                    self.list_of_ids[adv_id].append([x_adv, y_adv, 0])
                
                self.state[adv_id] = np.array(self.list_of_ids[adv_id][-50:])

        if self.safe_csv:
            self.write_csv()
                    
        self.timestamp += 1    

    def vehicle_in_range(self, adversary):
        if math.sqrt(pow(self.ego_vehicle.get_transform().location.x - adversary.get_transform().location.x ,2) + 
                     pow(self.ego_vehicle.get_transform().location.y - adversary.get_transform().location.y ,2)) < self.range:
                     return True; return False 
        
    def write_csv(self):
        
        results_path = "/workspace/team_code/catkin_ws/src/t4ac_unified_perception_layer/src/poses"

        if not os.path.exists(results_path):
            print("Create results path folder: ", results_path)
            os.makedirs(results_path)

        with open(f'{results_path}/poses_{self.timestamp}.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
            for key in self.state.keys():
                for num_obs in range(self.n_states):
                    writer.writerow([key, 
                                    num_obs, 
                                    self.state[key][num_obs][0],
                                    self.state[key][num_obs][1],
                                    self.state[key][num_obs][2]])
        
if __name__=="__main__":
    
    carla_gt = CarlaGT() 
    
    while not rospy.is_shutdown():
        print("ss")
        carla_gt.publish_detections()
        carla_gt.rate.sleep()


