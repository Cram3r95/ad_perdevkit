#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

# General purpose imports

import pdb
import math
import numpy as np
from scipy.spatial.distance import euclidean
from gt_publisher import GT_Object
import time
from shapely.geometry import box

# ROS imports

import rospy
import ros_numpy
from carla_msgs.msg import CarlaEgoVehicleInfo
from sensor_msgs.msg import PointCloud2
from derived_object_msgs.msg import ObjectArray
from tf.transformations import euler_from_quaternion
import std_msgs

# Suscribers and Publisher classes

from gt_publisher import GTPublisher
from transformation_functions import *

class Objects2GT():

    def __init__(self, gt_publisher, camera_info, vehicle_info):

        self.classification_list = ["Unknown", "Unknown_Small",
                                    "Unknown_Medium", "Unknown_Big",
                                    "Pedestrian", "Bike",
                                    # "General_Vehicle", # Bike, Motorcycle, Car, Truck are included
                                    "Car", "Truck",
                                    "Motorcycle", "Other_Vehicle",
                                    "Barrier", "Sign"]

        self.gt_publisher = gt_publisher
        self.first_time = 0
        self.camera_info = camera_info
        self.ego_vehicle_id = vehicle_info.id
        
        self.pointcloud = None
        self.RAY_TRACING = True

        self.intrisic_matrix = ros_intrinsics(self.camera_info.P)
        self.camera_resolution = (self.camera_info.width, self.camera_info.height)

    def store_pointcloud(self, pointcloud):
 
        cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud)
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

        points = np.zeros((len(cloud_array['x']),3))
        points[...,0] = cloud_array['x']
        points[...,1] = cloud_array['y']
        points[...,2] = cloud_array['z']

        self.pointcloud = points

    def object_in_range(self, ego_vehicle, adversary, range):
        """
        """
        
        if math.sqrt(pow(ego_vehicle.get_transform().location.x - adversary.get_transform().location.x, 2) + 
                     pow(ego_vehicle.get_transform().location.y - adversary.get_transform().location.y, 2)) < range:
                     return True; return False
                 
    def callback(self, carla_objects_msg):
        DEBUG = False
        
        if DEBUG: print("Preliminary objects: ", len(carla_objects_msg.objects))
        start_time = time.time()

        max_distance = rospy.get_param("/ad_devkit/generate_perception_groundtruth_node/max_distance_lidar")
        limit_front = max_distance
        limit_back = -max_distance
        limit_left = -max_distance
        limit_right = max_distance

        header = carla_objects_msg.header
        self.gt_publisher.new_msg(header)

        number_objects = len(carla_objects_msg.objects) # Including the ego-vehicle
        if number_objects > 0:
            if self.first_time == 0:
                self.first_time = 1
                self.first_seq_value = carla_objects_msg.header.seq
                
            ### Find ego_vehicle position, size and rotation ###

            i = 0
            identity = carla_objects_msg.objects[i].id
            
            while identity != self.ego_vehicle_id:
                i += 1
                identity = carla_objects_msg.objects[i].id
                
            xyz_ego = carla_objects_msg.objects[i].pose.position
            quat_ego = carla_objects_msg.objects[i].pose.orientation
            v_ego = carla_objects_msg.objects[i].twist.linear

            location_ego = [xyz_ego.x, xyz_ego.y, xyz_ego.z]
            
            published_obj = 0

            quaternion_ego = np.array((quat_ego.x, quat_ego.y, quat_ego.z, quat_ego.w))
            heading_ego = euler_from_quaternion(quaternion_ego)
            R = np.dot(np.dot(rotx(heading_ego[0]), roty(heading_ego[1])), rotz(-heading_ego[2]))

            # Where to store the objects temporarily
            objs = []

            for i in range(len(carla_objects_msg.objects)):

                obj = GT_Object()

                obj.object_id = carla_objects_msg.objects[i].id

                xyz = carla_objects_msg.objects[i].pose.position
                location = [xyz.x, xyz.y, xyz.z]
                
                obj.l, obj.w, obj.h = carla_objects_msg.objects[i].shape.dimensions
                obj.type = self.classification_list[carla_objects_msg.objects[i].classification]
                    
                if obj.object_id != self.ego_vehicle_id:			
                    # Global position
                    
                    obj.global_position_x, obj.global_position_y, obj.global_position_z = (location[0],
                                                                                           location[1],
                                                                                           location[2])

                    location_local = (np.asarray(location) - np.asarray(location_ego)).tolist()
                    
                    location_local = np.dot(R, location_local)

                    location_lidar = rospy.get_param("/t4ac/tf/base_link_to_lidar_tf")[:3] # x,y,z

                    # Local position
                    
                    obj.position_x, obj.position_y, obj.position_z = (location_local[0], 
                                                                      location_local[1],
                                                                      location_local[2] - location_lidar[2] + obj.h/2)

                    ### Rotation ###

                    quat_xyzw = carla_objects_msg.objects[i].pose.orientation
                    quaternion = np.array((quat_xyzw.x, quat_xyzw.y, quat_xyzw.z, quat_xyzw.w))
                    heading_object = heading_ego[2] - euler_from_quaternion(quaternion)[2]
                    
                    # Rotation base on KITTI. If using CARLA, do not substract np.pi/2!!
                    
                    # heading = normalizeAngle(heading - np.pi/2)
                    obj.rotation_z = heading_object

                    ### Reduce CARLA errors ###

                    # Detect bycicles
                    if (obj.type == "Car" and obj.w < 1):
                        obj.type = "Cyclist"

                    # The center of the pedestrian is the head in CARLA so move the center down
                    if (obj.type == "Pedestrian"):
                        obj.position_z = obj.position_z - obj.h/2

                    # Increase the heigth on cyclist by 3/4
                    # if (obj.type == "Cyclist"):
                    #     obj.position_z, obj.h = obj.position_z+obj.h*3/8, obj.h*7/4
                
                    ### Filter based on the distance from the bounding box ###

                    if (location_local[0] > limit_back) and (location_local[0] < limit_front) \
                        and (location_local[1] > limit_left) and (location_local[1] < limit_right):

                        ### Filter based on lidar points ###

                        n_points_in_bb = 0
                        
                        if self.RAY_TRACING and self.pointcloud is not None:
                            # Visibility approximation (without rotation)
                            # (Using a bigger bounding box to filter most of the points)

                            max_distance = math.sqrt((obj.l/2)**2+(obj.w/2)**2)

                            f_visible_bb_height = (lambda bb, points: abs(np.array(points[:,2]) - bb[2]) <= bb[5]/2)
                            f_visible_bb_radius = (lambda bb, points: np.sqrt((bb[0] - np.array(points[:,0])) ** 2 +\
                                                                            (bb[1] - np.array(points[:,1])) ** 2) <= max_distance)

                            bb_3d = (obj.position_x, obj.position_y, obj.position_z, obj.l, obj.w, obj.h, obj.rotation_z)
                            
                            self_pointcloud = self.pointcloud
                            mask1 = f_visible_bb_height(bb_3d, self_pointcloud)
                            filtered_pointcloud_height = self_pointcloud[mask1]

                            mask2 = f_visible_bb_radius(bb_3d, filtered_pointcloud_height)
                            filtered_pointcloud_radius = filtered_pointcloud_height[mask2]

                            # Prepare visibility execution with rotation
                            f_visible_bb = (lambda point3D, bb3D: pointInBB3D_rotation(point3D, bb3D))
                            f_visible_bb_bb3D = (lambda point3D: f_visible_bb(point3D, bb_3d))

                            # Parallel execution of the visibility
                            points_in_bb = map(f_visible_bb_bb3D, filtered_pointcloud_radius)

                            # Count number of points
                            n_points_in_bb = int(np.add.reduce(tuple(points_in_bb)))
                        
                        if n_points_in_bb > 0 or not self.RAY_TRACING:
                            
                            published_obj += 1

                            ### Local position, heading and velocities ###

                            vel_lin = carla_objects_msg.objects[i].twist.linear # Global 
                            vel_ang = carla_objects_msg.objects[i].twist.angular # Global

                            local_velocity = (np.asarray([vel_lin.x, vel_lin.y, vel_lin.z]) - np.asarray([v_ego.x, v_ego.y, v_ego.z])).tolist()
                            local_velocity = np.dot(R, local_velocity)
                            
                            obj.velocity_x, obj.velocity_y, obj.velocity_z = local_velocity[0], local_velocity[1], local_velocity[2]
                            
                            obj.global_velocity_x, obj.global_velocity_y, obj.global_velocity_z = (vel_lin.x,
                                                                                                   vel_lin.y,
                                                                                                   vel_lin.z)

                            obj.global_angular_velocity_x, obj.global_angular_velocity_y, obj.global_angular_velocity_z = (vel_ang.x,
                                                                                                                           vel_ang.y,
                                                                                                                           vel_ang.z)
                             
                            location_camera = rospy.get_param("/t4ac/tf/base_link_to_camera_center_tf")[:3] # x,y,z
                            obj.alpha = normalizeAngle(math.atan2(location_local[1]-location_camera[1], location_local[0]-location_camera[0]))

                            ### World-camera geometry ###

                            location_lidar_to_camera = np.subtract(location_camera, location_lidar)
                            obj_camera = np.subtract(location_lidar_to_camera, [-obj.position_x, obj.position_y, obj.position_z])
                            bb_3d = (obj_camera[1], obj_camera[2], obj_camera[0], obj.h, obj.w, obj.l, obj.rotation_z)
                            vertices = bb3DVertices(bb_3d)
                            
                            points = np.array([vertices[:,0], vertices[:,1], vertices[:,2], vertices[:,3]]).reshape((4,8))

                            projection_matrix = np.dot(self.intrisic_matrix, points)

                            bb3D_pixel = projection_matrix[:2,:] / projection_matrix[2,:].reshape((1,8))

                            bb2D_pixel = np.matrix([[round(np.amin(bb3D_pixel[0]), 0), round(np.amin(bb3D_pixel[1]), 0)],
                                                    [round(np.amax(bb3D_pixel[0]), 0), round(np.amax(bb3D_pixel[1]), 0)]])
                            
                            ### Visibility of the objects from the camera ###

                            # Check visibility from camera
                            box_camera = box(0,0,self.camera_resolution[0],self.camera_resolution[1])
                            box_object = box(bb2D_pixel[0,0],bb2D_pixel[0,1],bb2D_pixel[1,0],bb2D_pixel[1,1])
                            ioa = box_object.intersection(box_camera).area / box_object.area
                            obj.truncated = 1 - ioa
                            if obj.position_x < 0 or ioa < 0.3 or box_object.area/box_camera.area > 0.5 : # or ioa < 0.3:
                                bb2D_pixel = np.matrix([[-1, -1], [-1, -1]])
                            else:
                                # Set limits of BB inside FOV
                                bb2D_pixel[0,0], bb2D_pixel[0,1] = max(bb2D_pixel[0,0], 0), max(bb2D_pixel[0,1], 0)
                                bb2D_pixel[1,0], bb2D_pixel[1,1] = min(bb2D_pixel[1,0], self.camera_resolution[0]), min(bb2D_pixel[1,1], self.camera_resolution[1])

                            # Create BB 2D
                            obj.bounding_box_2D_size_x = bb2D_pixel[1,0] - bb2D_pixel[0,0]
                            obj.bounding_box_2D_size_y = bb2D_pixel[1,1] - bb2D_pixel[0,1]
                            obj.bounding_box_2D_center_x = (bb2D_pixel[0,0] + bb2D_pixel[1,0]) / 2
                            obj.bounding_box_2D_center_y = (bb2D_pixel[0,1] + bb2D_pixel[1,1]) / 2

                            # Store temporarily object visible from camera to analyze to occlusion
                            if (obj.bounding_box_2D_center_x == -1):
                                self.gt_publisher.add_object(obj)
                            else:
                                objs.append(obj)
                else: # Optional -> Append ego-vehicle information
                    # OBS: We assume default values for the ego-vehicle except for the position, dimensions, 
                    # velocity and rotation

                    published_obj += 1
                    
                    obj.type = "ego_vehicle"

                    obj.position_x = 0 # Since the ego-vehicle is the origin in local coordinates for each frame
                    obj.position_y = 0
                    obj.position_z = 0

                    # Global position
                    
                    obj.global_position_x, obj.global_position_y, obj.global_position_z = (location[0],
                                                                                           location[1],
                                                                                           location[2])

                    obj.rotation_z = heading_ego[2]

                    obj.velocity_x, obj.velocity_y, obj.velocity_z = 0, 0, 0

                    obj.global_velocity_x, obj.global_velocity_y, obj.global_velocity_z = (v_ego.x,
                                                                                           v_ego.y,
                                                                                           v_ego.z)
                    
                    self.gt_publisher.add_object(obj)
                               
            # Sort objects by distance (if reverse = True, sort from furthest to nearest object to the ego-vehicle. Otherwise,
            # from nearest to furthest)
            
            objs.sort(reverse = False, key = (lambda x: np.sqrt(x.position_x**2 + x.position_y**2 +x.position_z**2)))

            # Calculate the occlusion for all objects visible from the camera

            for i in range(len(objs)):
                box_obj = box(objs[i].bounding_box_2D_center_x - objs[i].bounding_box_2D_size_x/2,
                            objs[i].bounding_box_2D_center_y - objs[i].bounding_box_2D_size_y/2,
                            objs[i].bounding_box_2D_center_x + objs[i].bounding_box_2D_size_x/2,
                            objs[i].bounding_box_2D_center_y + objs[i].bounding_box_2D_size_y/2)
                box_obj_area_init = box_obj.area
                for obj_ in objs[i+1:]:
                    box_obj_ = box(obj_.bounding_box_2D_center_x - obj_.bounding_box_2D_size_x/2,
                                obj_.bounding_box_2D_center_y - obj_.bounding_box_2D_size_y/2,
                                obj_.bounding_box_2D_center_x + obj_.bounding_box_2D_size_x/2,
                                obj_.bounding_box_2D_center_y + obj_.bounding_box_2D_size_y/2)
                    if(box_obj.intersects(box_obj_) == True):
                        box_obj = (box_obj.symmetric_difference(box_obj_)).difference(box_obj_)

                objs[i].occluded = box_obj.area / box_obj_area_init
                if objs[i].occluded > 0.8:
                    objs[i].occluded = 0
                elif objs[i].occluded > 0.4:
                    objs[i].occluded = 1
                elif objs[i].occluded > 0:
                    objs[i].occluded = 2
                else:
                    objs[i].occluded = 3
                        
                self.gt_publisher.add_object(objs[i])
                
            if published_obj == 0:
                self.gt_publisher.add_empty_object()

        else:
            self.gt_publisher.add_empty_object()

        if DEBUG: print("--- %s seconds ---" % (time.time() - start_time))

        self.gt_publisher.publish_groundtruth()
        
        return self.gt_publisher.object_list