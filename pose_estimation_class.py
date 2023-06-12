
import numpy as np
import os
import cv2
import json 
import time
import torch
import argparse
import kornia as K
import pydegensac
import kornia.feature as KF
import matplotlib.pyplot as plt
from kornia.feature import laf_from_center_scale_ori as get_laf
from copy import deepcopy
from extract_patches.core import extract_patches

from affnet_descriptor import detect_affnet_descriptors, match_snn

class FeatureDescriptor:
    pass

def mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

def convertQuaternionToMatrix(w, x, y, z):
    rotation_v = np.array([
       1.0 - 2.0 * y * y - 2.0 * z * z,
       2.0 * x * y - 2.0 * w * z,
       2.0 * x * z + 2.0 * w * y,
       2.0 * x * y + 2.0 * w * z,
       1.0 - 2.0 * x * x - 2.0 * z * z,
       2.0 * y * z - 2.0 * w * x,
       2.0 * x * z - 2.0 * w * y,
       2.0 * y * z + 2.0 * w * x,
       1.0 - 2.0 * x * x - 2.0 * y * y
    ], dtype=np.float32)
    return rotation_v.reshape((3, 3)) # np.array([[fx, 0, cx],

class PoseEstimation:
    '''
    Class for estimating pose of query image using AffNet based approach

    Methods
    -------
    '''
    def __init__(self): #, query_path, views_path):
        '''
        Initialize data containers
        '''
        self.NFEATS = 5000 # 2000 # 
        self.RATIO = 0.45 # factor for resize images
        self.F_CONF = 0.98 # 0.95
        self.query_img = None
        self.query_pose = None
        self.view_pose = None
        self.view_images = []
        self.view_image_files = []
        self.view_names = []
        self.views_inliers = []
        self.view_inliers_numbers = []
        self.views_kps = []
        self.views_descriptors = []
        self.views_keypoints = []
        self.query_des = []
    
    def set_view_images(self, path):
        '''
        Set path to view image 
        '''
        self.views_path = path

    def set_query_image(self, path):
        '''
        Set path to query image 
        '''
        self.query_path = path
    
    def load_query_image(self):
        self.query_img = cv2.imread(self.query_path)
        width, height = self.query_img.shape[:2]
        dsize = (int(width * self.RATIO), int(height * self.RATIO))
        self.query_img = cv2.resize(self.query_img, dsize)
        
        #print("query image shape")
        #print(self.query_img.shape)
        
        # Load metatdata for query image
        query_dir_path = os.path.dirname(self.query_path)
        query_filename = os.path.basename(self.query_path).split('.')[0] + '.json'
        query_json_file = os.path.join(query_dir_path, query_filename)
        print('Load query metadata')
        query_calib, query_pose = self.load_image_metadata(query_json_file)
        self.query_pose = query_pose
        print('Query pose')
        print(self.query_pose)
        
        self.K_q = np.array([[query_calib['fx'], 0, query_calib['cx']],
                        [0, query_calib['fy'], query_calib['cy']],
                        [0, 0, 1]], dtype=np.float32) # double) # )

    def load_image_metadata(self, json_path):
        '''
        Load metadata for query image
        '''
        # load query pose
        print('Loading image metadata')
        json_data = json.load(open(json_path))
        calib_info = json_data["calibration"]
        #print(calib_info)
        fx = float(calib_info["fx"])
        fy = float(calib_info["fy"])
        cx = float(calib_info["cx"])
        cy = float(calib_info["cy"])
        calib_dict = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy }
        #print(calib_dict)
        # load query pose
        pose_json = json_data['pose']
        origin = pose_json['origin']
        rotation = pose_json['rotation']
        # gt_t = np.array(gt_origin) #.reshape(-1, 1)
        pose_dict = {'rotation': rotation, 'origin': origin}
        #print(pose_dict)
        return (calib_dict, pose_dict)

    def load_view_images(self):
        '''
        Load view images from folder
        '''
        print('Loading view images...')
        view_image_files = [f for f in os.listdir(self.views_path) if f.endswith('jpg')]
        self.view_names = [f.split('.')[0] for f in os.listdir(self.views_path) if f.endswith('jpg')]
        self.view_images = [cv2.imread(os.path.join(self.views_path, f)) for f in view_image_files]
        # self.view_images = self.view_images[:3] #10 # 80] # [:15]
        # print(self.view_names)

    def find_match(self):
        '''
        Find the best match for query image among view images usinf AffNet feature descriptors
        '''
        dev = torch.device('cpu')

        #print("query image shape")
        #print(self.query_img.shape)
        
        query_kp, query_des, As1 = detect_affnet_descriptors(self.query_img, self.NFEATS, dev)
        self.query_kp = query_kp
        self.query_des = query_des
        
        for i, view_img in enumerate(self.view_images):
            if i % 20 == 0:
                print('%s samples complete' % str(i-1))
            # resize
            width, height = view_img.shape[:2]

            dsize = (int(width * self.RATIO), int(height * self.RATIO))

            view_img = cv2.resize(view_img, dsize)
            view_kp, view_des, As2 = detect_affnet_descriptors(view_img, self.NFEATS, dev)
            # print('view %s has descriptors: %s' % (i, len(view_kp)))
            thresh = 0.4 
            tentatives = match_snn(self.query_des, view_des, thresh) # 0.85)
            # print('view %s has inliers: %s' % (i, len(tentatives)))
            self.views_keypoints.append(view_kp)
            self.views_descriptors.append(view_des)
            self.views_inliers.append(tentatives)
            self.view_inliers_numbers.append(len(tentatives))
        
        print('Looking for best match')
        max_inliers = 0
        self.best_view_index = 0
        for i, view_img in enumerate(self.view_images):
            if self.view_inliers_numbers[i] > max_inliers:
                max_inliers = self.view_inliers_numbers[i]
                self.best_view_index = i
        
        self.best_match_inliers_number = max_inliers 
        #print('best_view_index: ', self.best_view_index)
        #print('best view: ', self.view_names[self.best_view_index])
        self.best_view_kp = self.views_keypoints[self.best_view_index]
        # Read camera parameters for best view
        view_json_file = os.path.join(self.views_path, '%s.json' % self.view_names[self.best_view_index]) # args.query.split('.')[0] + '.json'
        #print('view_json_file: ', view_json_file)
        # print(json_file)
    
        view_calib, view_pose = self.load_image_metadata(view_json_file)
        self.view_pose = view_pose
        self.K_v = np.array([[view_calib['fx'], 0, view_calib['cx']],
                        [0, view_calib['fy'], view_calib['cy']],
                        [0, 0, 1]], dtype=np.float32) 


    def estimate_pose(self) -> (dict, int):
        '''
        Estimate the pose for the query image using the best matching image pose through fundamental matrix
        '''
        self.load_query_image()
        self.load_view_images()
        self.find_match()
        
        # Load point correspondences
        pts1 = []
        pts2 = []
        best_match_inliers = self.views_inliers[self.best_view_index]
        best_match_inliers = best_match_inliers[:100]
        for i, match in enumerate(best_match_inliers):
            pts1.append(self.query_kp[match.queryIdx].pt)
            pts2.append(self.best_view_kp[match.trainIdx].pt)
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        dist_coeffs = None 
        thresh = 0.0005 
        F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_LMEDS, confidence=self.F_CONF) 
        E = self.K_v.T @ F @ self.K_q
        # Use SVD to recover pose
        w,u,vt = cv2.SVDecomp(np.mat(E))   
        if np.linalg.det(u) < 0:
            u *= -1.0
        if np.linalg.det(vt) < 0:
            vt *= -1.0 
        #Find R and T from Hartley & Zisserman
        W=np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
        R_m = np.mat(u) * W * np.mat(vt)
        print('R_m')
        print(R_m)
        print('View pose')
        print(self.view_pose)
        origin = self.view_pose['origin']
        rotation = self.view_pose['rotation']
        t = np.array(origin).reshape(-1, 1)
        # Convert quaternion to Mat
        R = convertQuaternionToMatrix(*rotation)
        # Calculate camera pose for query image 
        # t_q = t + np.dot(R.T, best_pose['t']) # transpose()
        t_q = np.linalg.inv(R_m) @ t 
        print('t_q')
        print(t_q)
        R_q = R @ R_m # best_pose['R']
        # R_q = R @ best_pose['R']
        print('R_q')
        # print(R_q.shape)
        print(R_q)
        self.result_pose = {'R': R_q, 't': t_q}
        self.calculate_pose_error()
        return (self.result_pose, self.best_match_inliers_number)
    
    def calculate_pose_error(self):
        '''
        Calculating pose error using MAE metric
        '''
        print('calculating error')
        gt_origin = self.query_pose['origin']
        gt_t = np.array(gt_origin)
        gt_rotation = self.query_pose['rotation']
        gt_R = convertQuaternionToMatrix(*gt_rotation)
        
        pred_t = self.result_pose['t'].reshape(3, 1)
        #print(gt_t)
        t_error = mae(gt_t, pred_t)
        print('pose error: ', t_error)
        R_error = mae(gt_R, self.result_pose['R'])
        print('rotation error: ', R_error)
