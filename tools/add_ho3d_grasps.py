# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Generates datasets of synthetic point clouds, grasps, and grasp robustness metrics from a Dex-Net HDF5 database for GQ-CNN training.

Author
------
Jeff Mahler

YAML Configuration File Parameters
----------------------------------
database_name : str
    full path to a Dex-Net HDF5 database
target_object_keys : :obj:`OrderedDict`
    dictionary mapping dataset names to target objects (either 'all' or a list of specific object keys)
env_rv_params : :obj:`OrderedDict`
    parameters of the camera and object random variables used in sampling (see meshpy.UniformPlanarWorksurfaceImageRandomVariable for more info)
gripper_name : str
    name of the gripper to use
"""
import argparse
import collections
import cPickle as pkl
import gc
import IPython
import json
import logging
import numpy as np
import os
import random
import shutil
import sys
import time

from autolab_core import Point, RigidTransform, YamlConfig
import autolab_core.utils as utils
from gqcnn import Grasp2D
from dexnet.visualization.visualizer2d import Visualizer2D as vis2d
from meshpy import ObjFile, RenderMode, SceneObject, UniformPlanarWorksurfaceImageRandomVariable
from perception import CameraIntrinsics, BinaryImage, DepthImage, ColorImage, GrayscaleImage

from dexnet.constants import WRITE_ACCESS, READ_ONLY_ACCESS, OBJ_EXT, READ_WRITE_ACCESS
from dexnet.database import Hdf5Database
from dexnet.grasping import GraspCollisionChecker, RobotGripper, ParallelJawPtGrasp3D
from dexnet.learning import TensorDataset
import dexnet.database.keys as dexnet_db 

# TODO find new place for the import and the obj_file write routine
import meshpy.obj_file as obj_file
import meshpy.sdf_file as sdf_file
from dexnet.database.keys import *

import oaho.database_keys as oaho_db
import datetime

import cv2

ho3d_path = '/media/robotics/Seagate\ Expansion\ Drive/lwohlhart/work/ho3d/'#'/home/robotics/work/ho3d/'
import sys
sys.path.append(ho3d_path)

ycb_path = '/home/robotics/dataset/ycb_meshes_google/objects/'
ycb_obj_path_template = ycb_path+'{}/google_512k/nontextured.obj'
ycb_sdf_path_template = ycb_path+'{}/google_512k/nontextured.sdf'
ycb_google_to_ycb_offset_alignment_template = ycb_path + '{}/align.txt'


ycb_path = '/home/robotics/dataset/ycb_meshes/'
ycb_obj_path_template = ycb_path+'{}/textured.obj'
ycb_sdf_path_template = ycb_path+'{}/textured.sdf'


# grasps_path = '/home/robotics/code/grasp-pointnet/dex-net/apps/generated_grasps'
grasps_path = '/home/robotics/code/grasp-pointnet/dex-net/apps/generated_grasps/good_grasps'


grasps_files = os.listdir(grasps_path)

import vis_HO3D as ho3d

ho3d.baseDir = ho3d_path

mean_depth_list = []
mean_depth_clip2_list = []
mean_depth_clip3_list = []
segmentation_class_distribution_list = []

if __name__ == '__main__':

    dataset_path = os.path.abspath('./data/ho3d_database_.hdf5')
    database = Hdf5Database(dataset_path, access_level=READ_WRITE_ACCESS)

    ho3d_dataset = database.create_dataset('ho3d')

    sequences = os.listdir(os.path.join(ho3d_path, 'sequences'))
    object_ids = []
    for sequence in sequences:

        sequence_path = os.path.join(ho3d_path, 'sequences', sequence)
        anno_sequence = ho3d.parseAnnoTxt(os.path.join(sequence_path, 'anno.txt'))
        object_ids.append(next(anno_sequence.itervalues())['objID'])
    
    for obj_id in set(object_ids): # ['025_mug', '019_pitcher_base']: #

        if obj_id not in ho3d_dataset.object_keys:
            obj_path = ycb_obj_path_template.format(obj_id)
            obj_mesh = obj_file.ObjFile(obj_path).read()
            if ycb_sdf_path_template is not None:
                sdf_path = ycb_sdf_path_template.format(obj_id)
                sdf_data  = sdf_file.SdfFile(sdf_path).read()
            else:
                sdf_data = None
            ho3d_dataset.create_graspable(obj_id, obj_mesh, sdf_data)
        
        
        db_object = ho3d_dataset.object(obj_id)

        object_grasp_files = [f for f in grasps_files if obj_id in f and 'pickle' in f]
        if len(object_grasp_files) == 0:
            continue
        # grasps_array = np.load(os.path.join(grasps_path, object_grasp_files[0]))
        with open(os.path.join(grasps_path, object_grasp_files[0]), 'rb') as f:
            object_grasps = pkl.load(f)

        mesh_offset = np.zeros((3))        
        mesh_offset = np.loadtxt(ycb_google_to_ycb_offset_alignment_template.format(obj_id), delimiter=' ')


        good_grasps = list(object_grasps)
        good_grasps.sort(key=lambda x: np.linalg.norm(x[1:]))
        good_grasps = np.array(good_grasps[:2000])

        n_grasps = len(good_grasps)
        grasp_dissimilarity = np.zeros((n_grasps, n_grasps))
        for i in range(n_grasps):
            for j in range(i, n_grasps):
                if i == j:
                    continue
                pi_1, pi_2 = good_grasps[i][0].endpoints
                pj_1, pj_2 = good_grasps[j][0].endpoints
                pi_c = good_grasps[i][0].center
                pj_c = good_grasps[j][0].center

                # dissim = np.linalg.norm(pi_1 - pj_1) + np.linalg.norm(pi_2 - pj_2) + np.linalg.norm(pi_c - pj_c)
                dissim = min(np.linalg.norm(pi_1 - pj_1), np.linalg.norm(pi_2 - pj_2), np.linalg.norm(pi_1 - pj_2), np.linalg.norm(pi_2 - pj_1))  + np.linalg.norm(pi_c - pj_c)
                grasp_dissimilarity[j,i] = grasp_dissimilarity[i,j] = dissim
                

        grasp_quality = - np.linalg.norm(good_grasps[:,1:].astype(np.float), axis=1)
        # temp_grasp_dissimilarity = grasp_dissimilarity.copy()
        sampled_mask = np.zeros(n_grasps)
        sampled_grasps = []
        for i in range (200):
            # grasp_uniqueness = np.sum(temp_grasp_dissimilarity, axis=0)
            grasp_uniqueness = np.dot(grasp_dissimilarity, sampled_mask / max(1, i)) 
            grasp_selection_probability = grasp_uniqueness + grasp_quality

            index = np.argmax(grasp_selection_probability)
            print(index, grasp_selection_probability[index], grasp_dissimilarity[index, sampled_grasps])
            sampled_grasps.append(index)
            sampled_mask[index] = 1
            grasp_quality[index] = -100000
            


        good_grasps = good_grasps[sampled_grasps]
        grasps = []
        metrics = {}
        for i, object_grasp in enumerate(good_grasps):
            grasp = object_grasp[0]
            grasp.center += mesh_offset
            grasp.grasp_id_ = i
            grasp.frame = grasp.frame.encode('utf8')
            grasps.append(grasp)
            metrics[str(i)] = {
                'friction': object_grasp[1],
                'canny': object_grasp[2]
            }




        # object_grasp_configurations = [
        #     {
        #         'grasp': ParallelJawPtGrasp3D(g[:10], grasp_id=i), 
        #         'metrics':{
        #             'friction': g[10], 
        #             'canny': g[11]
        #         }
        #     } for i, g in enumerate(grasps_array)]
        # grasps  = [cfg['grasp'] for cfg in object_grasp_configurations]
        print('storing grasps for ', obj_id)
        ho3d_dataset.delete_grasps(obj_id, gripper='robotiq_85')
        ho3d_dataset.store_grasps(obj_id, grasps, gripper='robotiq_85')
        ho3d_dataset.store_grasp_metrics(obj_id, metrics, gripper='robotiq_85')
        
        database.flush()
        
        # if dexnet_db.HAND_OBJECT_POSES_KEY not in db_object.keys():
        #     db_object.create_group(oaho_db.HAND_OBJECT_POSES_KEY)
        # hand_object_poses = db_object[oaho_db.HAND_OBJECT_POSES_KEY]
    database.flush()
    database.close()
