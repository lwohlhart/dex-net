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

from dexnet.constants import WRITE_ACCESS, READ_ONLY_ACCESS, OBJ_EXT
from dexnet.database import Hdf5Database
from dexnet.grasping import GraspCollisionChecker, RobotGripper
from dexnet.learning import TensorDataset

# TODO find new place for the import and the obj_file write routine
import meshpy.obj_file as obj_file
import meshpy.sdf_file as sdf_file
from dexnet.database.keys import *

import oaho.database_keys as oaho_db
import datetime

import cv2

ho3d_path = '/home/robotics/work/ho3d/'
import sys
sys.path.append(ho3d_path)

ycb_path = '/home/robotics/dataset/ycb_meshes_google/objects/'
ycb_obj_path_template = ycb_path+'{}/google_512k/nontextured.obj'
ycb_sdf_path_template = ycb_path+'{}/google_512k/nontextured.sdf'


ycb_path = '/home/robotics/dataset/ycb_meshes/'
ycb_obj_path_template = ycb_path+'{}/textured.obj'
ycb_sdf_path_template = ycb_path+'{}/textured.sdf'


import vis_HO3D as ho3d

ho3d.baseDir = ho3d_path

mean_depth_list = []
mean_depth_clip2_list = []
mean_depth_clip3_list = []
segmentation_class_distribution_list = []

if __name__ == '__main__':

    dataset_path = os.path.abspath('./data/ho3d_database.hdf5')
    database = Hdf5Database(dataset_path, access_level=WRITE_ACCESS)

    ho3d_dataset = database.create_dataset('ho3d')

    sequences = os.listdir(os.path.join(ho3d_path, 'sequences'))
    for sequence in sequences:
        logging.info('Opening sequence : {}'.format(sequence))
        sequence_path = os.path.join(ho3d_path, 'sequences', sequence)
        anno_sequence = ho3d.parseAnnoTxt(os.path.join(sequence_path, 'anno.txt'))
        obj_id = next(anno_sequence.itervalues())['objID']
        
        logging.info('Processing object : {}'.format(obj_id))

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

        if oaho_db.HAND_OBJECT_POSES_KEY not in db_object.keys():
            db_object.create_group(oaho_db.HAND_OBJECT_POSES_KEY)
        hand_object_poses = db_object[oaho_db.HAND_OBJECT_POSES_KEY]

        for seq_id, anno in anno_sequence.iteritems():
            now = datetime.datetime.now()
            ts = '%04d%02d%02d%02d%02d%02d' % (now.year, now.month, now.day,now.hour, now.minute, now.second)

            hand_file = os.path.join(sequence_path,'hand/{}.obj'.format(seq_id))
            depth_file = os.path.join(sequence_path,'depth/{}.png'.format(seq_id))
            segmentation_file = os.path.join(sequence_path,'segr/{}.jpg'.format(seq_id))

            pose_name = 'seq_{}_pose_{}'.format(sequence, seq_id)

            if pose_name not in hand_object_poses.keys():
                hand_object_poses.create_group(pose_name)
                
            pose = hand_object_poses[pose_name]

            hand_mesh = obj_file.ObjFile(hand_file).read()

            if oaho_db.HAND_OBJECT_POSE_HAND_MESH_KEY not in pose.keys():
                pose.create_group(oaho_db.HAND_OBJECT_POSE_HAND_MESH_KEY)
            
            pose_hand_mesh = pose[oaho_db.HAND_OBJECT_POSE_HAND_MESH_KEY]
            pose_hand_mesh.clear()


            T_obj_cam = RigidTransform(rotation=RigidTransform.rotation_from_axis_angle(anno['objRot']),
                           translation=anno['objTrans'], from_frame='obj', to_frame='camera')
            
            T_cam_obj = T_obj_cam.inverse()

            T_hand_cam = RigidTransform(translation=anno['handTrans'], from_frame='hand', to_frame='camera')
            
            T_obj_hand = T_cam_obj * T_hand_cam 
            T_hand_obj = T_hand_cam.inverse() * T_obj_cam
            # T_hand_obj = T_obj_hand.inverse()

            if oaho_db.HAND_OBJECT_POSE_ROT_KEY in pose.attrs.keys():
                pose.attrs[oaho_db.HAND_OBJECT_POSE_ROT_KEY].clear()
            pose.attrs.create(oaho_db.HAND_OBJECT_POSE_PT_KEY, T_hand_obj.translation)

            if oaho_db.HAND_OBJECT_POSE_ROT_KEY in pose.attrs.keys():
                pose.attrs[oaho_db.HAND_OBJECT_POSE_ROT_KEY].clear()
            pose.attrs.create(oaho_db.HAND_OBJECT_POSE_ROT_KEY, T_hand_obj.rotation)
            
            if oaho_db.HAND_OBJECT_POSE_HAND_POSE_KEY in pose.attrs.keys():
                pose.attrs[oaho_db.HAND_OBJECT_POSE_HAND_POSE_KEY].clear()
            pose.attrs.create(oaho_db.HAND_OBJECT_POSE_HAND_POSE_KEY, anno['handPose'])  # TODO maybe it's anno['handJoints']
            # pose.attrs.create(oaho_db.HAND_OBJECT_POSE_HAND_POSE_KEY, oaho_scene.getHandPoseQuat())
            
            # hand_mesh_vertices_homogenous = np.hstack([hand_mesh.vertices, np.ones((hand_mesh.vertices.shape[0],1))])
            # hand_mesh_vertices = np.dot(T_obj_hand.matrix, np.transpose(hand_mesh_vertices_homogenous)).transpose()[:,:3]
            # hand_mesh_vertices = hand_mesh.vertices - hand_mesh.centroid

            pose_hand_mesh.create_dataset(oaho_db.MESH_VERTICES_KEY, data=hand_mesh.vertices)
            pose_hand_mesh.create_dataset(oaho_db.MESH_TRIANGLES_KEY, data=hand_mesh.triangles)

            if oaho_db.RENDERED_IMAGES_KEY not in pose.keys():
                pose.create_group(oaho_db.RENDERED_IMAGES_KEY)
            else:
                pose[oaho_db.RENDERED_IMAGES_KEY].clear() # TODO maybe don't clear images let's see

            rendered_images = pose[oaho_db.RENDERED_IMAGES_KEY]

            image_id = '{}_{}'.format(oaho_db.IMAGE_KEY, len(list(rendered_images.keys())))



            depth_map = ho3d.decodeDepthImg(depth_file)

            # depth_map = cv2.copyMakeBorder(depth_map, 1, 1, 1,  1, cv2.BORDER_DEFAULT)
            depth_scale = np.abs(depth_map).max()             
            # depth_map[:,0] = depth_scale
            mask = (depth_map == 0).astype(np.uint8)
            depth_map += depth_scale*mask
            # depth_map = depth_map.astype(np.float32)/depth_scale  # Has to be float32, 64 not supported.

            # depth_map = cv2.inpaint(depth_map, mask, 1, cv2.INPAINT_NS)
            # depth_map = depth_map[1:-1, 1:-1]
            # depth_map = depth_map * depth_scale


            mean_depth_list.append(np.mean(depth_map))
            mean_depth_clip2_list.append(np.mean(np.clip(depth_map, 0.0, 2.0)))
            mean_depth_clip3_list.append(np.mean(np.clip(depth_map, 0.0, 3.0)))

            segmentation_image = cv2.imread(segmentation_file)
            _, segmentation_image = cv2.threshold(segmentation_image, 100, 255, cv2.THRESH_BINARY)
            segmentation_image = cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2RGB)            
            segmentation_image = cv2.resize(segmentation_image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            segmentation_classes = np.argmax(np.dstack([0.1*np.ones(segmentation_image.shape[:2]),segmentation_image]), axis=2)

            segmentation_map = np.eye(4)[segmentation_classes]
            segmentation_class_distribution_list.append(np.mean(np.reshape(segmentation_map,(-1,4)), axis=0))



            rendered_images.create_group(image_id)
            image_data = rendered_images[image_id]

            image_data.create_dataset(oaho_db.IMAGE_DEPTH_KEY, data=depth_map)
            image_data.create_dataset(oaho_db.IMAGE_SEGMENTATION_KEY, data=segmentation_image)
            image_data.attrs.create(oaho_db.IMAGE_TIMESTAMP_KEY, np.string_(ts))
            image_data.attrs.create(oaho_db.IMAGE_FRAME_KEY, np.string_('image'))
            image_data.attrs.create(oaho_db.CAM_ROT_KEY, T_cam_obj.rotation)
            image_data.attrs.create(oaho_db.CAM_POS_KEY, T_cam_obj.translation)# pos_cam_to_obj)
            image_data.attrs.create(oaho_db.CAM_FRAME_KEY, np.string_('camera'))
            image_data.attrs.create(oaho_db.CAM_INTRINSICS_KEY, np.array([617.343, 617.343, 312.42, 241.42])) # from ho3d/README.txt


            mean_depth = np.mean(mean_depth_list)
            mean_depth_clip2 = np.mean(mean_depth_clip2_list)
            mean_depth_clip3 = np.mean(mean_depth_clip3_list)
            np.savetxt(os.path.join('data', 'ho3d_mean_depth.txt'), np.array([mean_depth, mean_depth_clip2, mean_depth_clip3]), fmt='%.4f')

            np.savetxt(os.path.join('data', 'ho3d_class_distribution.txt'), np.mean(segmentation_class_distribution_list, axis=0),  fmt='%.4f')


            logging.info('processed id: {}'.format(seq_id))

        database.flush()
        break
            
        
    database.flush()
    database.close()


