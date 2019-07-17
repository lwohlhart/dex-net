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

from dexnet.constants import READ_ONLY_ACCESS, OBJ_EXT
from dexnet.database import Hdf5Database
from dexnet.grasping import GraspCollisionChecker, RobotGripper
from dexnet.learning import TensorDataset

# TODO find new place for the import and the obj_file write routine
import meshpy.obj_file as obj_file
from dexnet.database.keys import *


import tensorflow as tf

# TODO temp just for debugging
import cv2
from skimage.draw import polygon

try:
    from dexnet.visualization import DexNetVisualizer3D as vis
except:
    logging.warning('Failed to import DexNetVisualizer3D, visualization methods will be unavailable')

logging.root.name = 'dex-net'

# seed for deterministic behavior when debugging
SEED = 197561

# name of the grasp cache file
CACHE_FILENAME = 'grasp_cache.pkl'

class GraspInfo(object):
    """ Struct to hold precomputed grasp attributes.
    For speeding up dataset generation.
    """
    def __init__(self, grasp, collision_free, phi=0.0):
        self.grasp = grasp
        self.collision_free = collision_free
        self.phi = phi

def generate_oaho_dataset(dataset_path,
                           database,
                           target_object_keys,
                           env_rv_params,
                           gripper_name,
                           config):
    """
    Generates a GQ-CNN TensorDataset for training models with new grippers, quality metrics, objects, and cameras.

    Parameters
    ----------
    dataset_path : str
        path to save the dataset to
    database : :obj:`Hdf5Database`
        Dex-Net database containing the 3D meshes, grasps, and grasp metrics
    target_object_keys : :obj:`OrderedDict`
        dictionary mapping dataset names to target objects (either 'all' or a list of specific object keys)
    env_rv_params : :obj:`OrderedDict`
        parameters of the camera and object random variables used in sampling (see meshpy.UniformPlanarWorksurfaceImageRandomVariable for more info)
    gripper_name : str
        name of the gripper to use
    config : :obj:`autolab_core.YamlConfig`
        other parameters for dataset generation

    Notes
    -----
    Required parameters of config are specified in Other Parameters

    Other Parameters
    ----------------    
    images_per_stable_pose : int
        number of object and camera poses to sample for each stable pose
    stable_pose_min_p : float
        minimum probability of occurrence for a stable pose to be used in data generation (used to prune bad stable poses
    
    gqcnn/crop_width : int
        width, in pixels, of crop region around each grasp center, before resize (changes the size of the region seen by the GQ-CNN)
    gqcnn/crop_height : int
        height, in pixels,  of crop region around each grasp center, before resize (changes the size of the region seen by the GQ-CNN)
    gqcnn/final_width : int
        width, in pixels,  of final transformed grasp image for input to the GQ-CNN (defaults to 32)
    gqcnn/final_height : int
        height, in pixels,  of final transformed grasp image for input to the GQ-CNN (defaults to 32)

    table_alignment/max_approach_table_angle : float
        max angle between the grasp axis and the table normal when the grasp approach is maximally aligned with the table normal
    table_alignment/max_approach_offset : float
        max deviation from perpendicular approach direction to use in grasp collision checking
    table_alignment/num_approach_offset_samples : int
        number of approach samples to use in collision checking

    collision_checking/table_offset : float
        max allowable interpenetration between the gripper and table to be considered collision free
    collision_checking/table_mesh_filename : str
        path to a table mesh for collision checking (default data/meshes/table.obj)
    collision_checking/approach_dist : float
        distance, in meters, between the approach pose and final grasp pose along the grasp axis
    collision_checking/delta_approach : float
        amount, in meters, to discretize the straight-line path from the gripper approach pose to the final grasp pose

    tensors/datapoints_per_file : int
        number of datapoints to store in each unique tensor file on disk
    tensors/fields : :obj:`dict`
        dictionary mapping field names to dictionaries specifying the data type, height, width, and number of channels for each tensor

    debug : bool
        True (or 1) if the random seed should be set to enforce deterministic behavior, False (0) otherwise
    vis/collision_checking : bool
        True (or 1) if the collision checking procedure should be displayed in openrave (for debugging)
    vis/candidate_grasps : bool
        True (or 1) if the collision free candidate grasps should be displayed in 3D (for debugging)
    vis/rendered_images : bool
        True (or 1) if the rendered images for each stable pose should be displayed (for debugging)
    vis/grasp_images : bool
        True (or 1) if the transformed grasp images should be displayed (for debugging)
    """
    # read data gen params
    output_dir = dataset_path
    gripper = RobotGripper.load(gripper_name)
    image_samples_per_stable_pose = config['images_per_stable_pose']
    stable_pose_min_p = config['stable_pose_min_p']
    
    # read gqcnn params
    gqcnn_params = config['gqcnn']
    im_crop_height = gqcnn_params['crop_height']
    im_crop_width = gqcnn_params['crop_width']
    im_final_height = gqcnn_params['final_height']
    im_final_width = gqcnn_params['final_width']
    cx_crop = float(im_crop_width) / 2
    cy_crop = float(im_crop_height) / 2

    # open database
    dataset_names = target_object_keys.keys()
    datasets = [database.dataset(dn) for dn in dataset_names]

    # set target objects
    for dataset in datasets:
        if target_object_keys[dataset.name] == 'all':
            target_object_keys[dataset.name] = dataset.object_keys

    # setup grasp params
    table_alignment_params = config['table_alignment']
    min_grasp_approach_offset = -np.deg2rad(table_alignment_params['max_approach_offset'])
    max_grasp_approach_offset = np.deg2rad(table_alignment_params['max_approach_offset'])
    max_grasp_approach_table_angle = np.deg2rad(table_alignment_params['max_approach_table_angle'])
    num_grasp_approach_samples = table_alignment_params['num_approach_offset_samples']

    phi_offsets = []
    if max_grasp_approach_offset == min_grasp_approach_offset:
        phi_inc = 1
    elif num_grasp_approach_samples == 1:
        phi_inc = max_grasp_approach_offset - min_grasp_approach_offset + 1
    else:
        phi_inc = (max_grasp_approach_offset - min_grasp_approach_offset) / (num_grasp_approach_samples - 1)
                                                            
    phi = min_grasp_approach_offset
    while phi <= max_grasp_approach_offset:
        phi_offsets.append(phi)
        phi += phi_inc

    # setup collision checking
    coll_check_params = config['collision_checking']
    approach_dist = coll_check_params['approach_dist']
    delta_approach = coll_check_params['delta_approach']
    table_offset = coll_check_params['table_offset']

    table_mesh_filename = coll_check_params['table_mesh_filename']
    if not os.path.isabs(table_mesh_filename):
        table_mesh_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', table_mesh_filename)
    table_mesh = ObjFile(table_mesh_filename).read()
    
    # set tensor dataset config
    tensor_config = config['tensors']
    tensor_config['fields']['depth_images']['height'] = im_final_height
    tensor_config['fields']['depth_images']['width'] = im_final_width
    tensor_config['fields']['segmentation_images']['height'] = im_final_height
    tensor_config['fields']['segmentation_images']['width'] = im_final_width

    tensor_config['fields']['quality_maps']['height'] = im_final_height
    tensor_config['fields']['quality_maps']['width'] = im_final_width

    tensor_config['fields']['angle_cos_maps']['height'] = im_final_height
    tensor_config['fields']['angle_cos_maps']['width'] = im_final_width
    tensor_config['fields']['angle_sin_maps']['height'] = im_final_height
    tensor_config['fields']['angle_sin_maps']['width'] = im_final_width
    tensor_config['fields']['width_maps']['height'] = im_final_height
    tensor_config['fields']['width_maps']['width'] = im_final_width

    # add available metrics (assuming same are computed for all objects)
    metric_names = []
    dataset = datasets[0]
    obj_keys = dataset.object_keys
    if len(obj_keys) == 0:
        raise ValueError('No valid objects in dataset %s' %(dataset.name))
    
    obj = dataset[obj_keys[0]]
    grasps = dataset.grasps(obj.key, gripper=gripper.name)
    grasp_metrics = dataset.grasp_metrics(obj.key, grasps, gripper=gripper.name)
    # metric_names = grasp_metrics[grasp_metrics.keys()[0]].keys()
    # for metric_name in metric_names:
    #     tensor_config['fields'][metric_name] = {}
    #     tensor_config['fields'][metric_name]['dtype'] = 'float32'

    # init tensor dataset
    tensor_dataset = TensorDataset(output_dir, tensor_config)
    tensor_datapoint = tensor_dataset.datapoint_template

    # init tf record writer

    
    tf_record_filename = os.path.join(output_dir, 'oaho_synth.tfrecord')
    tf_train_record_filename = os.path.join(output_dir, 'oaho_synth_train.tfrecord')
    tf_val_record_filename = os.path.join(output_dir, 'oaho_synth_val.tfrecord')
    tf_test_record_filename = os.path.join(output_dir, 'oaho_synth_test.tfrecord')

    tf_train_record_writer = tf.python_io.TFRecordWriter(tf_train_record_filename)
    tf_val_record_writer = tf.python_io.TFRecordWriter(tf_val_record_filename)
    tf_test_record_writer = tf.python_io.TFRecordWriter(tf_test_record_filename)

    tf_record_writers = [ tf_train_record_writer, tf_val_record_writer, tf_test_record_writer ]

    

    # setup log file
    experiment_log_filename = os.path.join(output_dir, 'dataset_generation.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    hdlr = logging.FileHandler(experiment_log_filename)
    hdlr.setFormatter(formatter)
    logging.getLogger().addHandler(hdlr)
    root_logger = logging.getLogger()

    # copy config
    out_config_filename = os.path.join(output_dir, 'dataset_generation.json')
    ordered_dict_config = collections.OrderedDict()
    for key in config.keys():
        ordered_dict_config[key] = config[key]
    with open(out_config_filename, 'w') as outfile:
        json.dump(ordered_dict_config, outfile)

    # 1. Precompute the set of valid grasps for each stable pose:
    #    i) Perpendicular to the table
    #   ii) Collision-free along the approach direction

    tvt_target = [0.8, 0.1, 0.1]
    tvt_choice = np.random.choice([0, 1, 2], 1000, p=tvt_target)
    record_index = 0
    # create grasps dict
    candidate_grasps_dict = {}
    
    # loop through datasets and objects
    for dataset in datasets:
        logging.info('Reading dataset %s' %(dataset.name))
        for obj in dataset:
            if obj.key not in target_object_keys[dataset.name]:
                continue

            # init candidate grasp storage
            candidate_grasps_dict[obj.key] = {}

            # setup collision checker
            collision_checker = GraspCollisionChecker(gripper, view=config['vis']['collision_checking'])
            collision_checker.set_graspable_object(obj)
            
            
            # read in the hand_object configurations for the mesh
            hand_object_poses = dataset.hand_object_poses(obj.key)
            for i, hand_object_pose in enumerate(hand_object_poses):
                # render images if stable pose is valid
                if not hand_object_pose.rendered_images or hand_object_pose.p < stable_pose_min_p:
                    continue
                candidate_grasps_dict[obj.key][hand_object_pose.id] = {}
                
                # write mesh as an obj file for collision checking
                hand_obj_filename = os.path.join(dataset.cache_dir, obj.key, 'hand_poses')
                if not os.path.exists(hand_obj_filename):
                    os.makedirs(hand_obj_filename)
                hand_obj_filename = os.path.join(hand_obj_filename, hand_object_pose.id + OBJ_EXT)
                if not os.path.exists(hand_obj_filename):
                    hof = obj_file.ObjFile(hand_obj_filename)
                    hof.write(hand_object_pose.hand_mesh)
                
                # setup hand in collision checker
                T_obj_hand = hand_object_pose.T_hand_obj.inverse()
                collision_checker.set_table(hand_obj_filename, T_obj_hand)

                for image_id, rendered_image in hand_object_pose.rendered_images.iteritems():

                    segmentation_im = ColorImage(np.array(rendered_image[IMAGE_SEGMENTATION_KEY]))
                    depth_im = DepthImage(np.array(rendered_image[IMAGE_DEPTH_KEY]))

                    fx, fy, cx, cy = rendered_image.attrs[CAM_INTRINSICS_KEY]
                    camera_intr = CameraIntrinsics('camera', fx=fx, fy=fy, cx=cx, cy=cy, skew=0.0, height=depth_im.height, width=depth_im.width)

                    candidate_grasps_dict[obj.key][hand_object_pose.id][image_id] = []

                    T_approach = RigidTransform(rotation=rendered_image.attrs[CAM_ROT_KEY].dot(RigidTransform.rotation_from_axis_angle(np.array([np.pi, 0, 0]))),
                                                        translation=rendered_image.attrs[CAM_POS_KEY],
                                                        from_frame='camera', to_frame='obj')

                    # for debug purposes
                    # xyz_mesh = os.path.join(os.path.dirname(table_mesh_filename), 'xyz_rave.obj')
                    if config['vis']['collision_checking']:
                        collision_checker.set_camera(T_approach.matrix, np.linalg.norm(T_approach.translation))
                    
                    # read grasp and metrics
                    grasps = dataset.grasps(obj.key, gripper=gripper.name)
                    logging.info('Aligning %d grasps for object %s in hand_object_pose %s' %(len(grasps), obj.key, hand_object_pose.id))

                    approach_vector = T_approach.z_axis / np.linalg.norm(T_approach.z_axis)
                    perpendicular_grasps = [g for g in grasps if np.abs(g.axis.dot(T_approach.z_axis)) < max_grasp_approach_table_angle]
                    aligned_perpendicular_grasps = [g.align_approach_vector(approach_vector) for g in perpendicular_grasps]

                    # collision_checker.clear_frames()
                    # collision_checker.draw_frame(T_approach.matrix)
                    
                    # [collision_checker.draw_frame(p.T_grasp_obj.matrix) for p in perpendicular_grasps]
                    # print("those are the perpendicular grasps")
                    # collision_checker.clear_frames()
                    
                    # collision_checker.draw_frame(T_approach.matrix)
                    # [collision_checker.draw_frame(p.T_grasp_obj.matrix) for p in aligned_perpendicular_grasps]
                    # print("those are the aligned perpendicular grasps")
                    # collision_checker.clear_frames()

                    candidate_grasps = []
                    # check grasp validity
                    logging.info('Checking collisions for %d grasps for object %s in hand_pose %s for image %s' %(len(aligned_perpendicular_grasps), obj.key, hand_object_pose.id, image_id))
                    for aligned_grasp in aligned_perpendicular_grasps:

                        # check whether any valid approach directions are collision free
                        collision_free = False
                        for phi_offset in phi_offsets:
                            rotated_grasp = aligned_grasp.grasp_y_axis_offset(phi_offset)
                            collides = collision_checker.collides_along_approach(rotated_grasp, approach_dist, delta_approach)
                            if not collides:
                                collision_free = True
                                break
                
                        # store if aligned to table
                        candidate_grasps.append(GraspInfo(aligned_grasp, collision_free))

                        # visualize if specified
                        if collision_free:
                            if config['vis']['candidate_grasps']:
                                logging.info('Grasp %d' %(aligned_grasp.id))
                                vis.figure()
                                vis.gripper_on_object(gripper, aligned_grasp, obj, hand_object_pose.T_hand_obj)
                                vis.show()
                    
                    candidate_grasps_dict[obj.key][hand_object_pose.id][image_id] = candidate_grasps

                    T_obj_camera = T_approach.inverse()
                    collision_free_projected_grasps = [gi.grasp.project_camera(T_obj_camera, camera_intr) for gi in candidate_grasps if gi.collision_free]

                    if len(collision_free_projected_grasps) == 0:
                        logging.info('no collision free grasps found')
                        continue # skip if no grasp was found

                    # project determined grasps to image space and draw quality, angle and width map
                    q_map = np.zeros(depth_im.shape)
                    a_map = np.zeros(depth_im.shape)
                    w_map = np.zeros(depth_im.shape)
                    for grasp_2d in collision_free_projected_grasps:
                        # print(grasp_2d.center.x)
                        # print(grasp_2d.center.y)
                        # print(grasp_2d.angle)
                        # print(grasp_2d.width_px)
                        
                        bb = grasp_2d.bounding_box_edges()
                        rr, cc = polygon(bb[1,:], bb[0,:], depth_im.shape)
                        q_map[rr, cc] = 1
                        a_map[rr, cc] = np.fmod(grasp_2d.angle + 2*np.pi , np.pi)
                        w_map[rr, cc] = grasp_2d.width_px
                        # print('-- angle rad: {}  deg: {}'.format(grasp_2d.angle, np.rad2deg(grasp_2d.angle)))
                        # a = np.fmod(grasp_2d.angle + 2*np.pi , np.pi)
                        # print('   wrap  rad: {}  deg: {}'.format(a, np.rad2deg(a)))
                        # print('   map  2sin: {} 2cos: {}'.format(np.sin(2*a), np.cos(2*a)))

                    a_cos_map = np.cos(2*a_map)
                    a_sin_map = np.sin(2*a_map) 
                            
                    if config['vis']['grasp_images']:
                        # plot 2D grasp image
                        vis2d.figure()
                        vis2d.subplot(3,2,1)
                        vis2d.imshow(segmentation_im)
                        [vis2d.grasp(g, color='m', show_axis=True) for g in collision_free_projected_grasps]
                        vis2d.subplot(3,2,2)
                        depth_im_clipped = DepthImage(np.clip(depth_im.data, 0, 1.5))
                        vis2d.imshow(depth_im_clipped)
                        [vis2d.grasp(g, color='m', show_axis=True) for g in collision_free_projected_grasps]

                        vis2d.subplot(3,2,3)
                        vis2d.imshow(GrayscaleImage(255*q_map.astype(np.uint8)))
                        vis2d.subplot(3,2,4)
                        vis2d.imshow(GrayscaleImage(w_map.astype(np.uint8)))
                        vis2d.subplot(3,2,5)
                        vis2d.imshow(GrayscaleImage((127*(1+a_cos_map)).astype(np.uint8)), vmin=0, vmax=255)
                        vis2d.subplot(3,2,6)
                        vis2d.imshow(GrayscaleImage((127*(1+a_sin_map)).astype(np.uint8)), vmin=0, vmax=255)
                        vis2d.title('Coll Free? %d' % collision_free)
                        vis2d.show()

                    segmentation_image = segmentation_im.raw_data
                    # _, segmentation_image = cv2.threshold(segmentation_im.raw_data, 127, 1, cv2.THRESH_BINARY)
                    segmentation_classes = np.argmax(np.dstack([0.1*np.ones(segmentation_image.shape[:2]),segmentation_image]), axis=2)
                    segmentation_map = np.eye(4)[segmentation_classes]

                        # store to data buffers
                    tensor_datapoint['depth_images'] = depth_im.raw_data
                    tensor_datapoint['segmentation_images'] = segmentation_map
                    tensor_datapoint['quality_maps'] = q_map
                    # tensor_datapoint['angle_maps'] = a_map
                    tensor_datapoint['angle_sin_maps'] = a_sin_map
                    tensor_datapoint['angle_cos_maps'] = a_cos_map
                    tensor_datapoint['width_maps'] = w_map
                    tensor_datapoint['camera_poses'] = T_approach.vec
                    tensor_datapoint['camera_intrs'] = camera_intr.vec[:4]
                    # tensor_datapoint['pose_labels'] = cur_pose_label
                    # tensor_datapoint['image_labels'] = cur_image_label

                    # for metric_name, metric_val in grasp_metrics[grasp.id].iteritems():
                    #     coll_free_metric = (1 * collision_free) * metric_val
                    #     tensor_datapoint[metric_name] = coll_free_metric
                    tensor_dataset.add(tensor_datapoint)

                    

                    tf_example = tf.train.Example(features=tf.train.Features(feature={
                        'scene/depth': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(depth_im.raw_data, -1))),
                        'scene/segmentation': tf.train.Feature(int64_list=tf.train.Int64List(value=np.reshape(segmentation_classes, -1))),
                        'scene/quality': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(q_map, -1))),
                        'scene/angle_sin': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(a_sin_map, -1))),
                        'scene/angle_cos': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(a_cos_map, -1))),
                        'scene/gripper_width': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(w_map, -1))),
                        'scene/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[im_final_width])),
                        'scene/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[im_final_height]))
                    }))

                    record_index = record_index + 1

                    tf_record_writer = tf_record_writers[ tvt_choice[ record_index % len(tvt_choice) ] ]

                    tf_record_writer.write(tf_example.SerializeToString())

    [writer.close() for writer in tf_record_writers]


    tensor_dataset.flush()
    
    logging.info('done')

    # save category mappings
    # obj_cat_filename = os.path.join(output_dir, 'object_category_map.json')
    # json.dump(obj_category_map, open(obj_cat_filename, 'w'))
    # pose_cat_filename = os.path.join(output_dir, 'pose_category_map.json')
    # json.dump(pose_category_map, open(pose_cat_filename, 'w'))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Create a GQ-CNN training dataset from a dataset of 3D object models and grasps in a Dex-Net database')
    parser.add_argument('dataset_path', type=str, default=None, help='name of folder to save the training dataset in')
    parser.add_argument('--config_filename', type=str, default=None, help='configuration file to use')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    config_filename = args.config_filename

    # handle config filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/generate_oaho_dataset.yaml')

    # turn relative paths absolute
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.getcwd(), dataset_path)
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # parse config
    config = YamlConfig(config_filename)

    # set seed
    debug = config['debug']
    if debug:
        random.seed(SEED)
        np.random.seed(SEED)
        
    # open database
    database = Hdf5Database(config['database_name'],
                            access_level=READ_ONLY_ACCESS)

    # read params
    target_object_keys = config['target_objects']
    env_rv_params = config['env_rv_params']
    gripper_name = config['gripper']

    # generate the dataset
    generate_oaho_dataset(dataset_path,
                           database,
                           target_object_keys,
                           env_rv_params,
                           gripper_name,
                           config)
