# Copyright (C) 2015 Institute for Computer Graphics and Vision (ICG),
#   Graz University of Technology (TU GRAZ)

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. All advertising materials mentioning features or use of this software
#    must display the following acknowledgement:
#    This product includes software developed by the ICG, TU GRAZ.
# 4. Neither the name of the ICG, TU GRAZ nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY ICG, TU GRAZ ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL ICG, TU GRAZ BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import h5py
import time
import random
from glob import glob
import datetime
import math
import numpy as np
import bpy
import bmesh
import bpy_extras.object_utils
import mathutils as mu

import argparse
import os
import sys
# import pyexr
import cv2
# import imp

blender_dir = os.path.dirname(bpy.data.filepath)
bpy.context.user_preferences.view.show_splash = False

# std imports

#DATASET_NAME = '3dnet'
DATASET_NAME = 'kit'

sys.path.append(os.path.join(blender_dir,  'src'))

import oaho.database_keys as oaho_db
import oaho.scene as oaho_scene
import oaho.constants as oaho_constants


def generate_oaho_(mesh_name, data_dir, pose_dir, object_database, distractor_objects):
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    scale_paths = glob(os.path.join(pose_dir, 'scale*csv'))
    arm_pose_paths = glob(os.path.join(pose_dir, 'arm_pose*csv'))
    hand_pose_paths = glob(os.path.join(pose_dir, 'hand_pose*csv'))

    print('%d scale_paths' % len(scale_paths))
    print('%d arm_pose_paths' % len(arm_pose_paths))
    print('%d hand_pose_paths' % len(hand_pose_paths))

    scales = oaho_scene.loadPoses(scale_paths)
    arm_poses = oaho_scene.loadPoses(arm_pose_paths)
    hand_poses = oaho_scene.loadPoses(hand_pose_paths)
    # arm_poses.extend(arm_poses)
    # arm_poses = interpolatePoses(arm_poses, arm_interpolation_steps, rand1IdxPairs)
    # hand_poses = interpolatePoses(hand_poses, hand_interpolation_steps, rand2IdxPairs)

    # write camera parameters
    calibration_camera_path = os.path.join(data_dir, 'calibration_depth.txt')
    oaho_scene.writeIntrinsics(calibration_camera_path)
    camera = bpy.data.objects[oaho_constants.CAMERA]

    start_time = time.time()

    datasets = object_database[oaho_db.DATASETS_KEY]
    objects_dataset = object_database[oaho_db.DATASETS_KEY][DATASET_NAME][oaho_db.OBJECTS_KEY]

    objects = list(object_database[oaho_db.DATASETS_KEY][DATASET_NAME][oaho_db.OBJECTS_KEY].keys())
    n_jitter = 0
    img_idx = 0
    n_images = scales.shape[0] * arm_poses.shape[0] * hand_poses.shape[0] * (1 + n_jitter)

    current_object = None
    for object_id in objects:
        print ('loading object: {}'.format(object_id))
        db_object = objects_dataset[object_id]
        if oaho_db.HAND_OBJECT_POSES_KEY in db_object.keys():
            if current_object is not None:
                oaho_scene.remove_object(current_object)
                current_object = None

            current_object = oaho_scene.add_object(db_object)
            oaho_scene.set_object_hand_constraint(current_object)
            
            hand_object_poses = db_object[oaho_db.HAND_OBJECT_POSES_KEY]
            pose_names = list(hand_object_poses.keys())
            for pose_name in pose_names:
                pose = hand_object_poses[pose_name]
                if oaho_db.HAND_OBJECT_POSE_QUATERNION_KEY in pose.attrs.keys():
                    q = pose.attrs[oaho_db.HAND_OBJECT_POSE_QUATERNION_KEY]
                elif oaho_db.HAND_OBJECT_POSE_ROT_KEY in pose.attrs.keys():
                    q = mu.Matrix(pose.attrs[oaho_db.HAND_OBJECT_POSE_ROT_KEY]).to_quaternion()
                    pose.attrs.create(oaho_db.HAND_OBJECT_POSE_QUATERNION_KEY, q)
                else:
                    continue                        
                
                current_object.location = pose.attrs[oaho_db.HAND_OBJECT_POSE_PT_KEY]
                current_object.rotation_mode = 'QUATERNION'
                current_object.rotation_quaternion = q
                oaho_scene.setHandPoseQuat(pose.attrs[oaho_db.HAND_OBJECT_POSE_HAND_POSE_KEY])
                
                oaho_scene.randomizeScene(distractor_objects)

                # prepare hdf5 dataset structure
                _, hand_vertices, hand_triangles = oaho_scene.get_hand_pose(mesh_name)
                if oaho_db.HAND_OBJECT_POSE_HAND_MESH_KEY not in pose.keys():
                    pose.create_group(oaho_db.HAND_OBJECT_POSE_HAND_MESH_KEY)
                
                pose_hand_mesh = pose[oaho_db.HAND_OBJECT_POSE_HAND_MESH_KEY]
                pose_hand_mesh.clear()

                pose_hand_mesh.create_dataset(oaho_db.MESH_VERTICES_KEY, data=hand_vertices)
                pose_hand_mesh.create_dataset(oaho_db.MESH_TRIANGLES_KEY, data=hand_triangles)

                # continue
                if oaho_db.RENDERED_IMAGES_KEY not in pose.keys():
                    pose.create_group(oaho_db.RENDERED_IMAGES_KEY)
                else:
                    pose[oaho_db.RENDERED_IMAGES_KEY].clear() # TODO maybe don't clear images let's see

                rendered_images = pose[oaho_db.RENDERED_IMAGES_KEY]

                for arm_pose_iter in range(2):
                    for arm_pose in arm_poses:
                        oaho_scene.setArmPoseQuat(arm_pose)
                        oaho_scene.positionCamera()
                        image_id = '{}_{}'.format(oaho_db.IMAGE_KEY, len(list(rendered_images.keys())))

                        out_dir = os.path.join(data_dir, 'clean')
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)

                        # define output paths
                        ts = oaho_scene.getTs()
                        depth_path = os.path.join(
                            out_dir, '%08d_%s_depth_' % (img_idx, ts))
                        rgb_path = os.path.join(
                            out_dir, '%08d_%s_rgb_' % (img_idx, ts))
                        anno3d_path = os.path.join(
                            out_dir, '%08d_%s_anno_blender.txt' % (img_idx, ts))

                        # create imgs
                        oaho_scene.render(depth_path, rgb_path)

                        # compute anno
                        anno3d, hand_vertices, hand_triangles = oaho_scene.get_hand_pose(mesh_name)
                        anno3d[2, :] *= 1000  # to mm
                        # save anno
                        anno3d = anno3d.T.reshape(1, np.prod(anno3d.shape))
                        np.savetxt(anno3d_path, anno3d, delimiter=' ', fmt='%.4f')
                        
                        depth_path += '0001.exr'
                        rgb_path += '0001.png'

                        # depth_image = pyexr.read(depth_path)
                        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

                        segmentation_image = cv2.imread(rgb_path)
                        segmentation_image = cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2RGB)

                        transform_cam_to_obj =  current_object.matrix_world.inverted() * camera.matrix_world
                        pos_cam_to_obj, rot_quat_cam_to_obj, _ = transform_cam_to_obj.decompose()
                        
                        rendered_images.create_group(image_id)
                        image_data = rendered_images[image_id]

                        image_data.create_dataset(oaho_db.IMAGE_DEPTH_KEY, data=depth_image[:,:,0])
                        image_data.create_dataset(oaho_db.IMAGE_SEGMENTATION_KEY, data=segmentation_image)
                        image_data.attrs.create(oaho_db.IMAGE_TIMESTAMP_KEY, np.string_(ts))
                        image_data.attrs.create(oaho_db.IMAGE_FRAME_KEY, np.string_('image'))
                        image_data.attrs.create(oaho_db.CAM_ROT_KEY, rot_quat_cam_to_obj.to_matrix())
                        image_data.attrs.create(oaho_db.CAM_POS_KEY, pos_cam_to_obj)
                        image_data.attrs.create(oaho_db.CAM_FRAME_KEY, np.string_('camera'))
                        image_data.attrs.create(oaho_db.CAM_INTRINSICS_KEY, np.array(oaho_scene.getIntrinsics()))
                        

                        # image_data.attrs.create(HAND_POSE_KEY, anno3d)


                        # print(hand_vertices)
                        # print(hand_triangles)
                        # t = ''
                        # for hv in hand_vertices:
                        #     t += 'v {:4f} {:4f} {:4f}\n'.format(hv[0], hv[1], hv[2])
                        # for ht in hand_triangles:
                        #     t += 'f {} {} {}\n'.format(ht[0]+1, ht[1]+1, ht[2]+1)                    
                        # with open(os.path.join(out_dir, '%08d_%s_hand.obj' % (img_idx, ts)), 'w+') as f:
                        #     f.write(t)
                        # return

                        # save anno constraint for first img
                        if img_idx == 0:
                            np.savetxt(os.path.join(
                                out_dir, 'anno_constraint.txt'), anno3d, delimiter=' ', fmt='%.4f')

                        img_idx += 1

                        # fps / remaining time
                        fps = float(img_idx) / (time.time() - start_time)

                        n_images_left = n_images - img_idx
                        remaining_time_min = n_images_left / fps / 60
                        remaining_time_h = math.floor(remaining_time_min / 60)
                        remaining_time_min = remaining_time_min - remaining_time_h * 60
                        print('fps = %f | remaining: %d:%f' %
                            (fps, remaining_time_h, remaining_time_min))
        # cleanup current object
        if current_object is not None:
            oaho_scene.remove_object(current_object)
            current_object = None

def generate_oaho(data_dir, mhx_dir, pose_dir, object_database):
    mesh_name = 'm_25:Body'

    oaho_scene.configRendering()

    # print(mhx_dir)
    for mhx_path in glob(os.path.join(mhx_dir, '*mhx2')):
        # print(mhx_path)

        oaho_scene.clearScene()
        # import scene
        # bpy.ops.import_scene.makehuman_mhx(filepath=mhx_path)
        if oaho_constants.USE_MHX_RIG:
            bpy.ops.import_scene.makehuman_mhx2(
                filter_glob='*.mhx2', filepath=mhx_path, useOverride=True, rigType='MHX')
        else:
            bpy.ops.import_scene.makehuman_mhx2(
                filter_glob='*.mhx2', filepath=mhx_path, useOverride=True, rigType='BASE')

        oaho_scene.positionCamera()
        oaho_scene.createCameraTrackingConstraint()
        materials = oaho_scene.defineMaterials()

        oaho_scene.createScene(materials)

        bpy.ops.mesh.primitive_plane_add(radius=5, location=(0,0,0))
        room_floor = bpy.context.active_object
        room_floor.name = 'room_floor'
        room_floor.data.materials.clear()
        room_floor.data.materials.append(materials['background'])

        oaho_scene.setMaterial(materials)
        
        file_name = os.path.splitext(os.path.basename(mhx_path))[0]
        mesh_name = file_name + ':Body'
        print(mesh_name)
        out_dir = os.path.join(data_dir, file_name)
        
        distractor_dir = '/home/robotics/work/datasets/random_urdfs'
        distractor_objects = glob(os.path.join(distractor_dir, '**/*.obj'), recursive=True)
        distractor_objects = [o for o in distractor_objects if '_coll' not in o]
        oaho_scene.randomizeScene(distractor_objects)

        # , arm_interpolation_steps, hand_interpolation_steps, n_jitter)
        generate_oaho_(mesh_name, out_dir, pose_dir, object_database, distractor_objects)
        #generate_(mesh_name, out_dir, pose_dir, arm_interpolation_steps, hand_interpolation_steps, n_jitter)

    # out_dir = os.path.join(data_dir, 'generic')
    # generate_(mesh_name, out_dir, pose_dir, arm_interpolation_steps, hand_interpolation_steps, n_jitter)

    print('finished')


if __name__ == '__main__':
    # logging.getLogger().setLevel(logging.INFO)

    # parser = argparse.ArgumentParser(description='Create a GQ-CNN training dataset from a dataset of 3D object models and grasps in a Dex-Net database')
    # parser.add_argument('dataset_path', type=str, default=None, help='name of folder to save the training dataset in')
    # parser.add_argument('--config_filename', type=str, default=None, help='configuration file to use')
    # args = parser.parse_args()
    # print(args)
    # dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data/dexnet_2_database.hdf5')
    # config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'cfg/tools/generate_gqcnn_dataset.yaml')
    dataset_path = os.path.join(blender_dir, 'data/dexnet_2_database.hdf5')

    data_dir = os.path.join(blender_dir, 'data')
    pose_dir = os.path.join(blender_dir, 'data/poses')
    mhx_dir = os.path.join(blender_dir, 'data/make_human')

    object_database = h5py.File(dataset_path, 'r+')

    generate_oaho('data/oaho/', mhx_dir, pose_dir, object_database)

