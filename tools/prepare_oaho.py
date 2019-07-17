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
from glob import glob
import numpy as np
import bpy
import bmesh
import mathutils as mu

import argparse
import os
import sys

blender_dir = os.path.dirname(bpy.data.filepath)
bpy.context.user_preferences.view.show_splash = False

# DATASET_NAME = '3dnet'
DATASET_NAME = 'kit'

# std imports
sys.path.append(os.path.join(blender_dir,  'src'))

import oaho.database_keys as oaho_db
import oaho.scene as oaho_scene
import oaho.constants as oaho_constants


def remove_object_grasps_visualization():
    if 'grasp_visualization_group' in bpy.data.groups:
        grasp_visualization_group = bpy.data.groups['grasp_visualization_group']
        bpy.ops.object.select_all(action='DESELECT')
        for grasp_visualization in grasp_visualization_group.objects:
            grasp_visualization.select = True
        bpy.ops.object.delete()
        bpy.data.groups.remove(grasp_visualization_group)


def add_object_grasps_visualization(db_object, obj, grasp_id=None):
    remove_object_grasps_visualization()
    bpy.ops.object.select_all(action='DESELECT')
    grasp_visualization_group = bpy.data.groups.new('grasp_visualization_group')
    grasp_boxes = []
    grippers = list(db_object[oaho_db.GRASPS_KEY].keys())
    grasp_ids = list(db_object[oaho_db.GRASPS_KEY][grippers[0]])
    if grasp_id is not None:
        grasp_ids = [grasp_ids[grasp_id]]
    for grasp_id in grasp_ids:
        grasp = db_object[oaho_db.GRASPS_KEY][grippers[0]][grasp_id]
        configuration = grasp.attrs[oaho_db.GRASP_CONFIGURATION_KEY]
        grasp_center = configuration[0:3]
        grasp_axis = configuration[3:6]
        max_width = configuration[6]
        angle = configuration[7]
        jaw_width = configuration[8]
        min_grasp_width = 0 if configuration.shape[0] <= 9 else configuration[9]

        # Add grid world position to cube local position.
        bpy.ops.mesh.primitive_cube_add(location=grasp_center, rotation=grasp_axis)#, radius=max_width)
        grasp_box = bpy.context.active_object
        grasp_box.name = grasp_id

        # grasp_box.scale = np.array([1, 0.1, 0.1])*max_width
        grasp_box.scale = (max_width, 0.0002, 0.0002)
        
        grasp_box = bpy.data.objects[grasp_id]
        grasp_child_of_object = grasp_box.constraints.new('CHILD_OF')
        grasp_child_of_object.target = obj#bpy.data.objects['Hum2:Body']
        #grasp_child_of_object.subtarget = RIG_SPECIFIC_PREFIX + 'hand.R'

        ## 
        m_cam = bpy.data.objects[oaho_constants.CAMERA].matrix_world
        m_obj = obj.matrix_world
        m_cam
        ##
        grasp_boxes.append(grasp_box)
    for grasp_box in grasp_boxes:
        grasp_visualization_group.objects.link(grasp_box)
    return grasp_boxes



def setup_oaho(data_dir, mhx_dir, pose_dir, object_database):
    for mhx_path in glob(os.path.join(mhx_dir, '*mhx2')):
        oaho_scene.clearScene()
        if oaho_constants.USE_MHX_RIG:
            bpy.ops.import_scene.makehuman_mhx2(
                filter_glob='*.mhx2', filepath=mhx_path, useOverride=True, rigType='MHX')
        else:
            bpy.ops.import_scene.makehuman_mhx2(
                filter_glob='*.mhx2', filepath=mhx_path, useOverride=True, rigType='BASE')
        # bpy.ops.mesh.primitive_plane_add(radius=6)
        # bpy.ops.mesh.primitive_plane_add(radius=6)

        oaho_scene.positionCamera()
        oaho_scene.createCameraTrackingConstraint()
        materials = oaho_scene.defineMaterials()
        oaho_scene.setMaterial(materials)
        # file_name = os.path.splitext(os.path.basename(mhx_path))[0]
        # out_dir = os.path.join(data_dir, file_name)

    print('setup_oaho done')


bl_info = {
    'name': 'Work Macro',
    'category': 'Object',
}

class WriteHandPoseMacro(bpy.types.Operator):
    """Write Hand Pose Macro"""
    bl_idname = 'object.write_hand_pose'
    bl_label = 'Store Current Hand Pose?'
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        print('write current hand pose')
        oaho_scene.writeHandPoseQuat(pose_dir)
        hand_pose_paths = get_hand_poses()
        current_hand_pose_index = len(hand_pose_paths)-1
        self.report({'INFO'}, 'hand pose is idx:{}, {}'.format(current_hand_pose_index, hand_pose_paths[current_hand_pose_index]))
        return {'FINISHED'}


def get_hand_poses():
    hp = list(glob(os.path.join(pose_dir, 'hand_pose*csv')))
    hp = sorted(hp, key=os.path.basename)
    return hp

def enumerate_hand_poses(scene, context):
    hand_pose_paths = get_hand_poses()
    items = [(p, p.split('/')[-1], '') for i, p in enumerate(hand_pose_paths)]
    return items

class SetNextHandPoseMacro(bpy.types.Operator):
    """set next Hand Pose Macro"""
    bl_idname = 'object.set_next_hand_pose'
    bl_label = 'set next Hand Pose Macro'
    bl_options = {'REGISTER', 'UNDO'}
    index_delta = bpy.props.IntProperty(default=1)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
    

    def invoke(self, context, event):
        self.index_delta = -1 if event.shift else 1
        return self.execute(context)

    def execute(self, context):
        hand_pose_paths = get_hand_poses()
        if len(hand_pose_paths) > 0:
            if context.scene.base_hand_pose and context.scene.base_hand_pose in hand_pose_paths:
                current_hand_pose_index = (hand_pose_paths.index(context.scene.base_hand_pose) + self.index_delta) % len(hand_pose_paths)
            else:
                current_hand_pose_index = 0
            context.scene.base_hand_pose = hand_pose_paths[current_hand_pose_index]
        else:
            return {'CANCELLED'}
        return {'FINISHED'}


current_object = None
current_hand_object_configuration_index = dict()

class SetNextGraspObjectMacro(bpy.types.Operator):
    """SetNext Grasp Object Macro"""
    bl_idname = 'object.set_next_grasp_object'
    bl_label = 'Set Next Grasp Object'
    bl_options = {'REGISTER', 'UNDO'}
    index_delta = bpy.props.IntProperty(default=1)

    def invoke(self, context, event):
        self.index_delta = -1 if event.shift else 1
        return self.execute(context)

    def execute(self, context):
        objects = get_database_object_names()
        if context.scene.current_object_name in objects:
            idx = (objects.index(context.scene.current_object_name) + self.index_delta) % len(objects)
        else:
            idx = 0

        context.scene.current_object_name = objects[idx]
        self.report({'INFO'}, 'current object is:{}, {}'.format(idx, objects[idx]))
        return {'FINISHED'}

class StoreHandObjectConfigurationMacro(bpy.types.Operator):
    """StoreHandObjectConfiguration Macro"""
    bl_idname = 'object.store_hand_object_configuration'
    bl_label = 'Store Hand Object Configuration'
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        objects = get_database_object_names()
        if context.scene.current_object_name not in objects:
            return {'CANCELLED'}
        if not current_object:
            print('no current_object defined')
            return {'CANCELLED'}

        print(current_object.location)
        current_object.rotation_mode = 'QUATERNION'
        print(current_object.rotation_quaternion)
        print(oaho_scene.getHandPoseQuat())

        objects_dataset = object_database[oaho_db.DATASETS_KEY][DATASET_NAME][oaho_db.OBJECTS_KEY]
        db_object = objects_dataset[context.scene.current_object_name]
        if oaho_db.HAND_OBJECT_POSES_KEY not in db_object.keys():
            db_object.create_group(oaho_db.HAND_OBJECT_POSES_KEY)
        hand_object_poses = db_object[oaho_db.HAND_OBJECT_POSES_KEY]
        num_poses = len(hand_object_poses.keys())
        pose_key = 'pose_{}'.format(num_poses)
        hand_object_poses.create_group(pose_key)
        hand_object_poses[pose_key].attrs.create(oaho_db.HAND_OBJECT_POSE_PT_KEY, current_object.location)

        rot_matrix = mu.Quaternion(current_object.rotation_quaternion).to_matrix()
        hand_object_poses[pose_key].attrs.create(oaho_db.HAND_OBJECT_POSE_ROT_KEY, rot_matrix)
        hand_object_poses[pose_key].attrs.create(oaho_db.HAND_OBJECT_POSE_QUATERNION_KEY, current_object.rotation_quaternion) # actually also store quaternion
        hand_object_poses[pose_key].attrs.create(oaho_db.HAND_OBJECT_POSE_HAND_POSE_KEY, oaho_scene.getHandPoseQuat())
        # hand_object_poses[pose_key].attrs.create(oaho_db.HAND_OBJECT_POSE_ROT_KEY, rot_matrix)

        self.report({'INFO'}, 'stored hand_object pose {}, x0:{}, q:{}, hand_pose:{}'.format(pose_key, 
            hand_object_poses[pose_key].attrs[oaho_db.HAND_OBJECT_POSE_PT_KEY],
            hand_object_poses[pose_key].attrs[oaho_db.HAND_OBJECT_POSE_QUATERNION_KEY],
            hand_object_poses[pose_key].attrs[oaho_db.HAND_OBJECT_POSE_HAND_POSE_KEY]))
        return {'FINISHED'}


class DeleteHandObjectConfigurationMacro(bpy.types.Operator):
    """DeleteHandObjectConfigurationMacro Macro"""
    bl_idname = 'object.delete_current_hand_object_configuration'
    bl_label = 'Delete Current Hand Object Configuration'
    bl_options = {'REGISTER', 'UNDO'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        objects = get_database_object_names()
        if context.scene.current_object_name not in objects:
            return {'CANCELLED'}

        objects_dataset = object_database[oaho_db.DATASETS_KEY][DATASET_NAME][oaho_db.OBJECTS_KEY]

        db_object = objects_dataset[context.scene.current_object_name]
        hand_object_configuration_index = -1
        if context.scene.current_object_name in current_hand_object_configuration_index:
            hand_object_configuration_index = current_hand_object_configuration_index[context.scene.current_object_name]
        else:
            return {'CANCELLED'}

        if oaho_db.HAND_OBJECT_POSES_KEY in db_object.keys():
            hand_object_poses = db_object[oaho_db.HAND_OBJECT_POSES_KEY]
            poses = list(hand_object_poses.keys())
            if len(poses) > 0 and hand_object_configuration_index < len(poses):
                try:
                    del hand_object_poses[poses[hand_object_configuration_index]]
                    self.report({'INFO'}, 'deleted hand_object pose idx:{}, {}'.format(hand_object_configuration_index, poses[hand_object_configuration_index]))
                    del poses[hand_object_configuration_index]
                    if len(poses) > 0:
                        current_hand_object_configuration_index[context.scene.current_object_name] = (hand_object_configuration_index - 1) % len(poses)
                    else:
                        current_hand_object_configuration_index[context.scene.current_object_name] = -1
                except Exception as e:
                    return {'CANCELLED'}
            else:
                return {'CANCELLED'}
            # rename poses
            for i, pose in enumerate(list(hand_object_poses.keys())):
                desired_pose_name = 'pose_{}'.format(i)
                if pose != desired_pose_name:
                    hand_object_poses.move(pose, desired_pose_name)
                    # hand_object_poses[desired_pose_name]


        return {'FINISHED'}

class SetNextHandObjectConfigurationMacro(bpy.types.Operator):
    """SetNextHandObjectConfiguration Macro"""
    bl_idname = 'object.set_next_hand_object_configuration'
    bl_label = 'Set Next Hand Object Configuration'
    bl_options = {'REGISTER', 'UNDO'}
    index_delta = bpy.props.IntProperty(default=1)


    def invoke(self, context, event):
        print(event)
        self.index_delta = -1 if event.shift else 1
        return self.execute(context)

    def execute(self, context):
        objects = get_database_object_names()
        if current_object is None or context.scene.current_object_name not in objects:
            return {'CANCELLED'}

        objects_dataset = object_database[oaho_db.DATASETS_KEY][DATASET_NAME][oaho_db.OBJECTS_KEY]
        db_object = objects_dataset[context.scene.current_object_name]
        hand_object_configuration_index = -1
        if context.scene.current_object_name in current_hand_object_configuration_index:
            hand_object_configuration_index = current_hand_object_configuration_index[context.scene.current_object_name]

        if oaho_db.HAND_OBJECT_POSES_KEY in db_object.keys():
            poses = list(db_object[oaho_db.HAND_OBJECT_POSES_KEY].keys())
            if len(poses) > 0:
                hand_object_configuration_index = (hand_object_configuration_index + self.index_delta) % len(poses)
                current_hand_object_configuration_index[context.scene.current_object_name] = hand_object_configuration_index
                pose = db_object[oaho_db.HAND_OBJECT_POSES_KEY][poses[hand_object_configuration_index]]

                if oaho_db.HAND_OBJECT_POSE_QUATERNION_KEY in pose.attrs.keys():
                    q = pose.attrs[oaho_db.HAND_OBJECT_POSE_QUATERNION_KEY]
                elif oaho_db.HAND_OBJECT_POSE_ROT_KEY in pose.attrs.keys():
                    q = mu.Matrix(pose.attrs[oaho_db.HAND_OBJECT_POSE_ROT_KEY]).to_quaternion()
                    pose.attrs.create(oaho_db.HAND_OBJECT_POSE_QUATERNION_KEY, q)
                else:
                    return {'CANCELLED'}
                
                current_object.location = pose.attrs[oaho_db.HAND_OBJECT_POSE_PT_KEY]
                current_object.rotation_mode = 'QUATERNION'
                current_object.rotation_quaternion = q
                oaho_scene.setHandPoseQuat(pose.attrs[oaho_db.HAND_OBJECT_POSE_HAND_POSE_KEY])
                self.report({'INFO'}, 'hand_object pose is idx:{}, {}'.format(hand_object_configuration_index, poses[hand_object_configuration_index]))
            else:
                return {'CANCELLED'}

        return {'FINISHED'}


def set_base_hand_pose(self, context):
    hand_poses = oaho_scene.loadPoses([self.base_hand_pose])
    if len(hand_poses) > 0:
        oaho_scene.setHandPoseQuat(hand_poses[0])

def get_database_object_names():
    return list(object_database[oaho_db.DATASETS_KEY][DATASET_NAME][oaho_db.OBJECTS_KEY].keys())

def enumerate_database_object_names(scene, context):
    return [(o, o, '') for o in get_database_object_names()]

def set_current_object(self, context):    
    global current_object
    db_object = object_database[oaho_db.DATASETS_KEY][DATASET_NAME][oaho_db.OBJECTS_KEY][self.current_object_name]

    if current_object is not None:
        oaho_scene.remove_object(current_object)
        current_object = None
    remove_object_grasps_visualization()
    current_object = oaho_scene.add_object(db_object)
    oaho_scene.set_object_hand_constraint(current_object)
    if self.visualize_grasps:
        add_object_grasps_visualization(db_object, current_object)

def set_visualize_grasps(self, context):
    if self.visualize_grasps and self.current_object_name and current_object is not None:
        db_object = object_database[oaho_db.DATASETS_KEY][DATASET_NAME][oaho_db.OBJECTS_KEY][self.current_object_name]
        add_object_grasps_visualization(db_object, current_object)
    else:
        remove_object_grasps_visualization()



bpy.types.Scene.base_hand_pose = bpy.props.EnumProperty(name='Base Hand Pose', items=enumerate_hand_poses, update=set_base_hand_pose)
bpy.types.Scene.current_object_name = bpy.props.EnumProperty(name='DB Object', items=enumerate_database_object_names, update=set_current_object)
bpy.types.Scene.visualize_grasps = bpy.props.BoolProperty(name='Visualize Grasps', default=False, update=set_visualize_grasps)



class BaseHandPosePanel(bpy.types.Panel):
    """Panel to select a base hand pose"""
    bl_label = 'BaseHandPose Panel'
    bl_idname = 'OBJECT_PT_BaseHandPosePanel'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'object'

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        col = layout.column()
        col.prop(scene, 'base_hand_pose')
        col.operator('object.set_next_hand_pose', text='Next Hand Pose', icon='HAND')
        col.operator('object.write_hand_pose', text='Store Hand Pose', icon='HAND')
        
        col.prop(scene, 'current_object_name')
        row = col.row()
        # row.operator('object.set_next_grasp_object', text='Previous Object', icon='OBJECT_DATA').index_delta = -1
        row.operator('object.set_next_grasp_object', text='Next Object', icon='OBJECT_DATA')
        col.prop(scene, 'visualize_grasps')

        # bl_idname = 'object.store_hand_object_configuration'
        # bl_idname = 'object.delete_current_hand_object_configuration'
        # bl_idname = 'object.set_next_hand_object_configuration'



# store keymaps here to access after registration
addon_keymaps = []


def register():
    wm = bpy.context.window_manager

    bpy.utils.register_class(WriteHandPoseMacro)
    bpy.utils.register_class(SetNextHandPoseMacro)
    bpy.utils.register_class(SetNextGraspObjectMacro)
    bpy.utils.register_class(StoreHandObjectConfigurationMacro)
    bpy.utils.register_class(SetNextHandObjectConfigurationMacro)
    bpy.utils.register_class(DeleteHandObjectConfigurationMacro)

    bpy.utils.register_class(BaseHandPosePanel)
    
    
    # handle the keymap
    km = wm.keyconfigs.addon.keymaps.new(name='Object Mode', space_type='EMPTY')
    kmi = km.keymap_items.new(WriteHandPoseMacro.bl_idname, 'NUMPAD_0', 'PRESS', ctrl=True, shift=False)
    kmi = km.keymap_items.new(SetNextHandPoseMacro.bl_idname, 'NUMPAD_0', 'PRESS', shift=False)
    kmi = km.keymap_items.new(SetNextHandPoseMacro.bl_idname, 'NUMPAD_0', 'PRESS', shift=True)

    kmi = km.keymap_items.new(SetNextGraspObjectMacro.bl_idname, 'NUMPAD_5', 'PRESS', shift=False)
    kmi = km.keymap_items.new(SetNextGraspObjectMacro.bl_idname, 'NUMPAD_5', 'PRESS', shift=True)

    kmi = km.keymap_items.new(StoreHandObjectConfigurationMacro.bl_idname, 'NUMPAD_8', 'PRESS', shift=False, ctrl=True)
    kmi = km.keymap_items.new(DeleteHandObjectConfigurationMacro.bl_idname, 'NUMPAD_8', 'PRESS', shift=True, ctrl=True)
    kmi = km.keymap_items.new(SetNextHandObjectConfigurationMacro.bl_idname, 'NUMPAD_8', 'PRESS', shift=False)
    kmi = km.keymap_items.new(SetNextHandObjectConfigurationMacro.bl_idname, 'NUMPAD_8', 'PRESS', shift=True)
    


    addon_keymaps.append(km)

def unregister():
    bpy.utils.unregister_class(WriteHandPoseMacro)
    bpy.utils.unregister_class(SetNextHandPoseMacro)
    bpy.utils.unregister_class(SetNextGraspObjectMacro)
    bpy.utils.unregister_class(StoreHandObjectConfigurationMacro)
    bpy.utils.unregister_class(SetNextHandObjectConfigurationMacro)
    bpy.utils.unregister_class(DeleteHandObjectConfigurationMacro)

    # handle the keymap
    wm = bpy.context.window_manager
    for km in addon_keymaps:
        wm.keyconfigs.addon.keymaps.remove(km)
    # clear the list
    del addon_keymaps[:]


if __name__ == '__main__':

    dataset_path = os.path.join(blender_dir, 'data/dexnet_2_database.hdf5')

    data_dir = os.path.join(blender_dir, 'data')
    pose_dir = os.path.join(blender_dir, 'data/poses')
    mhx_dir = os.path.join(blender_dir, 'data/make_human')

    object_database = h5py.File(dataset_path, 'r+')

    setup_oaho('out', mhx_dir, pose_dir, object_database)
    # unregister()
    register()



