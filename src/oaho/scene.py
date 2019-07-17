import os

import bpy
import bmesh
import bpy_extras.object_utils
import mathutils as mu
import datetime
import numpy as np
import math
import random

from oaho.constants import *
from oaho.database_keys import *

def getTs():
    now = datetime.datetime.now()
    ts = '%04d%02d%02d%02d%02d%02d' % (now.year, now.month, now.day,
                                       now.hour, now.minute, now.second)
    return ts


# ------------------------------------------------------------------------------
def printSelectedVertexInfo(mesh_name):
    object_reference = bpy.data.objects[mesh_name]

    bm = bmesh.from_edit_mesh(object_reference.data)
    for i, vert in enumerate(bm.verts):
        if vert.select:
            print('[VERT] array idx: %d; vert idx: %d ' % (i, vert.index))

    for i, face in enumerate(bm.faces):
        if face.select:
            print('[FACE] array idx: %d; face idx: %d ' % (i, face.index))
            print('[FACE] normal', face.normal)
            print(object_reference.data.polygons[i].normal)


def getPoseQuat(bone_names):
    pose = bpy.data.objects['Hum2'].pose
    state = np.zeros((4, len(bone_names)))
    for idx, bone_name in enumerate(bone_names):
        bone = pose.bones[bone_name]
        bone.rotation_mode = 'QUATERNION'
        for row in range(4):
            state[row, idx] = bone.rotation_quaternion[row]

    return state


def getArmPoseQuat():
    return getPoseQuat(ARM_BONE_NAMES)


def getHandPoseQuat():
    return getPoseQuat(HAND_BONE_NAMES)


def setPoseQuat(bone_names, state):
    pose = bpy.data.objects['Hum2'].pose
    for idx, bone_name in enumerate(bone_names):
        bone = pose.bones[bone_name]
        bone.rotation_mode = 'QUATERNION'
        for row in range(4):
            bone.rotation_quaternion[row] = state[row, idx]


def setArmPoseQuat(state):
    setPoseQuat(ARM_BONE_NAMES, state)


def setHandPoseQuat(state):
    setPoseQuat(HAND_BONE_NAMES, state)


def clearArmPoseQuat():
    state = np.zeros((4, len(ARM_BONE_NAMES)))
    state[0, :] = 1
    setPoseQuat(ARM_BONE_NAMES, state)


def clearHandPoseQuat():
    state = np.zeros((4, len(HAND_BONE_NAMES)))
    state[0, :] = 1
    setPoseQuat(HAND_BONE_NAMES, state)


def clearAllPoseQuat():
    clearArmPoseQuat()
    clearHandPoseQuat()


def clearScene():
    # bpy.context.area.type = 'VIEW_3D'
    # Clear data from previous scenes
    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)

    for texture in bpy.data.textures:
        texture.user_clear()
        bpy.data.textures.remove(texture)
    # Remove objects from previous scenes
    # objs = [ob for ob in bpy.context.scene.objects if ob.type in ('MESH')]
    # bpy.ops.object.delete({'selected_objects': objs})
    # for ob in bpy.data.objects:
    #   if ob.type == 'MESH':  #ob.type == 'ARMATURE' or
    #     bpy.data.objects[ob.name].select = True
    #     #ob.mode_set(mode='OBJECT')
    #     #scn.objects.unlink(ob)
    #     #ob.user_clear()
    #     #bpy.data.objects.remove(ob)
    #     bpy.ops.object.delete()

    # Remove objects from previsous scenes
    for item in bpy.data.objects:
        if item.type == 'MESH':
            bpy.data.objects[item.name].select = True
            bpy.ops.object.delete()

    for item in bpy.data.meshes:
        item.user_clear()
        bpy.data.meshes.remove(item)


def defineMaterials():
    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)

    skin_material = bpy.data.materials.new(name='Skin')
    skin_material.diffuse_color = (0, 1, 0)
    skin_material.diffuse_intensity = 1
    skin_material.use_shadeless = True

    hand_material = bpy.data.materials.new(name='Hand')
    hand_material.diffuse_color = (1, 0, 0)
    hand_material.diffuse_intensity = 1
    hand_material.use_shadeless = True

    object_material = bpy.data.materials.new(name='Object')
    object_material.diffuse_color = (0, 0, 1)
    object_material.diffuse_intensity = 1
    object_material.use_shadeless = True

    background_material = bpy.data.materials.new(name='Background')
    background_material.diffuse_color = (0, 0, 0)
    background_material.diffuse_intensity = 1
    background_material.use_shadeless = True
    return { 'skin': skin_material, 'hand': hand_material, 'object': object_material, 'background': background_material }


def positionCamera():
    camera = bpy.data.objects[CAMERA]
    # camera.location = (0.0, -2.0, 1.2) + 0.1*(2*np.random.random(3)-1.0)
    camera.location.x = np.random.uniform(-0.1, 0.1)
    camera.location.y = np.random.uniform(-0.8, -0.6)#np.random.uniform(-2.1, -1.9)
    camera.location.z = np.random.uniform(1.1, 1.3)
    camera.rotation_mode = 'XYZ'
    # camera.rotation_euler = np.array([1.57, 0, 0]) + 1.8*(2*np.random.random(3)-1.0)
    camera.rotation_euler.x = np.random.uniform(0.0, np.pi)
    camera.rotation_euler.y = np.random.uniform(0.0, np.pi)
    camera.rotation_euler.z = np.random.uniform(0.0, np.pi)
    # camera.keyframe_insert(data_path='location', frame=10.0)
    # camera.location.x = 10.0
    # camera.location.y = 0.0
    # camera.location.z = 5.0
    # camera.keyframe_insert(data_path='location', frame=20.0)
    


def createCameraTrackingConstraint():
    camera = bpy.data.objects[CAMERA]
    # for constraint in camera.constraints:
    #   camera.constraints.remove(constraint)
    CAMERA_CONSTRAINT = 'DAMPED_TRACK'
    # CAMERA_CONSTRAINT = 'TRACK_TO'
    camera_hand_tracking = camera.constraints.new(CAMERA_CONSTRAINT)
    camera_hand_tracking.target = bpy.data.objects['Hum2'] # camera_hand_tracking.target = bpy.data.objects['Hum2:Body']
    camera_hand_tracking.subtarget = RIG_SPECIFIC_PREFIX + 'hand.R'
    camera_hand_tracking.track_axis = 'TRACK_NEGATIVE_Z'
    if CAMERA_CONSTRAINT == 'TRACK_TO':
        camera_hand_tracking.use_target_z = False  # for TRACK_TO
        camera_hand_tracking.up_axis = 'UP_Y'  # for TRACK_TO
    camera_hand_tracking.influence = 0.925


def setupRenderingNodes(base_output_dir=''):
    scene = bpy.data.scenes[SCENE]
    scene.use_nodes = True
    tree = scene.node_tree

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    nodes = tree.nodes
    links = tree.links

    node_render_layer = nodes.new(type='CompositorNodeRLayers')
    node_render_layer.name = 'render_layers'
    node_render_layer.location = (100, 0)

    #K: [554.3826904296875, 0.0, 320.0, 0.0, 554.3826904296875, 240.0, 0.0, 0.0, 1.0]
    cam = bpy.data.cameras[CAMERA]


    # 'CompositorNodeMapRange'

    # node_segmentation_output_file = nodes.new(type='CompositorNodeOutputFile')
    # node_segmentation_output_file.name = 'segmentation_output_file'
    # #node_segmentation_output_file.format.file_format = 'PNG'
    # node_segmentation_output_file.location = (300,-150)

    # node_depth_output_file = nodes.new(type='CompositorNodeOutputFile')
    # node_depth_output_file.name = 'depth_output_file'
    # node_depth_output_file.format.file_format = 'OPEN_EXR'
    # node_depth_output_file.location = (300,150)

    # links.new(node_render_layer.outputs['Z'], node_depth_output_file.inputs[0])
    # links.new(node_render_layer.outputs['Image'], node_segmentation_output_file.inputs[0])

    node_output_file = nodes.new(type='CompositorNodeOutputFile')
    node_output_file.name = 'image_output'
    node_output_file.base_path = base_output_dir
    node_output_file.location = (300, 0)
    node_output_file.file_slots.clear()

    node_output_file.file_slots.new(name='depth')
    depth_out = node_output_file.file_slots['depth']
    # depth_out.type = 'VALUE'
    depth_out.path = 'depth'
    depth_out.use_node_format = False
    depth_out.format.file_format = 'OPEN_EXR'

    node_output_file.file_slots.new(name='segmentation')
    seg_out = node_output_file.file_slots['segmentation']
    #seg_out.type = 'RGBA'
    seg_out.path = 'segmentation'
    # seg_out.use_node_format = False
    # seg_out.format.file_format = 'PNG'

    links.new(node_render_layer.outputs['Image'],
              node_output_file.inputs['segmentation'])
    links.new(node_render_layer.outputs['Z'], node_output_file.inputs['depth'])


def writeArmPoseQuat(pose_dir):
    if not os.path.exists(pose_dir):
        os.makedirs(pose_dir)

    pose = getArmPoseQuat()

    ts = getTs()
    out_path = os.path.join(pose_dir, 'arm_pose_%s.csv' % ts)

    np.savetxt(out_path, pose, delimiter=',')


def writeHandPoseQuat(pose_dir):
    if not os.path.exists(pose_dir):
        os.makedirs(pose_dir)

    pose = getHandPoseQuat()

    ts = getTs()
    out_path = os.path.join(pose_dir, 'hand_pose_%s.csv' % ts)

    np.savetxt(out_path, pose, delimiter=',')


def setMaterial(materials):
    o = None
    for ob in bpy.data.objects:
        if ':Body' in ob.name:
            o = ob
            print (ob)
            break

    o.data.materials.clear()
    o.data.materials.append(materials['skin'])
    o.data.materials.append(materials['hand'])
    skin_material_slot_index = o.material_slots.find('Skin')
    hand_material_slot_index = o.material_slots.find('Hand')

    # skin_material_slot_index = o.material_slots.find('Skin')
    # if skin_material_slot_index < 0:
    #   o.data.materials.append(materials['skin'])
    # skin_material_slot_index = o.material_slots.find('Skin')

    # hand_material_slot_index = o.material_slots.find('Hand')
    # if hand_material_slot_index < 0:
    #   o.data.materials.append(materials['hand'])
    # hand_material_slot_index = o.material_slots.find('Hand')
    # print ('hand_material_slot_index: {}'.format(hand_material_slot_index))
    # print ('skin_material_slot_index: {}'.format(skin_material_slot_index))

    hand_vertex_group_names = [RIG_SPECIFIC_PREFIX + h for h in HAND_BONE_NAMES]
    group_indices = set([idx for idx, vg in enumerate(o.vertex_groups) if vg.name in hand_vertex_group_names])
    vertex_indices = set([idx for idx, v in enumerate(o.data.vertices) if group_indices & set([vg.group for vg in v.groups])])
    for p in o.data.polygons:
        p.material_index = skin_material_slot_index
        if vertex_indices & set(p.vertices):
            p.material_index = hand_material_slot_index


def setScale(scale):
    names = list(HAND_BONE_NAMES)
    names.extend(ARM_BONE_NAMES)

    pose = bpy.data.objects['Hum2'].pose
    for col, name in enumerate(names):
        for row in range(3):
            pose.bones[name].scale[row] = scale[row, col]


def writeScale(pose_dir):
    if not os.path.exists(pose_dir):
        os.makedirs(pose_dir)

    names = list(HAND_BONE_NAMES)
    names.extend(ARM_BONE_NAMES)

    pose = bpy.data.objects['Hum2'].pose
    scale = np.zeros((3, len(names)))
    for col, name in enumerate(names):
        for row in range(3):
            scale[row, col] = pose.bones[name].scale[row]

    ts = getTs()
    out_path = os.path.join(pose_dir, 'scale_%s.csv' % ts)

    print(out_path)
    print(scale)
    np.savetxt(out_path, scale, delimiter=',')


def readPose(pose_path):
    return np.genfromtxt(pose_path, delimiter=',')


def loadArmPose(path):
    pose = readPose(path)
    setArmPoseQuat(pose)


def loadHandPose(path):
    pose = readPose(path)
    setHandPoseQuat(pose)


def loadScale(path):
    scale = readPose(path)
    setScale(scale)

# ------------------------------------------------------------------------------


def getJointAnno3d(mesh, mesh_vert_idx):
    scene = bpy.data.scenes[SCENE]
    wco = mesh.vertices[mesh_vert_idx].co
    cco = bpy_extras.object_utils.world_to_camera_view(
        scene, bpy.data.objects[CAMERA], wco)

    cco[0] = cco[0] * scene.render.resolution_x
    cco[1] = (1-cco[1]) * scene.render.resolution_y

    coord = np.ndarray((3,))
    coord[0] = cco[0]
    coord[1] = cco[1]
    coord[2] = cco[2]

    return coord


def getJointAnno2d(mesh_name, mesh, mesh_vert_idx):
    scene = bpy.data.scenes[SCENE]
    wco = mesh.vertices[mesh_vert_idx].co
    cco = bpy_extras.object_utils.world_to_camera_view(
        scene, bpy.data.objects[CAMERA], wco)

    cam = bpy.data.objects[CAMERA]
    obj = bpy.data.objects[mesh_name]
    model_view = (
        cam.matrix_world.inverted() *
        obj.matrix_world
    )

    width = scene.render.resolution_x
    height = scene.render.resolution_y
    aspect_ratio = width / height

    n = cam.data.clip_start
    f = cam.data.clip_end
    fov = cam.data.angle

    proj = mu.Matrix()
    proj[0][0] = 1 / math.tan(fov / 2)
    proj[1][1] = aspect_ratio / math.tan(fov / 2)
    proj[2][2] = -(f + n) / (f - n)
    proj[2][3] = - 2*f*n / (f - n)
    proj[3][2] = - 1
    proj[3][3] = 0

    clip = proj * model_view

    v_4d = wco.copy()
    v_4d.resize_4d()

    v_clip = clip * v_4d
    v_clip /= v_clip[3]
    v_co = v_clip.resized(3)

    scrn_co_x = (v_co.x + 1) / 2 * width
    scrn_co_y = (v_co.y + 1) / 2 * height

    coord = np.ndarray((3,))
    coord[0] = scrn_co_x
    coord[1] = height - scrn_co_y
    coord[2] = cco[2]

    return coord


def get_hand_pose(mesh_name):
    scene = bpy.data.scenes[SCENE]
    scene.update()

    o = bpy.data.objects[mesh_name]
    # deactivate modifiers except armature
    viewport_states = []
    for mod in o.modifiers:
        viewport_states.append(mod.show_viewport)
        if mod.type != 'ARMATURE':
            mod.show_viewport = False

    # retrive posed mesh
    me = o.to_mesh(scene=scene, apply_modifiers=True, settings='PREVIEW')

    # activiate modifiers except armature
    for i, mod in enumerate(o.modifiers):
        mod.show_viewport = viewport_states[i]

    # joint coordinates
    anno3d = np.zeros((3, 20))
    vertice_idxs = [(3650, 3247),
                    (3603, 3639), (2713, 2720), (2677, 2777),
                    (1945, 3104), (1974, 1970), (1985, 1984), (1953, 2980),
                    (3269, 3110), (2194, 2095), (2177, 2176), (2145, 3019),
                    (2329, 3116), (2358, 2354), (2369, 2368), (2311, 2463),
                    (2521, 3124), (2637, 2478), (2636, 2597), (2529, 3097)]
    for joint_idx, vertice_idx in enumerate(vertice_idxs):
        vertice_anno = []
        if hasattr(vertice_idx, '__iter__'):
            for v_idx in vertice_idx:
                vertice_anno.append(getJointAnno3d(me, v_idx))
        else:
            vertice_anno.append(getJointAnno3d(me, vertice_idx))
        vertice_anno = np.array(vertice_anno)

        # print(80*'-')
        # print(vertice_anno)
        #print(np.mean(vertice_anno, axis=0))
        anno3d[:, joint_idx] = np.mean(vertice_anno, axis=0)


    bm = bmesh.new()
    bm.from_mesh(me)
    # bm = bmesh.from_edit_mesh(me)

    hand_vertex_group_names = [RIG_SPECIFIC_PREFIX + h for h in HAND_BONE_NAMES]
    group_indices = set([idx for idx, vg in enumerate(o.vertex_groups) if vg.name in hand_vertex_group_names])
    vertex_indices = set([idx for idx, v in enumerate(o.data.vertices) if group_indices & set([vg.group for vg in v.groups])])
    delete_verts = [v for v in bm.verts if v.index not in vertex_indices]

    bmesh.ops.delete(bm, geom=delete_verts, context=1)

    # bmesh.update_edit_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method=0, ngon_method=0)


    mat = bpy.data.objects['Hum2'].pose.bones['hand.R'].matrix

    bm.transform(mat.inverted())
    # Finish up, write the bmesh back to the mesh

    # vertices = np.array(db_object['mesh/vertices'])
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    hand_vertices = np.array([v.co for v in bm.verts])
    hand_triangles = np.array([v.index for f in bm.faces for v in f.verts]).reshape(-1, 3)


    # print([v.index for v in bm.faces[0].verts])
    # print([v.co for v in bm.faces[0].verts])
    # print([hand_vertices[v.index] for v in bm.faces[0].verts])
    # hand_triangles = []
    # for f in bm.faces:
    #     hand_triangles.append([v.index for v in f.verts])
    # hand_triangles = np.array(hand_triangles)

    #bm.to_mesh(me)
    # bm.free()

    # obj = bpy.data.objects.new('hand_copy', me)  # add a new object using the mesh
    # obj.rotation_mode = 'QUATERNION'

    # scene = bpy.context.scene
    # scene.objects.link(obj)  # put the object into the scene (link)
    # scene.objects.active = obj  # set as the active object in the scene
    # obj.select = True  # select object

    # # remove posed mesh
    bpy.data.meshes.remove(me)

    return anno3d, hand_vertices, hand_triangles


def render(depth_path, rgb_path):
    scene = bpy.data.scenes[SCENE]
    scene.update()

    node_output_file = scene.node_tree.nodes['image_output']
    node_output_file.base_path = os.path.dirname(depth_path)
    node_output_file.file_slots[0].path = os.path.basename(depth_path)
    node_output_file.file_slots[1].path = os.path.basename(rgb_path)

    bpy.ops.render.render()

# def getIntrinsics():
    # http://cmp.felk.cvut.cz/ftp/articles/svoboda/Mazany-TR-2007-02.pdf
    #f = bpy.data.cameras[CAMERA].lens / 16.0
    #width = bpy.context.scene.render.resolution_x
    #height = bpy.context.scene.render.resolution_y
    #kv = bpy.context.scene.render.pixel_aspect_x
    #ku = bpy.context.scene.render.pixel_aspect_y

    # if(width * kv > height * ku):
    #mv = width / 2
    #mu = mv * kv / ku
    # else:
    #mu = height / 2
    #mv = mu * ku / kv

    #f_x = mu * f
    #f_y = mv * f
    #p_x = width / 2
    #p_y = height / 2

    # return (f_x, f_y, p_x, p_y)


def getIntrinsics():
    scn = bpy.data.scenes[SCENE]

    scale = scn.render.resolution_percentage / 100
    size_w = scn.render.resolution_x * scale
    size_h = scn.render.resolution_y * scale

    cam = bpy.data.cameras[CAMERA]

    # compute K
    f_x = size_w * cam.lens / cam.sensor_width
    f_y = size_h * cam.lens / cam.sensor_height
    p_x = size_w / 2
    p_y = size_h / 2
    
    return (f_x, f_y, p_x, p_y)


def writeIntrinsics(path):
    f_x, f_y, p_x, p_y = getIntrinsics()

    with open(path, 'w') as f:
        f.write('%f %f\n' % (f_x, f_y))
        f.write('%f %f\n' % (p_x, p_y))
        f.write('0.0 0.0 0.0 0.0 0.0\n')
        f.close()


def loadPoses(pose_paths):
    poses = np.array([])
    for idx, pose_path in enumerate(pose_paths):
        if idx == 0:
            pose = readPose(pose_path)
            poses = np.ndarray((len(pose_paths), pose.shape[0], pose.shape[1]))
            poses[0, :, :] = pose
        else:
            poses[idx, :, :] = readPose(pose_path)

    return poses


def setScaleHand(x, y, z, scale_palm):
    bones = list(HAND_BONE_NAMES)
    if scale_palm:
        bones.append(ARM_BONE_NAMES[-1])

    pose = bpy.data.objects['Hum2'].pose
    for name in bones:
        pose.bones[name].scale[0] = x
        pose.bones[name].scale[1] = y
        pose.bones[name].scale[2] = z


def sequentialIdxPairs(n):  # idx with idx+1
    return [(idx, idx+1) for idx in range(n-1)]


def allIdxPairs(n):  # every idx with all other
    return [(idx1, idx2) for idx1 in range(n-1) for idx2 in range(idx1+1, n)]


def randKIdxPairs(n, k):  # idx with random k others
    idx_pairs = []
    for idx_1 in range(n):
        for k_ in range(k):
            idx_2 = random.randint(0, n-1)
            while idx_2 == idx_1:
                idx_2 = random.randint(0, n-1)
            idx_pairs.append((idx_1, idx_2))
    return idx_pairs


def rand1IdxPairs(n):
    return randKIdxPairs(n, 1)


def rand2IdxPairs(n):
    return randKIdxPairs(n, 2)


def rand3IdxPairs(n):
    return randKIdxPairs(n, 3)


def interpolatePoses(poses, steps, createIdxPairs):
    if poses.shape[0] == 1:
        return poses.copy()

    idx_pairs = createIdxPairs(poses.shape[0])

    space = np.linspace(0.0, 1.0, steps)

    inter_idx = 0
    interpolations = np.zeros(
        (len(idx_pairs) * steps, poses.shape[1], poses.shape[2]))
    for idx_1, idx_2 in idx_pairs:
        for s in space:
            interpolations[inter_idx] = s * \
                poses[idx_1] + (1.0 - s) * poses[idx_2]
            inter_idx += 1

    return interpolations


def add_object(db_object):
    # db_object['mesh'].keys()
    #   <KeysViewHDF5 ['connected_components', 'metadata', 'triangles', 'vertices']>

    # verts =  #[(1, 1, 1), (0, 0, 0)]  # 2 verts made with XYZ coords
    name = db_object.name.split('/')[-1]
    mesh = bpy.data.meshes.new(name+'_mesh')  # add a new mesh
    obj = bpy.data.objects.new(name, mesh)  # add a new object using the mesh
    obj.rotation_mode = 'QUATERNION'

    scene = bpy.context.scene
    scene.objects.link(obj)  # put the object into the scene (link)
    scene.objects.active = obj  # set as the active object in the scene
    obj.select = True  # select object

    mesh = bpy.context.object.data
    bm = bmesh.new()

    vertices_by_index = dict()
    vertices = np.array(db_object[MESH_KEY][MESH_VERTICES_KEY])
    for idx, v in enumerate(vertices):
        vertices_by_index[idx] = bm.verts.new(v)  # add a new vert

    # clean duplicate triangulation, there's triangles definitions such as [1 8 3] and [3 8 1] in 3dnet
    triangles = np.array(db_object[MESH_KEY][MESH_TRIANGLES_KEY])
    triangles = np.sort(triangles, axis=1)
    triangles = np.unique(triangles, axis=0)
    for idx, triangle in enumerate(triangles):
        # vertices_by_index[idx] = bm.verts.new(v)  # add a new vert
        bm.faces.new([vertices_by_index[idx]
                      for idx in triangle])  # add new triangular face

    obj.data.materials.append(bpy.data.materials['Object'])

    # make the bmesh the object's mesh
    bm.to_mesh(mesh)
    bm.free()  # always do this when finished
    return obj

def remove_object(obj):
    # bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.ops.object.delete()


def set_object_hand_constraint(obj):

    object_child_of_hand = obj.constraints.new('CHILD_OF')
    object_child_of_hand.target = bpy.data.objects['Hum2'] # object_child_of_hand.target = bpy.data.objects['Hum2:Body']
    object_child_of_hand.subtarget = RIG_SPECIFIC_PREFIX + 'hand.R'
    # object_child_of_hand.track_axis = 'TRACK_NEGATIVE_Z'
    # if CAMERA_CONSTRAINT == 'TRACK_TO':
    #   object_child_of_hand.use_target_z = False # for TRACK_TO
    #   object_child_of_hand.up_axis = 'UP_Y' # for TRACK_TO
    # object_child_of_hand.influence = 0.925



def configRendering():
    scene = bpy.data.scenes[SCENE]
    scene.render.layers['RenderLayer'].use_pass_combined = True  # render_img
    scene.render.layers['RenderLayer'].use_pass_z = True
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    bpy.data.worlds['World'].horizon_color = (0, 0, 0)

    size_w = scene.render.resolution_x * scene.render.resolution_percentage / 100
    size_h = scene.render.resolution_y * scene.render.resolution_percentage / 100

    cam = bpy.data.cameras[CAMERA]
    horizontal_fov = np.deg2rad(69.4)

    cam.lens = 1.88
    cam.sensor_fit = 'HORIZONTAL'
    cam.sensor_width = 2 * np.tan(horizontal_fov / 2) * cam.lens
    focus = size_w * cam.lens / cam.sensor_width
    cam.sensor_height = cam.lens * size_h / focus

    setupRenderingNodes()