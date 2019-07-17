

CAMERA = 'Camera'
SCENE = 'Scene'

USE_MHX_RIG = False
RIG_SPECIFIC_PREFIX = ''
if USE_MHX_RIG:
    RIG_SPECIFIC_PREFIX = 'DEF-'

ARM_BONE_NAMES = ['DEF-deltoid.R', 'upper_arm.R', 'forearm.R', 'hand.R']
HAND_BONE_NAMES = ['thumb.01.R', 'thumb.02.R', 'thumb.03.R',
                   'palm_index.R', 'f_index.01.R', 'f_index.02.R', 'f_index.03.R',
                   'palm_middle.R', 'f_middle.01.R', 'f_middle.02.R', 'f_middle.03.R',
                   'palm_ring.R', 'f_ring.01.R', 'f_ring.02.R', 'f_ring.03.R',
                   'palm_pinky.R', 'f_pinky.01.R', 'f_pinky.02.R', 'f_pinky.03.R']