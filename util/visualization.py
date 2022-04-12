

def render_smpl(img_filename, img_out_filename, mocap_points):
    mocap_joint_names = ['Hips', 'Spine', 'Spine1', 'Spine2', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftInHandPinky', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky3End', 'LeftInHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing3End', 'LeftInHandMiddle', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle3End', 'LeftInHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex3End', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb3End', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                         'RightInHandPinky', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky3End', 'RightInHandRing', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing3End', 'RightInHandMiddle', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle3End', 'RightInHandIndex', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex3End', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb3End', 'Neck', 'Neck1', 'Head', 'HeadEnd', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftFootEnd', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightFootEnd']

    name_to_index = dict(
        zip(mocap_joint_names, list(range(len(mocap_joint_names)))))

    smpl_to_imu = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg',
                   'RightLeg', 'Spine1', 'LeftFoot', 'RightFoot', 'Spine2',
                   'LeftFootEnd', 'RightFootEnd', 'Neck', 'LeftShoulder',
                   'RightShoulder', 'Head', 'LeftArm', 'RightArm',
                   'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand',
                   'LeftHandThumb2', 'RightHandThumb2']

    for each in smpl_to_imu:
        pass


if __name__ == '__main__':
    render_smpl(None, None, None)
