from VisualSound.pose_providers.kinect_v2_pose import KinectV2PoseProvider
from VisualSound.pose_providers.mediapipe_cam import MediaPipeCamPoseProvider
from VisualSound.pose_providers.toycv_tk import ToyCvTkPoseProvider


def test_pose_provider_default_acoustic_frame_policy_constants():
    assert ToyCvTkPoseProvider.default_acoustic_frame_provider_name == "identity"
    assert MediaPipeCamPoseProvider.default_acoustic_frame_provider_name == "flip-front"
    assert KinectV2PoseProvider.default_acoustic_frame_provider_name == "flip-front"
