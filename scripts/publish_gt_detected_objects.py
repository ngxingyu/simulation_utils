#!/usr/bin/env python3
from dataclasses import dataclass

import cv2
import numpy as np
import rospy
import tf2_ros
import tf.transformations
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from sensor_msgs.msg import CameraInfo

from bb_msgs.msg import DetectedObject, DetectedObjects

# moves object randomly in defined region


@dataclass
class Config:
    camera_link: str
    object_name: str
    camera_info: CameraInfo
    label: str
    object_centre: list
    object_dimensions: list


class DetectedObjectsBboxPublisher:
    """Publishes ground truth for detected objects"""

    def __init__(self, configs, output_topic: str):
        """
        :param configs: list of Config objects
        :param output_topic: topic to publish detected objects to
        """
        self.configs = configs
        self.output_topic = output_topic
        self.pub = rospy.Publisher(self.output_topic, DetectedObjects, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buffer)
        rospy.wait_for_service('/gazebo/get_model_state', 1.0)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish)

    def get_detected_object(self, config: Config):
        object_centre = np.array(config.object_centre)
        dim = config.object_dimensions
        camera_transform: tf2_ros.TransformStamped = self.tf_buffer.lookup_transform(
            config.camera_link, "world", rospy.Time(), rospy.Duration(1.0))
        object_pose: ModelState = self.get_model_state(config.object_name, '')
        object_x = object_pose.pose.position.x
        object_y = object_pose.pose.position.y
        object_z = object_pose.pose.position.z

        objectPoints = np.array([[object_x - dim[0], object_y - dim[1], object_z + dim[2]],
                                 [object_x + dim[0], object_y - dim[1], object_z + dim[2]],
                                 [object_x + dim[0], object_y + dim[1], object_z - dim[2]],
                                 [object_x - dim[0], object_y + dim[1], object_z - dim[2]]], dtype=np.float32)
        objectPoints += object_centre
        camera_trans = camera_transform.transform.translation
        camera_trans = np.array([camera_trans.x, camera_trans.y, camera_trans.z])
        camera_R = camera_transform.transform.rotation
        camera_R = tf.transformations.quaternion_matrix([camera_R.x, camera_R.y, camera_R.z, camera_R.w])

        img_points = cv2.projectPoints(objectPoints, cv2.Rodrigues(camera_R[:3, :3])[0],
                                       camera_trans,
                                       np.array(config.camera_info.K).reshape(3, 3),
                                       np.array(config.camera_info.D), None)[0].squeeze()
        img_points[:, :1] = np.clip(img_points[:, :1], 0, config.camera_info.width)
        img_points[:, 1:2] = np.clip(img_points[:, 1:2], 0, config.camera_info.height)
        bbox_x_min = min(img_points[:, 0])
        bbox_x_max = max(img_points[:, 0])
        bbox_y_min = min(img_points[:, 1])
        bbox_y_max = max(img_points[:, 1])
        detected_bbox = DetectedObject()
        detected_bbox.header.frame_id = config.camera_link
        detected_bbox.header.stamp = rospy.Time.now()
        detected_bbox.extra_codex = 1
        detected_bbox.name = config.label
        detected_bbox.extra = [0]
        detected_bbox.centre_x = int(bbox_x_min + (bbox_x_max - bbox_x_min) / 2)
        detected_bbox.centre_y = int(bbox_y_min + (bbox_y_max - bbox_y_min) / 2)
        detected_bbox.bbox_width = int(bbox_x_max - bbox_x_min)
        detected_bbox.bbox_height = int(bbox_y_max - bbox_y_min)
        detected_bbox.world_coords = (np.array([object_x, object_y, object_z]) + object_centre).tolist()
        if detected_bbox.bbox_width > 0 and detected_bbox.bbox_height > 0:
            detected_bbox.extra = [1]
            return detected_bbox
        else:
            return None

    def publish(self, event):
        detected = []
        for config in self.configs:
            detected_bbox = self.get_detected_object(config)
            if detected_bbox is not None:
                detected.append(detected_bbox)
            else:
                rospy.loginfo_throttle(1, "Object not in camera view")
        detected_objects = DetectedObjects(detected=detected)
        detected_objects.header.stamp = rospy.Time.now()
        self.pub.publish(detected_objects)


if __name__ == "__main__":
    rospy.init_node("random_model_motion", anonymous=True)
    front_camera_info = rospy.wait_for_message("/auv4/front_cam/camera_info", CameraInfo, timeout=2)
    # bot_camera_info = rospy.wait_for_message(
    #     "/auv4/bot_cam/camera_info", CameraInfo, timeout=2)

    configs = [
        # Config("auv4/front_cam_optical", "robosub_torpedo_gman", front_camera_info, "Gman", object_centre = [0, 0, 1.11], object_dimensions=[0.7, 0, 1.6]),
        Config("auv4/front_cam_optical", "robosub_torpedo_bootlegger", front_camera_info,
               "Bootlegger", object_centre=[0, 1, 1.11], object_dimensions=[0.4, 0, 0.8])
    ]
    motion = DetectedObjectsBboxPublisher(configs, "detected_objects")
    rospy.spin()
