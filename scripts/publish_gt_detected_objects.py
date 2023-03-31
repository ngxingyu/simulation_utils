#!/usr/bin/env python3
from dataclasses import dataclass

import cv2
import numpy as np
import rospy
import tf2_ros
import tf.transformations
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import LinkState
from gazebo_msgs.srv import GetLinkState
from sensor_msgs.msg import CameraInfo, CompressedImage
import cv_bridge
import os

from bb_msgs.msg import DetectedObject, DetectedObjects
from typing import Dict, List
import numpy as np
from tf.transformations import quaternion_matrix
from operator import attrgetter
# moves object randomly in defined region


@dataclass
class Config:
    camera_link: str
    object_name: str
    camera_info: CameraInfo
    task: int
    label: str
    # object_centre: list
    # object_dimensions: list
    # keypoints: list
    object_points: list
    use_link_state: bool=False

def save_detections(img, vis, detections, output_dir, output_format, task, camera_pose, stamp, objects):
    os.makedirs(f"{output_dir}/{output_format}/{task}/labels", exist_ok=True)
    os.makedirs(f"{output_dir}/{output_format}/{task}/segmentation_labels", exist_ok=True)
    os.makedirs(f"{output_dir}/{output_format}/{task}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/{output_format}/{task}/vis", exist_ok=True)
    print(os.path.abspath(output_dir))
    if len(detections.detected) == 0:
        return
    filename = f"sim_{stamp.to_nsec()}_"+"_".join(map(str, map(int, camera_pose)))
    os.path.join(output_dir, output_format, task, "images", filename+".jpg")
    cv2.imwrite(os.path.join(output_dir, output_format, task, "images", filename+".jpg"), img)
    cv2.imwrite(os.path.join(output_dir, output_format, task, "vis", filename+".jpg"), vis)
    
    with open(os.path.join(output_dir, output_format, task, "labels", filename+".txt"), "a") as f:
        for detection in detections.detected:
            if detection.name in objects:
                f.write(f"{objects[detection.name]} {detection.centre_x/detection.image_width} {detection.centre_y/detection.image_height} {detection.bbox_width/detection.image_width} {detection.bbox_height/detection.image_height}\n")

class DetectedObjectsBboxPublisher:
    """Publishes ground truth for detected objects"""

    def __init__(self, configs, image_topic:str, output_topic: str, tasks: List[Dict[str, int]]):
        """
        :param configs: list of Config objects
        :param output_topic: topic to publish detected objects to
        """
        self.task = rospy.get_param("~task", default=5)
        print(self.task)
        self.configs = [config for config in configs if config.task == self.task]
        self.output_topic = output_topic
        self.pub = rospy.Publisher(self.output_topic, DetectedObjects, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(0.5))
        tf2_ros.TransformListener(self.tf_buffer)
        rospy.wait_for_service('/gazebo/get_model_state', 1.0)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        rospy.wait_for_service('/gazebo/get_link_state', 1.0)
        self.get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        # self.timer = rospy.Timer(rospy.Duration(0.1), self.publish)
        self.cv_bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber(image_topic, CompressedImage, self.publish)
        self.vis_pub = rospy.Publisher("/detected_objects/visualize/compressed", CompressedImage, queue_size=1)
        self.output_dir = rospy.get_param("output_dir", default="detections")
        # self.output_dir=None
        self.output_format = rospy.get_param("output_format", default="yolov8")
        self.objects = tasks[self.task]

    def get_detected_object(self, config: Config, time: rospy.Time, img: np.array):
        # object_centre = np.array(config.object_centre)
        # dim = config.object_dimensions
        objectPoints = config.object_points
        camera_transform: tf2_ros.TransformStamped = self.tf_buffer.lookup_transform(
            config.camera_link, "world", time, rospy.Duration(0.05))
        if config.use_link_state:
            object_pose: LinkState = self.get_link_state(config.object_name, '').link_state            
        else:
            object_pose: ModelState = self.get_model_state(config.object_name, '')
        object_x = object_pose.pose.position.x
        object_y = object_pose.pose.position.y
        object_z = object_pose.pose.position.z
        R = quaternion_matrix(attrgetter('x','y','z','w')(object_pose.pose.orientation))

        # Create a 4x4 transformation matrix representing the pose
        T = np.identity(4)
        T[:3, :3] = R[:3, :3]
        T[:3, 3] = attrgetter('x','y','z')(object_pose.pose.position)
        objectPoints = np.hstack([objectPoints, np.ones((objectPoints.shape[0],1))])
        objectPoints = objectPoints@T.T
        objectPoints = objectPoints[:,:3]/objectPoints[:,3:]
        camera_trans = camera_transform.transform.translation
        camera_trans = np.array([camera_trans.x, camera_trans.y, camera_trans.z])
        camera_R = camera_transform.transform.rotation
        camera_R = tf.transformations.quaternion_matrix([camera_R.x, camera_R.y, camera_R.z, camera_R.w])
        original_img_points = cv2.projectPoints(objectPoints, cv2.Rodrigues(camera_R[:3, :3])[0],
                                       camera_trans,
                                       np.array(config.camera_info.K).reshape(3, 3),
                                       np.array(config.camera_info.D), None)[0].squeeze()
        img_points = original_img_points.copy()
        img_points[:, :1] = np.clip(original_img_points[:, :1], 0, config.camera_info.width)
        img_points[:, 1:2] = np.clip(original_img_points[:, 1:2], 0, config.camera_info.height)
        bbox_x_min = min(img_points[:, 0])
        bbox_x_max = max(img_points[:, 0])
        bbox_y_min = min(img_points[:, 1])
        bbox_y_max = max(img_points[:, 1])
        original_bbox_x_min = min(original_img_points[:, 0])
        original_bbox_x_max = max(original_img_points[:, 0])
        original_bbox_y_min = min(original_img_points[:, 1])
        original_bbox_y_max = max(original_img_points[:, 1])
        original_dims = (original_bbox_x_max-original_bbox_x_min) * (original_bbox_y_max - original_bbox_y_min)
        dims = (bbox_x_max-bbox_x_min) * (bbox_y_max-bbox_y_min)
        # if dims < original_dims * 0.45 or dims == 0:
        #     return None, img, camera_trans
        detected_bbox = DetectedObject()
        detected_bbox.header.frame_id = config.camera_link
        detected_bbox.header.stamp = time
        detected_bbox.extra_codex = 1
        detected_bbox.name = config.label
        detected_bbox.extra = [0]
        detected_bbox.centre_x = int(bbox_x_min + (bbox_x_max - bbox_x_min) / 2)
        detected_bbox.centre_y = int(bbox_y_min + (bbox_y_max - bbox_y_min) / 2)
        detected_bbox.bbox_width = int(bbox_x_max - bbox_x_min)
        detected_bbox.bbox_height = int(bbox_y_max - bbox_y_min)
        # detected_bbox.world_coords = (np.array([object_x, object_y, object_z]) + object_centre).tolist()
        detected_bbox.image_width = img.shape[1]
        detected_bbox.image_height = img.shape[0]
        
        for i, point in enumerate(img_points):
            cv2.circle(img, tuple(point.astype(int)), 5, (0, 255, 0), -1)
        cv2.rectangle(img, (int(bbox_x_min), int(bbox_y_min)), (int(bbox_x_max), int(bbox_y_max)), (255, 0, 0), 2)
        if detected_bbox.bbox_width > 0 and detected_bbox.bbox_height > 0:
            detected_bbox.extra = [1]
            return detected_bbox, img, camera_trans
        else:
            return None, img, camera_trans

    def publish(self, msg: CompressedImage):
        img = self.cv_bridge.compressed_imgmsg_to_cv2(msg)

        detected = []
        time = msg.header.stamp
        vis = img.copy()
        for config in self.configs:
            detected_bbox, vis, camera_pose = self.get_detected_object(config, time, vis)
            if detected_bbox is not None:
                detected.append(detected_bbox)
            else:
                rospy.loginfo_throttle(1, "Object not in camera view")
        detected_objects = DetectedObjects(detected=detected)

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', vis)[1]).tostring()
        self.vis_pub.publish(msg)
        detected_objects.header.stamp = time
        self.pub.publish(detected_objects)
        if self.output_dir is not None:
            save_detections(img, vis, detected_objects, self.output_dir, self.output_format, str(self.task), camera_pose, msg.header.stamp, self.objects, )
            



if __name__ == "__main__":
    rospy.init_node("random_model_motion", anonymous=True)
    front_camera_info = rospy.wait_for_message("/auv4/front_cam/camera_info", CameraInfo, timeout=2)
    bot_camera_info = rospy.wait_for_message("/auv4/bot_cam/camera_info", CameraInfo, timeout=2)
    # bot_camera_info = rospy.wait_for_message(
    #     "/auv4/bot_cam/camera_info", CameraInfo, timeout=2)
# -0.846967
    octagon_symbol_points = np.array([[-0.1, 0, -0.1], [0.1, 0, -0.1], [0.1, 0, 0.1], [-0.1, 0, 0.1]])
    bottle_points = np.array([[-0.057, -0.057, -0.089], [-0.057, 0.057, -0.089], [0.057, 0.057, -0.089], [0.057, -0.057, -0.089],
                                      [-0.057, -0.057, 0.089], [-0.057, 0.057, 0.089], [0.057, 0.057, 0.089], [0.057, -0.057, 0.089]])
    buoy_points = np.array([[-0.5842, 0, 0.5254], [-0.5842, 0, 1.6938], [0.5842, 0, 1.6938],  [0.5842, 0, 0.5254]])
    path_marker_points = np.array([[0.7,-0.3,0], [0.7,-0.3,0.4], [0,0,0], [0,0,0.4], [-0.7,0,0], [-0.7,0,0.4]])
    gate_symbol_points = np.array([[-0.05, 0, -0.05], [-0.05, 0, 0.05], [0.05, 0, 0.05], [0.05, 0, -0.05]])
    buoy_symbol_points = np.array([[-0.2921, 0, -0.2921], [0.2921, 0, -0.2921], [0.2921, 0, 0.2921], [0.2921, 0, -0.2921]])
    bin_points = np.array([
        []
    ])
    configs = [
        # Config("auv4/front_cam_optical", "robosub_torpedo_gman", front_camera_info, "Gman", object_centre = [0, 0, 1.11], object_dimensions=[0.7, 0, 1.6]),
        Config("auv4/front_cam_optical", "robosub23_gate", front_camera_info, 0, "Gate", object_points=np.array([[-1.5024,0,0], [-1.5024,0,1.5024], [1.5024,0,1.5024], [1.5024,0,0]]),use_link_state=False),
        # Config("auv4/front_cam_optical", "robosub23_buoy_v1", front_camera_info,
        #        0, "Buoy", object_points=np.array([[-0.5842, 0, 0.5254], [-0.5842, 0, 1.6938], [0.5842, 0, 1.6938],  [0.5842, 0, 0.5254]]), use_link_state=False),
        Config("auv4/front_cam_optical", "robosub23_buoy_v2", front_camera_info,
               0, "Buoy", object_points=buoy_points, use_link_state=False),
        Config("auv4/front_cam_optical", "robosub23_torpedo_panel_v2", front_camera_info,
               0, "Torpedo", object_points=buoy_points, use_link_state=False),
        Config("auv4/front_cam_optical", "path_marker_1::path_marker_45ccw", front_camera_info,
               0, "PathMarker", object_points=path_marker_points, use_link_state=True),
        Config("auv4/front_cam_optical", "path_marker_2::path_marker_45ccw", front_camera_info,
               0, "PathMarker", object_points=path_marker_points, use_link_state=True),
        # # right
        Config("auv4/front_cam_optical", "robosub23_gate::box_earth0", front_camera_info,
               1, "earth0", object_points=gate_symbol_points, use_link_state=True),
        Config("auv4/front_cam_optical", "robosub23_gate::box_abydos0", front_camera_info,
               1, "abydos0", object_points=gate_symbol_points, use_link_state=True),               
        # # earth1
        # Config("auv4/front_cam_optical", "robosub23_buoy_v1::box_tl", front_camera_info,
        #        2, "earth1", object_points=buoy_symbol_points, use_link_state=True),
        # # abydos1
        # Config("auv4/front_cam_optical", "robosub23_buoy_v1::box_tr", front_camera_info,
        #        2, "abydos1", object_points=buoy_symbol_points, use_link_state=True),
        # # abydos2
        # Config("auv4/front_cam_optical", "robosub23_buoy_v1::box_bl", front_camera_info,
        #        2, "abydos2", object_points=buoy_symbol_points, use_link_state=True),
        # # earth2
        # Config("auv4/front_cam_optical", "robosub23_buoy_v1::box_br", front_camera_info,
        #        2, "earth2", object_points=buoy_symbol_points, use_link_state=True),
        # earth1
        Config("auv4/front_cam_optical", "robosub23_buoy_v2::box_tl", front_camera_info,
               2, "abydos2", object_points=buoy_symbol_points, use_link_state=True),
        # abydos1
        Config("auv4/front_cam_optical", "robosub23_buoy_v2::box_tr", front_camera_info,
               2, "earth2", object_points=buoy_symbol_points, use_link_state=True),
        # abydos2
        Config("auv4/front_cam_optical", "robosub23_buoy_v2::box_bl", front_camera_info,
               2, "earth1", object_points=buoy_symbol_points, use_link_state=True),
        # earth2
        Config("auv4/front_cam_optical", "robosub23_buoy_v2::box_br", front_camera_info,
               2, "abydos1", object_points=buoy_symbol_points, use_link_state=True),

        Config("auv4/front_cam_optical", "robosub23_torpedo_panel_v1", front_camera_info,
               3, "opened", object_points=np.array([[-0.5842, 0, 1.1096], [-0.5842, 0, 1.1096 + 0.5842], [0.5842, 0, 1.1096+0.5842], [0.5842, 0, 1.1096]]), use_link_state=False),
        Config("auv4/front_cam_optical", "robosub23_torpedo_panel_v1", front_camera_info,
               3, "closed", object_points=np.array([[-0.5842, 0, 1.1096], [-0.5842, 0, 1.1096 - 0.5842], [0.5842, 0, 1.1096-0.5842], [0.5842, 0, 1.1096]]), use_link_state=False),
        Config("auv4/front_cam_optical", "robosub23_torpedo_panel_v1", front_camera_info,
               3, "hole", object_points=np.array([[-0.1, 0, 1.1096 + 0.19], [-0.1, 0, 1.1096 + 0.39], [0, 0, 1.1096 + 0.39], [0.1, 0, 1.1096 + 0.19]])),
        Config("auv4/front_cam_optical", "robosub23_torpedo_panel_v1", front_camera_info,
               3, "hole", object_points=np.array([[-0.1, 0, 1.1096 -0.1774], [-0.1, 0, 1.1096 -0.3774], [0, 0, 1.1096-0.3774], [0.1, 0, 1.1096 -0.1774]])),
        Config("auv4/front_cam_optical", "robosub23_torpedo_panel_v2", front_camera_info,
               3, "opened", object_points=np.array([[-0.5842, 0, 1.1096], [-0.5842, 0, 1.1096 + 0.5842], [0.5842, 0, 1.1096+0.5842], [0.5842, 0, 1.1096]]), use_link_state=False),
        Config("auv4/front_cam_optical", "robosub23_torpedo_panel_v2", front_camera_info,
               3, "closed", object_points=np.array([[-0.5842, 0, 1.1096], [-0.5842, 0, 1.1096 - 0.5842], [0.5842, 0, 1.1096-0.5842], [0.5842, 0, 1.1096]]), use_link_state=False),
        Config("auv4/front_cam_optical", "robosub23_torpedo_panel_v2", front_camera_info,
               3, "hole", object_points=np.array([[-0.1, 0, 1.1096 + 0.19], [-0.1, 0, 1.1096 + 0.39], [0, 0, 1.1096 + 0.39], [0.1, 0, 1.1096 + 0.19]])),
        Config("auv4/front_cam_optical", "robosub23_torpedo_panel_v2", front_camera_info,
               3, "hole", object_points=np.array([[-0.1, 0, 1.1096 -0.1774], [-0.1, 0, 1.1096 -0.3774], [0, 0, 1.1096-0.3774], [0.1, 0, 1.1096 -0.1774]])),

        Config("auv4/bot_cam_optical", "robosub23_octagon::symbol1", bot_camera_info, 5, "earth3", object_points=octagon_symbol_points, use_link_state=True),
        Config("auv4/bot_cam_optical", "robosub23_octagon::symbol2", bot_camera_info, 5, "earth4", object_points=octagon_symbol_points, use_link_state=True),
        Config("auv4/bot_cam_optical", "robosub23_octagon::symbol3", bot_camera_info, 5, "abydos3", object_points=octagon_symbol_points, use_link_state=True),
        Config("auv4/bot_cam_optical", "robosub23_octagon::symbol4", bot_camera_info, 5, "abydos4", object_points=octagon_symbol_points, use_link_state=True),
        Config("auv4/bot_cam_optical", "robosub23_octagon::symbol5", bot_camera_info, 5, "abydos5", object_points=octagon_symbol_points, use_link_state=True),
        Config("auv4/bot_cam_optical", "robosub23_octagon::symbol6", bot_camera_info, 5, "earth5", object_points=octagon_symbol_points, use_link_state=True),
        Config("auv4/bot_cam_optical", "robosub23_octagon::symbol7", bot_camera_info, 5, "abydos6", object_points=octagon_symbol_points, use_link_state=True),
        Config("auv4/bot_cam_optical", "robosub23_octagon::symbol8", bot_camera_info, 5, "earth6", object_points=octagon_symbol_points, use_link_state=True),
        Config("auv4/bot_cam_optical", "bottle_1::bottle", bot_camera_info, 5, "chevron", object_points=bottle_points, use_link_state=True),
        Config("auv4/bot_cam_optical", "bottle_2::bottle", bot_camera_info, 5, "chevron", object_points=bottle_points, use_link_state=True),
        Config("auv4/bot_cam_optical", "bottle_3::bottle", bot_camera_info, 5, "chevron", object_points=bottle_points, use_link_state=True),
        Config("auv4/bot_cam_optical", "bottle_4::bottle", bot_camera_info, 5, "chevron", object_points=bottle_points, use_link_state=True),
    ]
    tasks = [
        {"Gate" : 0, "Buoy" : 1, "Torpedo": 2, "Bin": 3, "DHD": 4, "PathMarker":5},
        {"earth0": 0, "abydos0": 1}, # gate
        {"earth1": 0, "earth2": 1, "abydos1": 2, "abydos2": 3}, # buoy
        {"opened": 0, "closed": 1, "hole": 2}, # torpedo
        {"earth": 0, "abydos": 1, "bin": 2, "lid": 3, "handle": 4}, # bin
        {"earth3": 0, "earth4": 1, "earth5": 2, "earth6": 3, "abydos3": 4, "abydos4": 5, "abydos5": 6, "abydos6": 7, "chevron": 8, "octagon": 9} # DHD
    ]
    motion = DetectedObjectsBboxPublisher(configs, "/auv4/front_cam/image_rect_color/compressed", "detected_objects", tasks)
    # motion = DetectedObjectsBboxPublisher(configs, "/auv4/bot_cam/image_rect_color/compressed", "detected_objects", tasks)
    rospy.spin()
