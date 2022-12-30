#!/usr/bin/env python3
import cv2
import message_filters
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage

from bb_msgs.msg import DetectedObjects

# moves object randomly in defined region

class DetectedObjectsVisualizer:
    """Visualizes detected objects for a given camera"""
    def __init__(self, camera_image_topic, detected_objects_topic, output_topic):
        self.camera_image_topic = camera_image_topic
        self.detected_objects_topic = detected_objects_topic
        self.output_topic = output_topic
        self.bridge = CvBridge()
        self.pub = rospy.Publisher(self.output_topic, CompressedImage, queue_size=1)
        self.detected_objects_sub = message_filters.Subscriber(detected_objects_topic, DetectedObjects)
        self.image_sub = message_filters.Subscriber(camera_image_topic, CompressedImage)
        self.detected_objects_cache = message_filters.Cache(self.detected_objects_sub, 100, allow_headerless=True)
        ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.detected_objects_sub], 10, 0.2, allow_headerless=True)
        ts.registerCallback(self.visualize_objects)

    def visualize_objects(self, img_msg, detected_objects):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        camera_frame = img_msg.header.frame_id

        for obj in detected_objects.detected:
            if obj.header.frame_id == camera_frame:
                x_min = int(obj.centre_x - obj.bbox_width / 2)
                y_min = int(obj.centre_y - obj.bbox_height / 2)
                x_max = int(obj.centre_x + obj.bbox_width / 2)
                y_max = int(obj.centre_y + obj.bbox_height / 2)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(img, obj.name, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        self.pub.publish(self.bridge.cv2_to_compressed_imgmsg(img))

if __name__ == "__main__":
    rospy.init_node("detected_objects_visualiser")

    detected_objects_topic = rospy.get_param("~detected_objects_topic", "/detected_objects")
    camera_image_topic = rospy.get_param("~camera_image_topic", "/auv4/front_cam/image_rect_color/compressed")
    output_topic = rospy.get_param("~output_topic","/auv4/front_cam/visualize/compressed")
    DetectedObjectsVisualizer(camera_image_topic, detected_objects_topic, output_topic)
    rospy.spin()
