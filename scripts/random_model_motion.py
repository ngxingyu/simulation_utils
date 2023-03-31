#!/usr/bin/env python3
from math import pi

import numpy as np
import rospy
import tf2_ros
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from tf.transformations import quaternion_from_euler

# moves object randomly in defined region


class ObjectMotion:
    def __init__(self, child, parent, child_tf=None, parent_tf=None, x_min=-3, x_max=-3, y_min=-3, y_max=-3, z_min=0, z_max=2,
                 roll_min=-pi/5, roll_max=pi/5, pitch_min=-pi/5, pitch_max=pi/5, yaw_min=-pi/5, yaw_max=pi/5,
                 vx=0.04, vy=0.04, vz=0.04, vroll=pi/30, vpitch=pi/30, vyaw=pi/30):
        if child_tf is None:
            child_tf = child
        if parent_tf is None:
            parent_tf = parent
        self.v_lim = np.array([vx, vy, vz, vroll, vpitch, vyaw])
        self.X = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2,
                           (roll_min + roll_max) / 2, (pitch_min + pitch_max) / 2, (yaw_min + yaw_max) / 2, 0, 0, 0, 0, 0, 0])
        self.min = np.array([x_min, y_min, z_min, roll_min, pitch_min, yaw_min])
        self.max = np.array([x_max, y_max, z_max, roll_max, pitch_max, yaw_max])
        self.mean = np.array([x_max + x_min, y_max + y_min, z_max + z_min,
                             roll_max + roll_min, pitch_max + pitch_min, yaw_max + yaw_min]) / 2
        self.parent_link = parent
        self.child_link = child
        self.parent_tf = parent_tf
        self.child_tf = child_tf
        self.br = tf2_ros.TransformBroadcaster()
        self.transform_stamped = tf2_ros.TransformStamped()
        self.seq=0

    def start(self):
        self.timer = rospy.Timer(rospy.Duration(0.1), self.update)

    def stop(self):
        if self.timer:
            self.timer.shutdown()

    def update(self, event):
        self.X[6:] = (self.X[6:] + np.random.uniform(-1, 1, 6)) / 3
        self.X[6:] = np.clip(self.X[6:], -self.v_lim, self.v_lim)

        self.X[:6] = self.X[:6] + self.X[6:]
        self.X[:6] = np.clip(self.X[:6], self.min, self.max)
        self.publish_state()

    def publish_tf(self, pose: Pose, frame_id, child_frame_id):
        self.transform_stamped.transform.translation = pose.position
        self.transform_stamped.transform.rotation = pose.orientation

        self.transform_stamped.header.stamp = rospy.Time.now()
        self.transform_stamped.header.frame_id = frame_id
        self.transform_stamped.child_frame_id = child_frame_id
        self.br.sendTransform(self.transform_stamped)

    def publish_state(self):
        rospy.wait_for_service('/gazebo/set_model_state', 1.0)
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        pose = Pose(
            position=Point(x=self.X[0], y=self.X[1], z=self.X[2]),
            orientation=Quaternion(*quaternion_from_euler(self.X[3], self.X[4], self.X[5]))
        )
        set_state(ModelState(
            model_name=self.child_link,
            pose=pose,
            twist=Twist(
                linear=Vector3(x=self.X[6], y=self.X[7], z=self.X[8]),
                angular=Vector3(x=self.X[9], y=self.X[10], z=self.X[11])
            ),
            reference_frame=self.parent_link
        ))
        # self.publish_tf(pose, self.parent_tf, self.child_tf)


if __name__ == "__main__":
    rospy.init_node("random_model_motion", anonymous=True)
    # motion = ObjectMotion("robosub_buoy_tommy_gun", "auv4", x_min=1, x_max=6, y_min=-3, y_max=3, z_min=-3, z_max=-1,
    #                     roll_min=-pi/7, roll_max=pi/7, pitch_min=-pi/7, pitch_max=pi/7, yaw_min=-pi/7 + pi/2, yaw_max=pi/7 + pi/2)
    # motion = ObjectMotion("robosub_torpedo_bootlegger", "auv4", x_min=1, x_max=6, y_min=-3, y_max=3, z_min=-2, z_max=-0.7,
    #                       roll_min=-pi/7, roll_max=pi/7, pitch_min=-pi/7, pitch_max=pi/7, yaw_min=-pi/7 - pi/2, yaw_max=pi/7 - pi/2)
    # motion = ObjectMotion("auv4", "robosub_torpedo_bootlegger", "auv4/base_link", "robosub_torpedo_bootlegger", y_min=-4, y_max=-1, x_min=-3, x_max=3, z_min=0.7, z_max=2,
    #                       roll_min=-pi/9, roll_max=pi/9, pitch_min=-pi/9, pitch_max=pi/9, yaw_min=-pi/9 + pi/2, yaw_max=pi/9 + pi/2,
    #                       vyaw=pi/40, vroll=pi/40, vpitch=pi/40)
    # motion = ObjectMotion("auv4",
    #                       "robosub_torpedo_bootlegger",
    #                       "auv4/base_link",
    #                       "robosub_torpedo_bootlegger",
    #                       x_min=-0.5, x_max=0.5,
    #                       y_min=-1.5, y_max=-4,
    #                       z_min=0.7, z_max=2,
    #                       roll_min=-pi/9, roll_max=-pi/18,
    #                       pitch_min=-pi/18, pitch_max=-pi/18,
    #                       yaw_min=pi/2-pi/6, yaw_max=pi/2 + pi/6,
    #                       vyaw=pi/40, vroll=pi/40, vpitch=pi/40)

    # motion = ObjectMotion("auv4",
    #                     "robosub23_gate",
    #                     "auv4/base_link",
    #                     "robosub23_gate",
    #                     # x_min=-4, x_max=1,
    #                     # y_min=-4, y_max=-2,
    #                     # z_min=-2, z_max=1,
    #                     x_min = -1.7, y_min=-4, z_min=-0.5, x_max=1.7, y_max=-0.5, z_max=1.5,
    #                     roll_min=-pi/9, roll_max=pi/9,
    #                     pitch_min=-pi/18, pitch_max=pi/18,
    #                     yaw_min=-pi/6 + pi/4 + 0.846967, yaw_max=pi/6 + pi/4 + 0.846967,
    #                     vyaw=pi/40, vroll=pi/40, vpitch=pi/40)
    # motion = ObjectMotion("auv4",
    #                 "robosub23_buoy_v2",
    #                 "auv4/base_link",
    #                 "robosub23_buoy_v2",
    #                 x_min = -1.2, y_min=-7, z_min=-0.5, x_max=1.2, y_max=-0.8, z_max=1.5,
    #                 roll_min=-pi/9, roll_max=pi/9,
    #                 pitch_min=-pi/18, pitch_max=pi/18,
    #                 yaw_min=-pi/6 + pi/4 +0.846967, yaw_max=pi/6 + pi/4 +0.846967,
    #                 vyaw=pi/40, vroll=pi/40, vpitch=pi/40)

    motion = ObjectMotion("auv4",
                    "robosub23_torpedo_panel_v2",
                    "auv4/base_link",
                    "robosub23_torpedo_panel_v2",
                    x_min = -1.2, y_min=0.3, z_min=-0.1, x_max=1.2, y_max=1, z_max=1.5,
                    roll_min=-pi/9, roll_max=pi/9,
                    pitch_min=-pi/18, pitch_max=pi/18,
                    yaw_min=-pi/6 + pi/4 +pi+0.846967, yaw_max=pi/6 + pi/4  +pi+0.846967,
                    vyaw=pi/40, vroll=pi/40, vpitch=pi/40)
    # motion = ObjectMotion("auv4",
    #             "robosub23_torpedo_panel_v1",
    #             "auv4/base_link",
    #             "robosub23_torpedo_panel_v1",
    #             x_min = -1.2, y_min=-2, z_min=-0.5, x_max=1.2, y_max=-1, z_max=1.5,
    #             roll_min=-pi/9, roll_max=pi/9,
    #             pitch_min=-pi/18, pitch_max=pi/18,
    #             yaw_min=-pi/6 + pi/4+0.846967, yaw_max=pi/6 + pi/4+0.846967,
    #             vyaw=pi/40, vroll=pi/40, vpitch=pi/40)

    # motion = ObjectMotion("auv4",
    #                 "robosub23_octagon",
    #                 "auv4/base_link",
    #                 "robosub23_octagon",
    #                 x_min = -0.7, y_min=-0.7, z_min=-2.9, x_max=0.7, y_max=0.7, z_max=-1,
    #                 roll_min=-pi/9, roll_max=pi/9,
    #                 pitch_min=-pi/18, pitch_max=pi/18,
    #                 yaw_min=-pi, yaw_max=pi,
    #                 vyaw=pi/4, vroll=pi/40, vpitch=pi/40)

    # motion = ObjectMotion("auv4",
    #             "path_marker_1",
    #             "auv4/base_link",
    #             "path_marker_1",
    #             x_min = 0, y_min=-3, z_min=0, x_max=0, y_max=-3, z_max=1,
    #             roll_min=0, roll_max=0,
    #             pitch_min=pi/18, pitch_max=pi/18,
    #             yaw_min=pi/2, yaw_max=pi/2,
    #             vyaw=pi/40, vroll=pi/40, vpitch=pi/40)
    motion.start()
    rospy.spin()
