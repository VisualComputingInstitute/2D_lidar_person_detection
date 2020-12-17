#!/usr/bin/env python

import rospy
from dr_spaam_ros.dr_spaam_ros import DrSpaamROS


if __name__ == '__main__':
    rospy.init_node('dr_spaam_ros')
    try:
        DrSpaamROS()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()
