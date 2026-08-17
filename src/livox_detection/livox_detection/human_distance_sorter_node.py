#!/usr/bin/env python3
"""Node 1: Human Distance Sorter Node.

Subscribes to `/g1/detections/livox`, filters for human/pedestrian detections,
calculates their 3D distance from the robot pelvis frame, and sorts them
in increasing distance order (closest human first).

Publishes:
    - `/g1/sorted_humans` (vision_msgs/msg/Detection3DArray)
    - `/g1/sorted_human_markers` (visualization_msgs/msg/MarkerArray)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from vision_msgs.msg import Detection3D, Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray


class HumanDistanceSorterNode(Node):
    def __init__(self):
        super().__init__("g1_human_distance_sorter")

        # Declare parameters
        self.declare_parameter("input_topic", "/g1/detections/livox")
        self.declare_parameter("output_topic", "/g1/sorted_humans")
        self.declare_parameter("min_score", 0.30)
        self.declare_parameter("target_classes", ["pedestrian", "human"])

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.min_score = float(self.get_parameter("min_score").value)
        self.target_classes = [c.lower() for c in self.get_parameter("target_classes").value]

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.sub_detections = self.create_subscription(
            Detection3DArray, self.input_topic, self.on_detections, sensor_qos
        )

        self.pub_sorted = self.create_publisher(
            Detection3DArray, self.output_topic, 10
        )
        self.pub_markers = self.create_publisher(
            MarkerArray, "/g1/sorted_human_markers", 10
        )

        self.get_logger().info(
            f"HumanDistanceSorterNode started. Subscribing: {self.input_topic} -> Publishing: {self.output_topic}"
        )

    def on_detections(self, msg: Detection3DArray) -> None:
        """Filter for humans, calculate distance from pelvis, and sort."""
        humans_with_dist: List[Tuple[float, Detection3D]] = []

        for det in msg.detections:
            # Check hypothesis class
            is_human = False
            best_score = 0.0
            for hyp in det.results:
                cls_id = hyp.hypothesis.class_id.lower()
                score = hyp.hypothesis.score
                if cls_id in self.target_classes and score >= self.min_score:
                    is_human = True
                    best_score = max(best_score, score)

            if not is_human:
                continue

            pos = det.bbox.center.position
            dist = math.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
            humans_with_dist.append((dist, det))

        # Sort in increasing order of distance (closest human first)
        humans_with_dist.sort(key=lambda item: item[0])

        # Prepare sorted Detection3DArray
        sorted_array = Detection3DArray()
        sorted_array.header = msg.header
        for _, det in humans_with_dist:
            sorted_array.detections.append(det)

        self.pub_sorted.publish(sorted_array)

        # Publish sorted visualization markers
        markers = self._create_sorted_markers(msg.header, humans_with_dist)
        self.pub_markers.publish(markers)

    def _create_sorted_markers(
        self, header, humans_with_dist: List[Tuple[float, Detection3D]]
    ) -> MarkerArray:
        markers = MarkerArray()

        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for rank, (dist, det) in enumerate(humans_with_dist, start=1):
            pos = det.bbox.center.position
            size = det.bbox.size

            # 1. Floating numbered index + distance badge
            tag = Marker()
            tag.header = header
            tag.ns = "human_rank_tags"
            tag.id = rank
            tag.type = Marker.TEXT_VIEW_FACING
            tag.action = Marker.ADD
            tag.pose.position.x = pos.x
            tag.pose.position.y = pos.y
            tag.pose.position.z = pos.z + (size.z / 2.0 if size.z > 0 else 0.8) + 0.35
            tag.pose.orientation.w = 1.0
            tag.scale.z = 0.32
            tag.text = f"#{rank} [{dist:.2f} m]"
            # Highlight closest human in bright green, others in yellow/cyan
            if rank == 1:
                tag.color.r, tag.color.g, tag.color.b, tag.color.a = 0.0, 1.0, 0.0, 1.0
            else:
                tag.color.r, tag.color.g, tag.color.b, tag.color.a = 1.0, 0.85, 0.2, 1.0
            markers.markers.append(tag)

            # 2. Footprint ring / target cylinder on ground
            ring = Marker()
            ring.header = header
            ring.ns = "human_footprints"
            ring.id = rank + 100
            ring.type = Marker.CYLINDER
            ring.action = Marker.ADD
            ring.pose.position.x = pos.x
            ring.pose.position.y = pos.y
            ring.pose.position.z = pos.z - (size.z / 2.0 if size.z > 0 else 0.8)
            ring.pose.orientation.w = 1.0
            ring.scale.x = 0.6
            ring.scale.y = 0.6
            ring.scale.z = 0.05
            if rank == 1:
                ring.color.r, ring.color.g, ring.color.b, ring.color.a = 0.0, 1.0, 0.2, 0.7
            else:
                ring.color.r, ring.color.g, ring.color.b, ring.color.a = 1.0, 0.8, 0.0, 0.5
            markers.markers.append(ring)

        return markers


def main(args=None):
    rclpy.init(args=args)
    node = HumanDistanceSorterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
