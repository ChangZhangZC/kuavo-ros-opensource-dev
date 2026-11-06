#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ICRA Eval1, left arm only. Dry-run by default; --execute enables MuJoCo motion.

Uses native Kuavo SDK FK/IK, Utils joint interpolation, TrajectoryController,
and GripperController. Does not modify the evaluator or the original demo.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation
from kuavo_humanoid_sdk import KuavoSDK, KuavoRobot, KuavoRobotState, KuavoPose
from kuavo_humanoid_sdk.interfaces.data_types import KuavoIKParams
from kuavo_msgs.msg import sensorsData

SIM_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data_challenge_simulator"
))
if SIM_ROOT not in sys.path:
    sys.path.insert(0, SIM_ROOT)
from utils.gripper_controller import GripperController
from utils.trajectory_controller import TrajectoryController
from utils.utils import Utils

TOYS = ("dino", "dog", "elephant", "shark", "sheep")
CARS = ("fire", "roller", "school", "boat", "fire_engine")
LEFT_X = (0.23, 0.43)
LEFT_Y = (0.22, 0.45)


class GatedRobot:
    """Prevent native TrajectoryController from publishing its initial zero target."""

    def __init__(self, robot):
        self.robot = robot
        self.enabled = False

    def control_arm_joint_positions(self, q):
        if not self.enabled:
            return False
        return self.robot.control_arm_joint_positions(q)


class Task1Left:
    def __init__(self, args):
        self.args = args
        self.robot = KuavoRobot()
        self.state = KuavoRobotState()
        self.trajectory = None
        self.gripper = None
        rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=5.0)
        rospy.sleep(0.2)
        self.initial_q = self.read_arm_q()
        rospy.loginfo("Measured Eval1 start left joints (deg): %s",
                      np.round(np.degrees(self.initial_q[:7]), 2).tolist())
        self.right_q = list(self.initial_q[7:14])
        self.base_xyz = np.asarray(self.state.robot_position(), dtype=float)
        base_xyzw = np.asarray(self.state.robot_orientation(), dtype=float)
        self.require_finite(self.base_xyz, 3, "robot base position")
        self.require_finite(base_xyzw, 4, "robot base orientation")
        self.base_rot = Rotation.from_quat(base_xyzw)

        pair = self.robot.arm_fk(self.initial_q)
        if pair is None or pair[0] is None or pair[1] is None:
            raise RuntimeError("Initial SDK FK failed")
        self.right_pose = pair[1]
        if args.grasp_base_rpy is None:
            grasp_base_rot = Rotation.from_quat(pair[0].orientation)
            rospy.logwarn("Grasp orientation defaults to the current left FK orientation; calibrate it before grasping")
        else:
            grasp_base_rot = Rotation.from_euler("xyz", args.grasp_base_rpy, degrees=True)
        self.grasp_world_xyzw = (self.base_rot * grasp_base_rot).as_quat()
        self.ik_params = KuavoIKParams(
            major_optimality_tol=1e-3, major_feasibility_tol=1e-3,
            minor_feasibility_tol=1e-3, major_iterations_limit=100,
            oritation_constraint_tol=1e-3, pos_constraint_tol=1e-3,
            pos_cost_weight=0.0
        )

    @staticmethod
    def require_finite(values, count, label):
        if len(values) != count or not np.all(np.isfinite(values)):
            raise RuntimeError("Invalid %s: %s" % (label, values))

    def read_arm_q(self):
        q = list(self.state.arm_joint_state().position)
        self.require_finite(q, 14, "arm joint feedback")
        return q

    @staticmethod
    def read_scene_pose(name, timeout=0.4):
        msg = rospy.wait_for_message("/mujoco/%s/pose" % name, PoseStamped, timeout=timeout)
        if msg.header.frame_id != "odom":
            raise RuntimeError("%s must have odom frame, got %s" % (name, msg.header.frame_id))
        xyz = np.asarray([msg.pose.position.x, msg.pose.position.y,
                          msg.pose.position.z], dtype=float)
        Task1Left.require_finite(xyz, 3, "%s position" % name)
        return xyz

    def select_left_object(self):
        basket1 = self.read_scene_pose("basket1", timeout=2.0)
        expected_z = basket1[2] + 0.005
        active = []
        for name in TOYS + CARS:
            try:
                position = self.read_scene_pose(name)
            except rospy.ROSException:
                continue
            if (LEFT_X[0] - 0.01 <= position[0] <= LEFT_X[1] + 0.01
                    and LEFT_Y[0] - 0.01 <= position[1] <= LEFT_Y[1] + 0.01
                    and abs(position[2] - expected_z) <= 0.08):
                active.append((name, position))
        if len(active) != 1:
            raise RuntimeError("Expected one left object; found: %s" %
                               [name for name, _ in active])
        name, object_xyz = active[0]
        category = "toy" if name in TOYS else "car"
        basket_name = "basket2" if category == "toy" else "basket1"
        basket_xyz = self.read_scene_pose(basket_name, timeout=2.0)
        rospy.loginfo("Eval1 V1: left object=%s (%s), destination=%s",
                      name, category, basket_name)
        rospy.loginfo("Object odom xyz=%s; basket odom xyz=%s",
                      object_xyz.round(4), basket_xyz.round(4))
        return name, category, object_xyz, basket_name, basket_xyz

    def target_in_base(self, world_xyz):
        base_xyz = self.base_rot.inv().apply(np.asarray(world_xyz) - self.base_xyz)
        base_quat = (self.base_rot.inv()
                     * Rotation.from_quat(self.grasp_world_xyzw)).as_quat()
        return KuavoPose(position=base_xyz.tolist(), orientation=base_quat.tolist())

    def solve_key_pose(self, label, world_xyz, seed_q):
        left_target = self.target_in_base(world_xyz)
        result = self.robot.arm_ik(
            left_target, self.right_pose,
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            list(seed_q), self.ik_params
        )
        if result is None or len(result) != 14:
            raise RuntimeError("%s: SDK IK failed" % label)
        self.require_finite(result, 14, "%s IK result" % label)

        # Match the real command: no IK-produced changes to the inactive arm.
        goal_q = list(result[:7]) + self.right_q
        if any(abs(v) > math.pi for v in goal_q):
            raise RuntimeError("%s: IK joint angle outside SDK [-pi, pi] range" % label)
        fk = self.robot.arm_fk(goal_q)
        if fk is None or fk[0] is None or fk[1] is None:
            raise RuntimeError("%s: SDK FK failed" % label)
        left_error = float(np.linalg.norm(
            np.asarray(fk[0].position) - np.asarray(left_target.position)
        ))
        right_error = float(np.linalg.norm(
            np.asarray(fk[1].position) - np.asarray(self.right_pose.position)
        ))
        orientation_error = (Rotation.from_quat(fk[0].orientation).inv()
                             * Rotation.from_quat(left_target.orientation)).magnitude()
        joint_delta = max(abs(a - b) for a, b in zip(goal_q[:7], seed_q[:7]))
        rospy.loginfo(
            "%s IK: left_pos_err=%.4fm, right_pos_err=%.4fm, "
            "left_ori_err=%.2fdeg, max_left_joint_delta=%.3frad",
            label, left_error, right_error, math.degrees(orientation_error), joint_delta
        )
        if (left_error > self.args.max_fk_error or
                right_error > self.args.max_fk_error or
                orientation_error > math.radians(self.args.max_orientation_error_deg) or
                joint_delta > self.args.max_segment_delta):
            raise RuntimeError("%s: IK rejected (FK residual, orientation or joint jump)" % label)
        return goal_q

    def plan(self, object_xyz, basket_xyz):
        grasp = object_xyz + np.asarray(self.args.grasp_offset)
        pregrasp = grasp + np.array([0.0, 0.0, self.args.pregrasp_height])
        lift = grasp + np.array([0.0, 0.0, self.args.lift_height])
        place_xy = basket_xyz + np.array([0.0, self.args.place_y_offset, 0.0])
        preplace = place_xy + np.array([0.0, 0.0, self.args.preplace_height])
        place = place_xy + np.array([0.0, 0.0, self.args.place_height])
        stages = [
            ("pregrasp", pregrasp), ("grasp", grasp), ("lift", lift),
            ("return_ready", None),
            ("preplace", preplace), ("place", place), ("retract", preplace),
        ]
        targets = []
        seed = self.initial_q
        for label, point in stages:
            if label == "return_ready":
                # Eval1 brings the robot to q1_target2 before /simulator/init.
                # Return only the left arm to its measured start joint pose.
                # The right arm remains fixed and the gripper stays closed.
                ready_q = list(self.initial_q)
                delta = max(abs(a - b) for a, b in zip(ready_q[:7], seed[:7]))
                rospy.loginfo("return_ready: max left joint delta=%.3frad", delta)
                if delta > self.args.max_segment_delta:
                    raise RuntimeError("return_ready: joint jump exceeds limit")
                seed = ready_q
            else:
                rospy.loginfo("%s world xyz=%s", label, point.round(4))
                seed = self.solve_key_pose(label, point, seed)
            targets.append((label, list(seed)))
        rospy.loginfo("IK-only plan passed: %d IK poses plus return_ready joint waypoint; right arm held",
                      len(targets) - 1)
        return targets

    def init_controllers(self):
        # The environment/arm state must not have changed since IK-only planning.
        live_q = self.read_arm_q()
        max_drift = max(abs(a - b) for a, b in zip(live_q, self.initial_q))
        live_base = np.asarray(self.state.robot_position(), dtype=float)
        live_base_xyzw = np.asarray(self.state.robot_orientation(), dtype=float)
        self.require_finite(live_base, 3, "live robot position")
        self.require_finite(live_base_xyzw, 4, "live robot orientation")
        base_move = float(np.linalg.norm(live_base - self.base_xyz))
        base_turn = (self.base_rot.inv() * Rotation.from_quat(live_base_xyzw)).magnitude()
        if max_drift > 0.08 or base_move > 0.02 or base_turn > 0.05:
            raise RuntimeError(
                "Robot moved since planning: arm=%.3frad, base=%.3fm, yaw/tilt=%.3frad; re-plan" %
                (max_drift, base_move, base_turn)
            )
        # Publish only in execute mode, after explicit user confirmation.
        if not self.robot.set_external_control_arm_mode():
            raise RuntimeError("Failed to switch to external arm control mode")
        gate = GatedRobot(self.robot)
        self.trajectory = TrajectoryController(gate, publish_frequency=100.0)
        if not self.trajectory.set_target_positions(self.initial_q):
            raise RuntimeError("Cannot set initial trajectory target")
        gate.enabled = True


    def start_gripper(self):
        rospy.wait_for_message("/gripper/state", JointState, timeout=3.0)
        self.gripper = GripperController()
        self.gripper.set_gripper_position(0.0, 0.0)
        for _ in range(30):
            if self.gripper.gripper_cmd_pub.get_num_connections():
                break
            rospy.sleep(0.1)
        else:
            raise RuntimeError("No /gripper/command subscriber connected")
        rospy.sleep(0.5)

    def follow_segment(self, label, goal, q_start):
        degrees_start = [math.degrees(x) for x in q_start]
        degrees_goal = [math.degrees(x) for x in goal]
        frames = Utils.interpolate_joint_trajectory(
            degrees_goal, degrees_start, num=self.args.interpolation_points
        )
        frames.append(list(goal))  # Native helper excludes the exact end point.
        self.trajectory.execute_trajectory(frames, sleep_time=self.args.step_dt)
        deadline = time.monotonic() + self.args.tracking_timeout
        max_error = float("inf")
        while time.monotonic() < deadline and not rospy.is_shutdown():
            actual = self.read_arm_q()
            max_error = max(abs(a - b) for a, b in zip(actual[:7], goal[:7]))
            if max_error <= self.args.max_tracking_error:
                break
            rospy.sleep(0.05)
        rospy.loginfo("%s: measured max left joint tracking error=%.3frad",
                      label, max_error)
        if max_error > self.args.max_tracking_error:
            raise RuntimeError("%s: left arm did not track target" % label)
        right_actual = self.read_arm_q()[7:14]
        right_err = max(abs(a - b) for a, b in zip(right_actual, self.right_q))
        if right_err > self.args.max_tracking_error:
            raise RuntimeError("%s: right arm moved beyond hold tolerance (%.3frad)" %
                               (label, right_err))

    def execute(self, targets, name, category, basket_xyz):
        print("Simulation motion requested. Type EXECUTE to enable arm and gripper commands:")
        if input().strip() != "EXECUTE":
            rospy.logwarn("No confirmation; aborting without motion")
            return False
        self.init_controllers()
        self.start_gripper()
        previous = self.initial_q
        for label, goal in targets:
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown during execution")
            rospy.loginfo("Executing stage: %s", label)
            self.follow_segment(label, goal, previous)
            previous = goal
            if label == "grasp":
                self.gripper.control_left_gripper(255.0)
                rospy.sleep(1.0)
                rospy.loginfo("Left gripper feedback: %s", self.gripper.get_current_positions())
            elif label == "place":
                self.gripper.control_left_gripper(0.0)
                rospy.sleep(1.0)
        obj_after = self.read_scene_pose(name, timeout=2.0)
        basket_h = basket_xyz[2] - 0.005
        x_ok = 0.31 <= obj_after[0] <= 0.49
        y_ok = (-0.15 <= obj_after[1] <= 0.0) if category == "toy" else (
            0.0 <= obj_after[1] <= 0.15)
        z_ok = basket_h <= obj_after[2] <= basket_h + 0.12
        ok = x_ok and y_ok and z_ok
        rospy.loginfo("V1 object=%s final world position=%s; Eval1 category region=%s",
                      name, obj_after.round(4), "PASS" if ok else "FAIL")
        return ok

    def stop(self):
        if self.trajectory is not None:
            self.trajectory.stop()
        if self.gripper is not None:
            self.gripper.stop()


def main():
    parser = argparse.ArgumentParser(description="ICRA Eval1 left-arm SDK baseline")
    parser.add_argument("--execute", action="store_true",
                        help="Enable MuJoCo motion after explicit console confirmation")
    parser.add_argument("--grasp-offset", nargs=3, type=float,
                        default=[0.0, 0.0, 0.05], metavar=("X", "Y", "Z"))
    parser.add_argument("--grasp-base-rpy", nargs=3, type=float, default=None,
                        metavar=("ROLL", "PITCH", "YAW"),
                        help="Optional left TCP orientation in base_link, degrees")
    parser.add_argument("--pregrasp-height", type=float, default=0.12)
    parser.add_argument("--lift-height", type=float, default=0.18)
    parser.add_argument("--preplace-height", type=float, default=0.18)
    parser.add_argument("--place-height", type=float, default=0.10)
    parser.add_argument("--place-y-offset", type=float, default=0.0,
                        help="World-frame Y shift of place/preplace targets, in meters")
    parser.add_argument("--max-fk-error", type=float, default=0.02)
    parser.add_argument("--max-orientation-error-deg", type=float, default=15.0)
    parser.add_argument("--max-segment-delta", type=float, default=2.0)
    parser.add_argument("--max-tracking-error", type=float, default=0.15)
    parser.add_argument("--tracking-timeout", type=float, default=4.0)
    parser.add_argument("--interpolation-points", type=int, default=60)
    parser.add_argument("--step-dt", type=float, default=0.02)
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    safety_values = (
        args.grasp_offset + (args.grasp_base_rpy or []) +
        [args.pregrasp_height, args.lift_height, args.preplace_height,
         args.place_height, args.place_y_offset,
         args.max_fk_error, args.max_orientation_error_deg,
         args.max_segment_delta, args.max_tracking_error,
         args.tracking_timeout, args.step_dt]
    )
    if not all(math.isfinite(x) for x in safety_values):
        parser.error("Trajectory and IK parameters must be finite")
    if not 0 < args.max_orientation_error_deg <= 90:
        parser.error("Invalid orientation acceptance threshold")
    if (args.interpolation_points < 2 or args.step_dt <= 0.0 or
            args.pregrasp_height <= 0 or args.lift_height <= 0 or
            args.preplace_height <= args.place_height or args.place_height <= 0 or
            args.max_fk_error <= 0 or args.max_tracking_error <= 0 or
            args.tracking_timeout <= 0 or args.max_segment_delta <= 0):
        parser.error("Invalid trajectory or safety parameter")
    rospy.init_node("task1_icra_sdk_left")
    task = None
    try:
        if not KuavoSDK().Init(options=KuavoSDK.Options.WithIK):
            raise RuntimeError("KuavoSDK WithIK initialization failed")
        task = Task1Left(args)
        name, category, obj, _, basket = task.select_left_object()
        targets = task.plan(obj, basket)
        if not args.execute:
            rospy.loginfo("DRY RUN ONLY: no arm/gripper control commands have been published")
            return 0
        result = task.execute(targets, name, category, basket)
        return 0 if result else 2
    except (RuntimeError, rospy.ROSException, rospy.ROSInterruptException) as exc:
        rospy.logerr("Task1 V1 aborted: %s", exc)
        return 1
    finally:
        if task is not None:
            task.stop()


if __name__ == "__main__":
    sys.exit(main())
