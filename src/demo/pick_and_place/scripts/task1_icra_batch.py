#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-episode ROS1 collector for an EXTERNALLY managed Kuavo simulator.

Start the official data_challenge_simulator deploy.py in another terminal.
This script only serves /simulator/init, invokes /simulator/start and
/simulator/reset, runs an interchangeable one-episode controller command,
and optionally records an independent ROS bag per episode.

It does NOT launch roscore/MuJoCo, randomize scenes, or modify the controller.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse


TOYS = ("dino", "dog", "elephant", "shark", "sheep")
CARS = ("fire", "roller", "school", "boat", "fire_engine")
DEFAULT_TOPICS = (
    "/sensors_data_raw", "/kuavo_arm_traj", "/gripper/command", "/gripper/state",
    "/mujoco/basket1/pose", "/mujoco/basket2/pose",
) + tuple("/mujoco/%s/pose" % name for name in TOYS + CARS)
SUMMARY_FIELDS = (
    "episode", "outcome", "success", "controller_returncode", "timed_out",
    "duration_s", "bag_ok", "bag_bytes", "object_name", "category", "error",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
    temp.replace(path)


def command_from_string(value):
    args = shlex.split(value)
    if not args:
        raise ValueError("Command cannot be empty")
    return args


def stop_process(process, graceful_signal, timeout):
    """Terminate only the subprocess's own process group, never the simulator."""
    if process is None:
        return None
    if process.poll() is None:
        try:
            os.killpg(process.pid, graceful_signal)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    return process.returncode


def launch_logged(command, logfile, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    with logfile.open("w", encoding="utf-8") as log:
        return subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
        )


class SimulatorProtocol:
    """Generation counter prevents losing an init event during reset/restart."""

    def __init__(self):
        self.cv = threading.Condition()
        self.generation = 0
        self.success_evt = threading.Event()
        self.init_service = rospy.Service("/simulator/init", Trigger, self.on_init)
        self.success_sub = rospy.Subscriber(
            "/simulator/success", Bool, self.on_success, queue_size=10
        )

    def on_init(self, _request):
        with self.cv:
            self.generation += 1
            self.cv.notify_all()
        rospy.loginfo("Received /simulator/init generation=%d", self.generation)
        return TriggerResponse(success=True, message="Collector ready")

    def on_success(self, msg):
        if msg.data:
            self.success_evt.set()

    def wait_next_init(self, previous_generation, timeout):
        deadline = time.monotonic() + timeout
        with self.cv:
            while self.generation <= previous_generation and not rospy.is_shutdown():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for next /simulator/init")
                self.cv.wait(timeout=min(remaining, 0.5))
            if rospy.is_shutdown():
                raise RuntimeError("ROS shutdown while waiting for simulator")
            return self.generation

    @staticmethod
    def call_trigger(name, timeout):
        rospy.wait_for_service(name, timeout=timeout)
        result = rospy.ServiceProxy(name, Trigger)(TriggerRequest())
        if not result.success:
            raise RuntimeError("%s rejected request: %s" % (name, result.message))

    def close(self):
        self.success_sub.unregister()
        self.init_service.shutdown()


def scene_position(name, timeout=0.25):
    message = rospy.wait_for_message("/mujoco/%s/pose" % name, PoseStamped, timeout=timeout)
    if message.header.frame_id != "odom":
        raise RuntimeError("Unexpected frame for %s: %s" % (name, message.header.frame_id))
    p = message.pose.position
    return [float(p.x), float(p.y), float(p.z)]


def find_task1_left_object():
    """Optional task-specific evaluator. Mirror the existing V1 region checks."""
    basket1 = scene_position("basket1", timeout=2.0)
    expected_z = basket1[2] + 0.005
    found = []
    for name in TOYS + CARS:
        try:
            xyz = scene_position(name)
        except rospy.ROSException:
            continue
        if (0.22 <= xyz[0] <= 0.44 and 0.21 <= xyz[1] <= 0.46
                and abs(xyz[2] - expected_z) <= 0.08):
            found.append((name, xyz))
    if len(found) != 1:
        raise RuntimeError("Expected one left-side Task1 object, found %s" %
                           [name for name, _ in found])
    name, xyz = found[0]
    return {
        "object_name": name,
        "category": "toy" if name in TOYS else "car",
        "initial_object_odom": xyz,
        "basket1_odom": basket1,
    }


def check_task1_left_pose(scene):
    name = scene["object_name"]
    xyz = scene_position(name, timeout=2.0)
    basket_h = scene["basket1_odom"][2] - 0.005
    toy = scene["category"] == "toy"
    success = (
        0.31 <= xyz[0] <= 0.49
        and ((-0.15 <= xyz[1] <= 0.0) if toy else (0.0 <= xyz[1] <= 0.15))
        and basket_h <= xyz[2] <= basket_h + 0.12
    )
    return bool(success), {"final_object_odom": xyz, "basket_height": basket_h}


def stop_bag(process, bag_file, stop_timeout):
    if process is None:
        return {"bag_ok": False, "bag_bytes": 0, "bag_status": "disabled"}
    stop_process(process, signal.SIGINT, stop_timeout)
    size = bag_file.stat().st_size if bag_file.is_file() else 0
    active = Path(str(bag_file) + ".active")
    good = size > 1024 and not active.exists()
    return {
        "bag_ok": bool(good),
        "bag_bytes": size,
        "bag_status": "completed" if good else "missing_empty_or_unfinalized",
    }


def evaluate(args, context, protocol, episode_dir):
    mode = args.success_mode
    if mode == "exit_code":
        return context["controller_returncode"] == 0, "controller_exit_code"
    if mode == "task1_left_pose":
        passed, detail = check_task1_left_pose(context["scene"])
        context["evaluation"] = detail
        return passed, "task1_left_object_in_target_region"
    if mode == "ros_success":
        return protocol.success_evt.is_set(), "simulator_success_topic"
    if mode == "command":
        source = episode_dir / "evaluation_context.json"
        write_json(source, context)
        env = {"BATCH_EPISODE_JSON": str(source)}
        command = command_from_string(args.success_command)
        with (episode_dir / "evaluator.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT,
                timeout=args.evaluator_timeout, env={**os.environ, **env},
                check=False,
            )
        return proc.returncode == 0, "custom_evaluator_exit_code"
    raise ValueError("Unsupported success mode: %s" % mode)


def run_one(args, protocol, episode, controller_cmd, topics, output_dir):
    episode_dir = output_dir / ("episode_%06d" % episode)
    episode_dir.mkdir(parents=True, exist_ok=False)
    bag_file = episode_dir / "episode.bag"
    context = {
        "episode": episode,
        "started_at": utc_now(),
        "controller_command": controller_cmd,
        "success_mode": args.success_mode,
        "topics": list(topics) if args.record else [],
        "controller_returncode": None,
        "timed_out": False,
        "bag_ok": False,
        "bag_bytes": 0,
        "success": False,
        "outcome": "error",
        "error": "",
    }
    controller = None
    recorder = None
    start = time.monotonic()
    try:
        if args.success_mode == "task1_left_pose":
            context["scene"] = find_task1_left_object()
            context["object_name"] = context["scene"]["object_name"]
            context["category"] = context["scene"]["category"]
        if args.record:
            recorder = launch_logged(
                ["rosbag", "record", "-O", str(bag_file), *topics],
                episode_dir / "rosbag.log",
            )
            time.sleep(args.record_start_delay)
            if recorder.poll() is not None:
                raise RuntimeError("rosbag recorder exited during startup")
        protocol.success_evt.clear()
        protocol.call_trigger("/simulator/start", args.service_timeout)
        controller = launch_logged(
            controller_cmd, episode_dir / "controller.log",
            {"BATCH_EPISODE_DIR": str(episode_dir), "BATCH_EPISODE_INDEX": str(episode)},
        )
        try:
            context["controller_returncode"] = controller.wait(timeout=args.episode_timeout)
        except subprocess.TimeoutExpired:
            context["timed_out"] = True
            context["error"] = "Controller episode timeout"
    except Exception as error:
        context["error"] = str(error)
    finally:
        stop_process(controller, signal.SIGINT, args.shutdown_timeout)
        context.update(stop_bag(recorder, bag_file, args.shutdown_timeout))

    if not context["error"] and not context["timed_out"]:
        try:
            passed, reason = evaluate(args, context, protocol, episode_dir)
            context["success"] = bool(passed)
            context["evaluation_reason"] = reason
            context["outcome"] = "success" if passed else "failure"
        except Exception as error:
            context["error"] = "Evaluation failed: %s" % error
    if context["error"]:
        context["outcome"] = "timeout" if context["timed_out"] else "error"
    if args.record and not context["bag_ok"]:
        context["recording_warning"] = "Bag missing/unfinished; inspect rosbag.log"
    context["duration_s"] = round(time.monotonic() - start, 3)
    context["ended_at"] = utc_now()
    write_json(episode_dir / "result.json", context)
    return context


def append_summary(output_dir, context):
    target = output_dir / "summary.csv"
    with target.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow({key: context.get(key, "") for key in SUMMARY_FIELDS})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-episode ROS1 controller runner; official deploy.py runs separately."
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--controller-command", default=None,
                        help="One-episode process command (shell quoting supported, shell disabled). "
                             "Default: task1_icra_sdk.py --execute")
    parser.add_argument("--success-mode",
                        choices=("exit_code", "task1_left_pose", "ros_success", "command"),
                        default="exit_code")
    parser.add_argument("--success-command", default=None,
                        help="Evaluator command for mode=command. It receives BATCH_EPISODE_JSON, "
                             "exits 0 for success, nonzero for failure.")
    parser.add_argument("--evaluator-timeout", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task1_batch"))
    parser.add_argument("--no-record", dest="record", action="store_false", default=True)
    parser.add_argument("--topic", action="append", default=[],
                        help="Extra ROS topic to record (repeatable). For LeRobot include cameras.")
    parser.add_argument("--init-timeout", type=float, default=120.0)
    parser.add_argument("--service-timeout", type=float, default=20.0)
    parser.add_argument("--episode-timeout", type=float, default=60.0)
    parser.add_argument("--shutdown-timeout", type=float, default=10.0)
    parser.add_argument("--record-start-delay", type=float, default=0.6)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--abort-on-failure", action="store_true")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if any(value <= 0 for value in (
        args.init_timeout, args.service_timeout, args.episode_timeout,
        args.shutdown_timeout, args.evaluator_timeout,
    )) or args.record_start_delay < 0 or args.settle_seconds < 0:
        parser.error("Invalid timeout/delay")
    if args.success_mode == "command" and not args.success_command:
        parser.error("--success-command is required when --success-mode=command")
    if args.controller_command:
        args.controller = command_from_string(args.controller_command)
    else:
        args.controller = [
            sys.executable, str(Path(__file__).resolve().with_name("task1_icra_sdk.py")),
            "--execute",
        ]
    return args


def main():
    args = parse_args()
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output_dir = output_root / run_name
    output_dir.mkdir(exist_ok=False)
    topics = tuple(dict.fromkeys((*DEFAULT_TOPICS, *args.topic)))
    write_json(output_dir / "run_config.json", {
        "started_at": utc_now(), "episodes": args.episodes,
        "controller_command": args.controller,
        "success_mode": args.success_mode, "record": args.record,
        "topics": topics if args.record else [],
        "note": "Scene seeds and MuJoCo lifecycle are owned by official deploy.py.",
    })
    rospy.init_node("task1_icra_batch", anonymous=True, disable_signals=True)
    protocol = SimulatorProtocol()
    successes = 0
    seen_generation = 0
    completed = 0
    try:
        rospy.loginfo("Collector ready; start official deploy.py in another terminal.")
        if args.record and not args.topic:
            rospy.logwarn("No camera topics added. Use --topic for RGB streams before LeRobot conversion.")
        for episode in range(args.episodes):
            try:
                seen_generation = protocol.wait_next_init(seen_generation, args.init_timeout)
            except (TimeoutError, RuntimeError) as error:
                rospy.logerr("Could not initialize episode %d: %s", episode, error)
                break
            rospy.loginfo("Episode %d/%d, init generation %d",
                          episode + 1, args.episodes, seen_generation)
            time.sleep(args.settle_seconds)
            result = run_one(args, protocol, episode, args.controller, topics, output_dir)
            if result["success"]:
                successes += 1
            completed += 1
            # Rosbag and controller processes have exited before requesting reset.
            try:
                protocol.call_trigger("/simulator/reset", args.service_timeout)
                result["reset_sent"] = True
            except Exception as error:
                result["reset_sent"] = False
                result["reset_error"] = str(error)
                rospy.logerr("Simulator reset failed: %s", error)
            write_json(output_dir / ("episode_%06d" % episode) / "result.json", result)
            append_summary(output_dir, result)
            rospy.loginfo("Episode %d: %s, bag_ok=%s, %d/%d successes",
                          episode, result["outcome"], result["bag_ok"],
                          successes, completed)
            if not result["reset_sent"]:
                break
            if args.abort_on_failure and result["outcome"] != "success":
                break
    except KeyboardInterrupt:
        rospy.logwarn("Interrupted; any active episode process is cleaned up.")
    finally:
        protocol.close()
        write_json(output_dir / "run_summary.json", {
            "finished_at": utc_now(), "requested_episodes": args.episodes,
            "completed_episodes": completed, "successes": successes,
            "success_rate": successes / completed if completed else None,
        })
    print("Episodes: %d/%d, successes: %d, output: %s" %
          (completed, args.episodes, successes, output_dir))
    return 0 if completed == args.episodes else 1


if __name__ == "__main__":
    sys.exit(main())
