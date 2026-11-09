#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only FK analysis of LeRobot demonstrations for Kuavo ICRA Task 1.

Uses observation.state (measured joints), not action (commanded joints), for FK.
SDK arm_fk returns poses in the robot base frame. World/odom output requires
an explicitly supplied fixed base transform or an opt-in live SDK snapshot.
No robot arm/gripper control calls are made.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rospy
from scipy.spatial.transform import Rotation
from kuavo_humanoid_sdk import KuavoSDK, KuavoRobot, KuavoRobotState


def load_frames(dataset_root):
    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files under {dataset_root / 'data'}")
    frames = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    required = {"episode_index", "observation.state", "action"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"Missing LeRobot columns: {sorted(missing)}")
    order = ["episode_index"]
    if "frame_index" in frames:
        order.append("frame_index")
    elif "timestamp" in frames:
        order.append("timestamp")
    frames = frames.sort_values(order, kind="stable").reset_index(drop=True)
    return frames, len(files)


def stack_vectors(series, label, min_dim=16):
    try:
        result = np.stack([np.asarray(v, dtype=float).reshape(-1) for v in series])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Cannot parse {label} as fixed-size numeric vectors") from exc
    if result.ndim != 2 or result.shape[1] < min_dim or not np.isfinite(result).all():
        raise ValueError(f"Invalid {label} shape/values: {result.shape}; expected >= {min_dim}D")
    return result


def get_times(episode, fps):
    if "timestamp" in episode:
        ts = episode["timestamp"].to_numpy(dtype=float)
        if np.isfinite(ts).all() and np.all(np.diff(ts) >= 0) and ts[-1] > ts[0]:
            return ts - ts[0]
    if "frame_index" in episode:
        idx = episode["frame_index"].to_numpy(dtype=float)
        if np.isfinite(idx).all() and np.all(np.diff(idx) >= 0):
            return (idx - idx[0]) / fps
    return np.arange(len(episode), dtype=float) / fps


def detect_gripper_toggles(command, time_s, min_gap_s):
    low, high = np.percentile(command, [10, 90])
    if high - low < 1e-5:
        return []
    is_high = command >= 0.5 * (low + high)
    candidates = np.flatnonzero(is_high[1:] != is_high[:-1]) + 1
    selected = []
    for idx in candidates:
        if not selected or time_s[idx] - time_s[selected[-1]] >= min_gap_s:
            selected.append(int(idx))
    return selected


def get_base_transform(args):
    if args.world_from_sdk:
        state = KuavoRobotState()
        xyz = np.asarray(state.robot_position(), dtype=float)
        xyzw = np.asarray(state.robot_orientation(), dtype=float)
        source = "live_sdk_snapshot"
    elif args.base_xyz is not None:
        xyz = np.asarray(args.base_xyz, dtype=float)
        xyzw = np.asarray(args.base_xyzw, dtype=float)
        source = "manual_fixed_transform"
    else:
        return None, None, "unknown"
    if (xyz.shape != (3,) or xyzw.shape != (4,) or
            not np.isfinite(xyz).all() or not np.isfinite(xyzw).all() or
            np.linalg.norm(xyzw) < 1e-8):
        raise ValueError("Invalid base position or quaternion")
    return xyz, Rotation.from_quat(xyzw), source


def pose_values(pose):
    pos = np.asarray(pose.position, dtype=float)
    quat = np.asarray(pose.orientation, dtype=float)
    if (pos.shape != (3,) or quat.shape != (4,) or
            not np.isfinite(pos).all() or not np.isfinite(quat).all() or
            np.linalg.norm(quat) < 1e-8):
        raise RuntimeError("SDK FK returned invalid pose")
    return pos, Rotation.from_quat(quat).as_quat()


def build_fk_rows(robot, episode_id, episode, args, base_xyz, base_rot):
    states = stack_vectors(episode["observation.state"], "observation.state")
    actions = stack_vectors(episode["action"], "action")
    times = get_times(episode, args.fps)
    # Contract: [left arm 7, left gripper 1, right arm 7, right gripper 1].
    joints = states[:, list(range(7)) + list(range(8, 15))]
    joint_unit = args.joint_unit
    if joint_unit == "auto":
        joint_unit = "deg" if np.max(np.abs(joints)) > (2 * np.pi + 0.1) else "rad"
    joint_rad = np.deg2rad(joints) if joint_unit == "deg" else joints
    if np.max(np.abs(joint_rad)) > 2 * np.pi + 0.1:
        raise ValueError(f"Episode {episode_id}: suspicious joint values; verify --joint-unit")
    toggles = detect_gripper_toggles(actions[:, args.left_gripper_index],
                                      times, args.event_min_gap)
    if len(toggles) < 2:
        print(f"[WARN] Episode {episode_id}: only {len(toggles)} left gripper toggles; "
              "close/release may be unavailable")
    if len(toggles) > 2:
        print(f"[WARN] Episode {episode_id}: {len(toggles)} toggles; "
              "first two assumed to be close/release, verify plots")
    close_i = toggles[0] if len(toggles) >= 1 else None
    release_i = toggles[1] if len(toggles) >= 2 else None
    if release_i is not None and close_i is not None and release_i <= close_i:
        raise RuntimeError(f"Episode {episode_id}: invalid gripper event order")

    rows = []
    for i, q in enumerate(joint_rad):
        result = robot.arm_fk(q.tolist())
        if result is None or len(result) != 2 or result[0] is None or result[1] is None:
            raise RuntimeError(f"Episode {episode_id} frame {i}: SDK arm_fk failed")
        lp, lquat = pose_values(result[0])
        rp, rquat = pose_values(result[1])
        row = {"episode_index": episode_id, "frame": i,
               "time_s": float(times[i]),
               "left_gripper_action": float(actions[i, args.left_gripper_index]),
               "left_gripper_state": float(states[i, 7])}
        for j in range(7):
            row[f"left_joint_{j+1}_rad"] = float(q[j])
            row[f"right_joint_{j+1}_rad"] = float(q[j+7])
        for prefix, pos, quat in (("left", lp, lquat), ("right", rp, rquat)):
            row.update({f"{prefix}_{axis}_base": float(pos[k])
                        for k, axis in enumerate("xyz")})
            row.update({f"{prefix}_q{axis}_base": float(quat[k])
                        for k, axis in enumerate("xyzw")})
            if base_rot is not None:
                world_pos = base_rot.apply(pos) + base_xyz
                world_quat = (base_rot * Rotation.from_quat(quat)).as_quat()
                row.update({f"{prefix}_{axis}_odom": float(world_pos[k])
                            for k, axis in enumerate("xyz")})
                row.update({f"{prefix}_q{axis}_odom": float(world_quat[k])
                            for k, axis in enumerate("xyzw")})
        rows.append(row)
    trajectory = pd.DataFrame(rows)
    return trajectory, close_i, release_i, joint_unit, toggles


def keyframe_indices(traj, close_i, release_i, window_s):
    times = traj["time_s"].to_numpy()
    z = traj["left_z_base"].to_numpy()
    indices = {"start": 0}
    if close_i is not None:
        indices["close"] = close_i
    if release_i is not None:
        if close_i is not None and close_i < release_i:
            indices["lift_peak"] = int(close_i + np.argmax(z[close_i:release_i + 1]))
        near = np.flatnonzero((times >= times[release_i] - window_s) &
                              (times <= times[release_i]))
        if len(near):
            indices["pre_release_peak"] = int(near[np.argmax(z[near])])
        indices["release"] = release_i
    indices["end"] = len(traj) - 1
    return indices


def plot_episode(traj, keys, folder, base_known):
    frame = "odom" if base_known else "base"
    xyz = traj[[f"left_{a}_{frame}" for a in "xyz"]].to_numpy()
    times = traj["time_s"].to_numpy()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label="left EEF (feedback FK)")
    for label, i in keys.items():
        ax.scatter(*xyz[i], s=30)
        ax.text(*xyz[i], f" {label}", fontsize=8)
    ax.set(xlabel=f"X {frame} (m)", ylabel=f"Y {frame} (m)",
           zlabel=f"Z {frame} (m)", title="Left EEF trajectory (SDK FK)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "eef_trajectory_3d.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, xyz[:, 2], label=f"left EEF Z ({frame})")
    for label, i in keys.items():
        if label in ("close", "pre_release_peak", "release"):
            ax.axvline(times[i], linestyle="--", alpha=0.6, label=label)
    ax.set(xlabel="Time in episode (s)", ylabel="EEF Z (m)",
           title="Height vs time: look for descent before release")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "eef_height_timeline.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Read-only LeRobot -> Kuavo SDK FK trajectory analyzer")
    parser.add_argument("--dataset", required=True, type=Path, help="Unzipped LeRobot dataset root")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task1_lerobot_fk"))
    parser.add_argument("--joint-unit", choices=("auto", "rad", "deg"), default="auto")
    parser.add_argument("--fps", type=float, default=10.0, help="Fallback when no timestamps")
    parser.add_argument("--left-gripper-index", type=int, default=7)
    parser.add_argument("--event-min-gap", type=float, default=0.5)
    parser.add_argument("--release-window", type=float, default=2.0,
                        help="Seconds before gripper release for vertical-motion diagnostics")
    parser.add_argument("--world-from-sdk", action="store_true",
                        help="Use live robot base pose as FIXED transform for ALL demonstration frames")
    parser.add_argument("--base-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--base-xyzw", nargs=4, type=float, metavar=("QX", "QY", "QZ", "QW"))
    args = parser.parse_args(rospy.myargv()[1:])
    if args.world_from_sdk and (args.base_xyz is not None or args.base_xyzw is not None):
        parser.error("Use --world-from-sdk OR --base-xyz/--base-xyzw, not both")
    if (args.base_xyz is None) != (args.base_xyzw is None):
        parser.error("--base-xyz and --base-xyzw must be supplied together")
    if (args.fps <= 0 or args.event_min_gap <= 0 or args.release_window <= 0 or
            not 0 <= args.left_gripper_index < 16):
        parser.error("Invalid fps, timing or gripper index")
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        parser.error(f"LeRobot dataset directory not found: {dataset}")
    data, file_count = load_frames(dataset)
    print(f"Loaded {len(data)} frames, {data['episode_index'].nunique()} episodes "
          f"from {file_count} Parquet file(s)")

    rospy.init_node("task1_icra_lerobot_fk", anonymous=True, disable_signals=True)
    if not KuavoSDK().Init(options=KuavoSDK.Options.WithIK):
        raise RuntimeError("KuavoSDK WithIK initialization failed")
    robot = KuavoRobot()
    base_xyz, base_rot, base_source = get_base_transform(args)
    if base_rot is None:
        print("[WARN] No base transform: output is BASE-frame only; no valid odom positions")
    else:
        print(f"[WARN] Applying one fixed base transform ({base_source}) to all episodes. "
              "Verify the recorded base did not move and matches this pose.")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    keyframes, summaries = [], []
    for episode_id, ep in data.groupby("episode_index", sort=True):
        episode_id = int(episode_id)
        traj, close_i, release_i, unit, toggles = build_fk_rows(
            robot, episode_id, ep.reset_index(drop=True), args, base_xyz, base_rot)
        folder = output / f"episode_{episode_id:06d}"
        folder.mkdir(exist_ok=True)
        traj.to_csv(folder / "eef_trajectory.csv", index=False)
        keys = keyframe_indices(traj, close_i, release_i, args.release_window)
        for label, idx in keys.items():
            record = {"event": label, "joint_unit_source": unit,
                      **traj.iloc[idx].to_dict()}
            keyframes.append(record)
        plot_episode(traj, keys, folder, base_rot is not None)

        item = {"episode_index": episode_id, "frames": len(traj),
                "joint_unit_source": unit, "gripper_toggle_count": len(toggles),
                "close_time_s": float(traj.iloc[close_i]["time_s"]) if close_i is not None else None,
                "release_time_s": float(traj.iloc[release_i]["time_s"]) if release_i is not None else None}
        if release_i is not None and "pre_release_peak" in keys:
            peak_i = keys["pre_release_peak"]
            dz = float(traj.iloc[peak_i]["left_z_base"] -
                       traj.iloc[release_i]["left_z_base"])
            item["pre_release_peak_to_release_drop_m"] = dz
            item["pre_release_peak_time_s"] = float(traj.iloc[peak_i]["time_s"])
            # Diagnostic, NOT semantic proof of a dedicated pre-place stage.
            item["vertical_drop_over_3cm"] = bool(dz > 0.03 and
                traj.iloc[release_i]["time_s"] - traj.iloc[peak_i]["time_s"] >= 0.2)
            for axis in "xyz":
                item[f"release_{axis}_base"] = float(traj.iloc[release_i][f"left_{axis}_base"])
                if base_rot is not None:
                    item[f"release_{axis}_odom"] = float(traj.iloc[release_i][f"left_{axis}_odom"])
        summaries.append(item)
        print(f"Episode {episode_id:02d}: {len(traj)} frames; "
              f"close={item['close_time_s']}, release={item['release_time_s']}, "
              f"pre-release peak drop={item.get('pre_release_peak_to_release_drop_m')} m")

    pd.DataFrame(keyframes).to_csv(output / "keyframes.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    with (output / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump({"dataset": str(dataset), "frames": len(data),
                   "episodes": int(data["episode_index"].nunique()),
                   "fk_source": "observation.state", "fk_frame": "base",
                   "odom_source": base_source,
                   "fixed_base_xyz": base_xyz.tolist() if base_xyz is not None else None,
                   "fixed_base_xyzw": base_rot.as_quat().tolist() if base_rot is not None else None,
                   "event_detection": "first two left gripper action toggles, assumed close/release",
                   "pre_release_window_s": args.release_window,
                   "vertical_drop_note": "diagnostic only; not proof of pre-place stage"}, file, indent=2)
    print(f"Finished: {output}\n  summary.csv\n  keyframes.csv\n"
          "  episode_*/eef_trajectory.csv + 2 PNG plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
