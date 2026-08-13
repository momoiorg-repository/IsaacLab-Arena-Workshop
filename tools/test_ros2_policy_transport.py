#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Loopback test for the ROS2 remote-policy transport (no GPU, no Isaac Sim).

Runs a Ros2PolicyServer wrapping a tiny echo policy and a Ros2PolicyClient, and
exercises ping -> get_init_info handshake -> set_task_description -> get_action
with realistic numpy payloads (camera image + joint state). This validates the
ROS2 transport + msgpack numpy round-trip end-to-end before the heavy two-
container e2e.

Run inside the Isaac Sim container with the bundled ROS2 env sourced:

  source tools/setup_isaacsim_ros2_env.sh
  export ROS_DOMAIN_ID=99
  /isaac-sim/python.sh tools/test_ros2_policy_transport.py            # role=both

It spawns the server as a child process and drives the client, then kills the
server. Exit code 0 means success.
"""

from __future__ import annotations

import argparse
import numpy as np
import os
import subprocess
import sys
from typing import Any

from isaaclab_arena.remote_policy.action_protocol import ChunkingActionProtocol
from isaaclab_arena.remote_policy.remote_policy_config import Ros2RemotePolicyConfig
from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy

ACTION_MARKER = 0.123
NUM_ENVS = 1


class EchoServerSidePolicy(ServerSidePolicy):
    """Minimal server-side policy: returns a constant action chunk, echoes obs shapes."""

    def __init__(self, action_dim: int, chunk_len: int, horizon: int) -> None:
        super().__init__(config=None)
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.horizon = horizon
        self.required_observation_keys = ["camera_obs.wrist_cam_rgb", "policy.robot_joint_pos"]

    def _build_protocol(self) -> ChunkingActionProtocol:
        return ChunkingActionProtocol(
            action_dim=self.action_dim,
            observation_keys=self.required_observation_keys,
            action_chunk_length=self.chunk_len,
            action_horizon=self.horizon,
        )

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # observation is the packed dict that the client sent (numpy arrays).
        shapes = {
            k: list(np.asarray(v).shape)
            for k, v in observation.items()
            if hasattr(v, "shape") or isinstance(v, np.ndarray)
        }
        action = np.full((NUM_ENVS, self.horizon, self.action_dim), ACTION_MARKER, dtype=np.float32)
        return {"action": action}, {"echo_obs_shapes": shapes}

    def set_task_description(self, task_description: str | None) -> dict[str, Any]:
        self._task_description = task_description or ""
        return {"status": "ok", "echo": self._task_description}

    def reset(self, env_ids: Any = None, reset_options: dict[str, Any] | None = None) -> dict[str, Any]:
        # Mirror Gr00tRemoteServerSidePolicy.reset signature (PolicyServer._handle_reset
        # calls reset(env_ids=..., reset_options=...)).
        return {"status": "reset_success"}

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> EchoServerSidePolicy:
        return EchoServerSidePolicy(action_dim=args.action_dim, chunk_len=args.chunk_len, horizon=args.horizon)


def run_server(args: argparse.Namespace) -> int:
    from isaaclab_arena.remote_policy.ros2_transport import Ros2PolicyServer

    policy = EchoServerSidePolicy(action_dim=args.action_dim, chunk_len=args.chunk_len, horizon=args.horizon)
    server = Ros2PolicyServer(policy=policy, namespace=args.namespace, domain_id=args.domain_id)
    print("[test-server] running")
    server.run()
    print("[test-server] stopped")
    return 0


def run_client(args: argparse.Namespace) -> int:
    from isaaclab_arena.remote_policy.ros2_transport import Ros2PolicyClient

    cfg = Ros2RemotePolicyConfig(namespace=args.namespace, domain_id=args.domain_id, timeout_ms=10000)
    client = Ros2PolicyClient(config=cfg)

    ok = True

    print("[test-client] ping...")
    assert client.ping(discovery_timeout_ms=60000), "ping failed: server not reachable"
    print("[test-client] ping OK")

    print("[test-client] get_init_info handshake...")
    init = client.get_init_info(ChunkingActionProtocol.MODE.value)
    assert init.get("status") == "success", f"handshake failed: {init}"
    cfg_dict = init["config"]
    assert cfg_dict["action_dim"] == args.action_dim, cfg_dict
    assert cfg_dict["action_chunk_length"] == args.chunk_len, cfg_dict
    proto = ChunkingActionProtocol.from_dict(cfg_dict)
    print(f"[test-client] handshake OK: {proto}")

    print("[test-client] set_task_description...")
    sd = client.set_task_description("pick up the cube")
    assert sd.get("status") == "ok" and sd.get("echo") == "pick up the cube", sd
    print("[test-client] set_task_description OK")

    print("[test-client] get_action with numpy obs...")
    obs = {
        "camera_obs.wrist_cam_rgb": np.zeros((NUM_ENVS, 256, 256, 3), dtype=np.uint8),
        "policy.robot_joint_pos": np.arange(NUM_ENVS * 9, dtype=np.float32).reshape(NUM_ENVS, 9),
    }
    resp = client.get_action(obs)
    assert "action" in resp, f"no action in response: {list(resp.keys())}"
    action = np.asarray(resp["action"])
    assert action.shape == (NUM_ENVS, args.horizon, args.action_dim), action.shape
    assert np.allclose(action, ACTION_MARKER), "action values did not round-trip"
    print(f"[test-client] get_action OK: action.shape={action.shape}, echo={resp.get('echo_obs_shapes')}")

    print("[test-client] reset...")
    client.reset(env_ids=[0])
    print("[test-client] reset OK")

    print("[test-client] kill server + close...")
    client.kill()
    client.close()

    print("\n[test-client] ALL CHECKS PASSED" if ok else "[test-client] FAILED")
    return 0 if ok else 1


def run_both(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    if args.domain_id is not None:
        env["ROS_DOMAIN_ID"] = str(args.domain_id)

    # Use Isaac Sim's python.sh for the child so msgpack/numpy/isaaclab_arena and the
    # bundled rclpy all resolve exactly as for the parent (sys.executable is the bare
    # interpreter, which may not see Isaac Sim's site-packages).
    launcher = os.environ.get("ROS2_TEST_PYTHON")
    if launcher:
        py_cmd = launcher.split()
    elif os.path.exists("/isaac-sim/python.sh"):
        py_cmd = ["/isaac-sim/python.sh"]
    else:
        py_cmd = [sys.executable]

    server_cmd = [
        *py_cmd,
        os.path.abspath(__file__),
        "--role",
        "server",
        "--namespace",
        args.namespace,
        "--action-dim",
        str(args.action_dim),
        "--chunk-len",
        str(args.chunk_len),
        "--horizon",
        str(args.horizon),
    ]
    if args.domain_id is not None:
        server_cmd += ["--domain-id", str(args.domain_id)]

    print(f"[test-both] spawning server: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(server_cmd, env=env)
    try:
        rc = run_client(args)
    finally:
        try:
            server_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            print("[test-both] server did not exit after kill; killing")
            server_proc.kill()
            server_proc.wait(timeout=10)
    server_rc = server_proc.returncode
    print(f"[test-both] client_rc={rc}, server_rc={server_rc}")
    return rc if rc != 0 else (0 if server_rc in (0, None) else 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=["both", "server", "client"], default="both")
    parser.add_argument("--namespace", default="/gr00t_policy_test")
    parser.add_argument("--domain-id", dest="domain_id", type=int, default=None)
    parser.add_argument("--action-dim", dest="action_dim", type=int, default=8)
    parser.add_argument("--chunk-len", dest="chunk_len", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()

    if args.role == "server":
        return run_server(args)
    if args.role == "client":
        return run_client(args)
    return run_both(args)


if __name__ == "__main__":
    sys.exit(main())
