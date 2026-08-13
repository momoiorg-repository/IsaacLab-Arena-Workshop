# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ROS2 transport for the remote-policy framework.

This is a drop-in alternative to the ZeroMQ :class:`PolicyClient` /
:class:`PolicyServer` transport. It carries the *exact same* msgpack payloads
(via :class:`MessageSerializer`) over a topic-based request/reply pair, so all
policy logic, action chunking, the handshake protocol, and numpy serialization
are reused unchanged. Only the wire transport differs.

Wire format (both directions use ``std_msgs/UInt8MultiArray``):

* request  -> ``<namespace>/request``:  msgpack of
  ``{"request_id": int, "endpoint": str, "data": {...}, "api_token": str?}``
* response -> ``<namespace>/response``: msgpack of
  ``{"request_id": int, "payload": <handler result dict>}``

The client keeps a single request in flight and blocks (spinning) until the
response with the matching ``request_id`` arrives or the timeout elapses, which
makes the synchronous request/reply semantics identical to ZeroMQ REQ/REP.

Sim-side note: inside the Isaac Sim container, ``rclpy`` is provided by the
bundled ROS2 Humble (built for Isaac Sim's Python). Set, before launching::

    export ROS_DISTRO=humble
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.bridge/humble/rclpy:$PYTHONPATH

(see ``tools/setup_isaacsim_ros2_env.sh``).
"""

from __future__ import annotations

import os
import time
import warnings
from array import array
from typing import Any

from isaaclab_arena.remote_policy.message_serializer import MessageSerializer
from isaaclab_arena.remote_policy.policy_server import PolicyServer
from isaaclab_arena.remote_policy.remote_policy_config import Ros2RemotePolicyConfig
from isaaclab_arena.remote_policy.server_side_policy import ServerSidePolicy

# --------------------------------------------------------------------------- #
# rclpy import + small helpers
# --------------------------------------------------------------------------- #


def _import_rclpy():
    """Import rclpy and the message/QoS types, with a helpful error if missing.

    Returns:
        Tuple ``(rclpy, UInt8MultiArray, qos_module)``.
    """
    try:
        import rclpy
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import UInt8MultiArray
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Failed to import rclpy / std_msgs for the ROS2 policy transport. Inside the Isaac Sim "
            "container, source tools/setup_isaacsim_ros2_env.sh first (it points PYTHONPATH/"
            "LD_LIBRARY_PATH at Isaac Sim's bundled ROS2 Humble). Original error: "
            f"{exc}"
        ) from exc

    return rclpy, UInt8MultiArray, (QoSProfile, HistoryPolicy, ReliabilityPolicy)


def _ensure_rclpy_init(rclpy, domain_id: int | None) -> bool:
    """Initialize rclpy if needed. Returns True if this call performed the init."""
    if rclpy.ok():
        return False
    if domain_id is not None:
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    rclpy.init()
    return True


def _reliable_qos(qos_types, depth: int):
    QoSProfile, HistoryPolicy, ReliabilityPolicy = qos_types
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _encode_msg(UInt8MultiArray, raw: bytes):
    msg = UInt8MultiArray()
    # array('B', ...) is accepted efficiently by rclpy uint8[] fields.
    msg.data = array("B", raw)
    return msg


def _decode_msg(msg) -> bytes:
    return bytes(msg.data)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class Ros2PolicyClient:
    """Synchronous client for talking to a :class:`Ros2PolicyServer` over ROS2.

    The public API mirrors :class:`isaaclab_arena.remote_policy.policy_client.PolicyClient`
    exactly, so it is a drop-in replacement on the client side.
    """

    def __init__(self, config: Ros2RemotePolicyConfig) -> None:
        self._config = config
        self._rclpy, self._UInt8MultiArray, qos_types = _import_rclpy()
        self._owns_rclpy = _ensure_rclpy_init(self._rclpy, config.domain_id)

        self._node = self._rclpy.create_node("gr00t_policy_client")
        qos = _reliable_qos(qos_types, config.qos_depth)
        self._pub = self._node.create_publisher(self._UInt8MultiArray, config.request_topic, qos)
        self._sub = self._node.create_subscription(self._UInt8MultiArray, config.response_topic, self._on_response, qos)

        self._responses: dict[int, dict[str, Any]] = {}
        self._next_request_id = 0
        print(
            f"[Ros2PolicyClient] node up: request_topic={config.request_topic}, "
            f"response_topic={config.response_topic}, domain_id={os.environ.get('ROS_DOMAIN_ID')}"
        )

    # ------------------------------------------------------------------ #
    # ROS2 plumbing
    # ------------------------------------------------------------------ #

    def _on_response(self, msg) -> None:
        try:
            env = MessageSerializer.from_bytes(_decode_msg(msg))
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"[Ros2PolicyClient] failed to decode response: {exc}")
            return
        if isinstance(env, dict) and "request_id" in env:
            self._responses[int(env["request_id"])] = env

    def _spin_for_response(self, request_id: int, timeout_ms: int) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            if request_id in self._responses:
                return self._responses.pop(request_id)
            remaining = max(0.0, deadline - time.monotonic())
            self._rclpy.spin_once(self._node, timeout_sec=min(0.05, remaining))
        raise TimeoutError(
            f"[Ros2PolicyClient] timed out after {timeout_ms} ms waiting for response to request_id={request_id}"
        )

    # ------------------------------------------------------------------ #
    # Public API (mirrors PolicyClient)
    # ------------------------------------------------------------------ #

    def ping(self, discovery_timeout_ms: int = 60000, retry_interval_ms: int = 500) -> bool:
        """Check if the server is reachable, retrying to absorb DDS discovery latency."""
        deadline = time.monotonic() + discovery_timeout_ms / 1000.0
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                self.call_endpoint("ping", requires_input=False, timeout_ms=retry_interval_ms)
                return True
            except Exception:
                # Keep retrying; the server may not have been discovered yet.
                if attempt % 10 == 0:
                    print(f"[Ros2PolicyClient] still waiting for server (attempt {attempt})...")
                continue
        warnings.warn(
            f"[Ros2PolicyClient] failed to reach server on {self._config.request_topic} "
            f"within {discovery_timeout_ms} ms"
        )
        return False

    def reset(self, env_ids=None, options: dict[str, Any] | None = None) -> Any:
        resp = self.call_endpoint(
            endpoint="reset",
            data={"env_ids": env_ids, "options": options},
            requires_input=True,
        )
        if isinstance(resp, dict):
            status = resp.get("status")
            if status not in ("reset_success", "ok", "reset_ok", None):
                raise RuntimeError(f"Remote reset failed with status={status}, resp={resp}")
        return resp

    def kill(self) -> Any:
        return self.call_endpoint("kill", requires_input=False)

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {"observation": observation}
        return self.call_endpoint(endpoint="get_action", data=payload, requires_input=True)

    def get_init_info(self, requested_action_mode: str) -> dict[str, Any]:
        payload = {"requested_action_mode": requested_action_mode}
        resp = self.call_endpoint("get_init_info", data=payload, requires_input=True)
        if not isinstance(resp, dict):
            raise TypeError(f"Expected dict from get_init_info, got {type(resp)!r}")
        return resp

    def set_task_description(self, task_description: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {"task_description": task_description}
        resp = self.call_endpoint("set_task_description", data=payload, requires_input=True)
        if not isinstance(resp, dict):
            raise TypeError(f"Expected dict from set_task_description, got {type(resp)!r}")
        return resp

    def call_endpoint(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        requires_input: bool = True,
        timeout_ms: int | None = None,
    ) -> Any:
        """Generic RPC helper: publish a request and block for the matching response."""
        request_id = self._next_request_id
        self._next_request_id += 1

        request: dict[str, Any] = {"request_id": request_id, "endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        if self._config.api_token:
            request["api_token"] = self._config.api_token

        self._pub.publish(_encode_msg(self._UInt8MultiArray, MessageSerializer.to_bytes(request)))

        env = self._spin_for_response(request_id, timeout_ms if timeout_ms is not None else self._config.timeout_ms)
        payload = env.get("payload")
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(f"Server error: {payload['error']}")
        return payload

    def close(self) -> None:
        try:
            self._node.destroy_node()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[Ros2PolicyClient] destroy_node() error: {exc}")
        if self._owns_rclpy and self._rclpy.ok():
            try:
                self._rclpy.shutdown()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[Ros2PolicyClient] shutdown() error: {exc}")


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #


class Ros2PolicyServer:
    """Serve a :class:`ServerSidePolicy` over ROS2 (mirror of :class:`PolicyServer`).

    Reuses a :class:`PolicyServer` instance purely as a transport-agnostic
    dispatcher (its endpoint table + :meth:`PolicyServer.dispatch`); the ZeroMQ
    socket is never bound because :meth:`PolicyServer.run` is not called here.
    """

    def __init__(
        self,
        policy: ServerSidePolicy,
        namespace: str = "/gr00t_policy",
        domain_id: int | None = None,
        api_token: str | None = None,
        qos_depth: int = 10,
    ) -> None:
        self._config = Ros2RemotePolicyConfig(
            namespace=namespace, domain_id=domain_id, api_token=api_token, qos_depth=qos_depth
        )
        # PolicyServer is used here only as the endpoint dispatcher (no ZMQ bind).
        self._dispatcher = PolicyServer(policy=policy, api_token=api_token)

        self._rclpy, self._UInt8MultiArray, qos_types = _import_rclpy()
        self._owns_rclpy = _ensure_rclpy_init(self._rclpy, domain_id)

        self._node = self._rclpy.create_node("gr00t_policy_server")
        qos = _reliable_qos(qos_types, qos_depth)
        self._pub = self._node.create_publisher(self._UInt8MultiArray, self._config.response_topic, qos)
        self._sub = self._node.create_subscription(
            self._UInt8MultiArray, self._config.request_topic, self._on_request, qos
        )

    def _on_request(self, msg) -> None:
        try:
            request = MessageSerializer.from_bytes(_decode_msg(msg))
        except Exception as exc:
            print(f"[Ros2PolicyServer] failed to decode request: {exc}")
            return

        request_id = request.get("request_id") if isinstance(request, dict) else None
        result = self._dispatcher.dispatch(request)
        env = {"request_id": request_id, "payload": result}
        try:
            self._pub.publish(_encode_msg(self._UInt8MultiArray, MessageSerializer.to_bytes(env)))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[Ros2PolicyServer] failed to publish response: {exc}")

    def run(self) -> None:
        print(
            f"[Ros2PolicyServer] listening: request_topic={self._config.request_topic}, "
            f"response_topic={self._config.response_topic}, "
            f"domain_id={os.environ.get('ROS_DOMAIN_ID')}, api_token={self._config.api_token!r}"
        )
        try:
            while self._rclpy.ok() and self._dispatcher.running:
                self._rclpy.spin_once(self._node, timeout_sec=0.1)
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._node.destroy_node()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[Ros2PolicyServer] destroy_node() error: {exc}")
        if self._owns_rclpy and self._rclpy.ok():
            try:
                self._rclpy.shutdown()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[Ros2PolicyServer] shutdown() error: {exc}")

    @staticmethod
    def start(
        policy: ServerSidePolicy,
        namespace: str = "/gr00t_policy",
        domain_id: int | None = None,
        api_token: str | None = None,
        qos_depth: int = 10,
    ) -> None:
        server = Ros2PolicyServer(
            policy=policy,
            namespace=namespace,
            domain_id=domain_id,
            api_token=api_token,
            qos_depth=qos_depth,
        )
        server.run()
