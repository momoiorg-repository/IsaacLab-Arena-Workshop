# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RemotePolicyConfig:
    """Configuration for using a remote PolicyServer over ZeroMQ."""

    host: str
    port: int
    api_token: str | None = None
    timeout_ms: int = 15000

    @property
    def description(self) -> str:
        """Human-readable target, for transport-agnostic error/log messages."""
        return f"tcp://{self.host}:{self.port}"


@dataclass
class Ros2RemotePolicyConfig:
    """Configuration for using a remote policy server over ROS2.

    The transport uses a topic-based request/reply pair under ``namespace``:
    requests are published on ``<namespace>/request`` and responses are read
    from ``<namespace>/response`` (both ``std_msgs/UInt8MultiArray`` carrying the
    same msgpack payload as the ZeroMQ transport).
    """

    namespace: str = "/gr00t_policy"
    domain_id: int | None = None
    api_token: str | None = None
    timeout_ms: int = 15000
    qos_depth: int = 10

    @property
    def request_topic(self) -> str:
        return f"{self.namespace.rstrip('/')}/request"

    @property
    def response_topic(self) -> str:
        return f"{self.namespace.rstrip('/')}/response"

    @property
    def description(self) -> str:
        """Human-readable target, for transport-agnostic error/log messages."""
        return f"ROS2 {self.request_topic} (domain {self.domain_id})"
