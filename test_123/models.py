# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Test 123 Environment.

The test_123 environment is a simple test environment that echoes back messages.
"""

from dataclasses import dataclass

from openenv_core.env_server.types import Action, Observation


@dataclass(kw_only=True)
class Test123Action(Action):
    """Action for the Test 123 environment - just a message to echo."""

    message: str


@dataclass(kw_only=True)
class Test123Observation(Observation):
    """Observation from the Test 123 environment - the echoed message."""

    echoed_message: str
    message_length: int = 0

