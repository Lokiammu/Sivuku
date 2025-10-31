"""Data models for the WebArena environment.

WebArena is a realistic web environment for building autonomous agents that
interact with web interfaces like shopping sites, forums, and more.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.env_server.types import Action, Observation, State


@dataclass(kw_only=True)
class WebArenaAction(Action):
    """Action to be executed in the WebArena environment.

    WebArena supports various action types including:
    - click: Click on an element
    - type: Type text into an element
    - goto: Navigate to a URL
    - scroll: Scroll the page
    - select_option: Select from a dropdown
    - And many more...

    The action_str follows the WebArena action format like:
    - "click [123]" - Click element with ID 123
    - "type [45] [hello world]" - Type into element with ID 45
    - "goto [http://example.com]" - Navigate to URL
    - "scroll [down]" - Scroll down
    - "stop []" - Stop the episode
    """

    action_str: str
    """The action string in WebArena format (e.g., 'click [123]', 'type [45] [text]')"""


@dataclass(kw_only=True)
class WebArenaObservation(Observation):
    """Observation returned from the WebArena environment.

    Contains the current state of the web page as an accessibility tree or HTML,
    along with additional metadata about the page and action execution.
    """

    text: str = ""
    """The text representation of the page (accessibility tree or HTML)"""

    url: str = ""
    """The current URL of the page"""

    success: bool = True
    """Whether the last action executed successfully"""

    fail_error: str = ""
    """Error message if the last action failed"""

    observation_metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the observation"""


@dataclass
class WebArenaState(State):
    """State of the WebArena environment.

    Tracks the current episode, configuration, and progress through a task.
    """

    config_file: Optional[str] = None
    """Path to the config file for the current task"""

    task_id: Optional[str] = None
    """The task ID being executed"""

    intent: str = ""
    """The task intent/instruction"""

    current_url: str = ""
    """Current URL of the active page"""

    terminated: bool = False
    """Whether the episode has ended"""
