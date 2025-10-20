# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Web interface for OpenEnv environments.

This module provides a web-based interface for interacting with OpenEnv environments,
including a two-pane layout for HumanAgent interaction and state observation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Type
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .interfaces import Environment
from .types import Action, Observation, State


@dataclass
class ActionLog:
    """Log entry for an action taken."""
    timestamp: str
    action: Dict[str, Any]
    observation: Dict[str, Any]
    reward: Optional[float]
    done: bool
    step_count: int


@dataclass
class EpisodeState:
    """Current episode state for the web interface."""
    episode_id: Optional[str]
    step_count: int
    current_observation: Optional[Dict[str, Any]]
    action_logs: List[ActionLog]
    is_reset: bool = True


class WebInterfaceManager:
    """Manages the web interface for an environment."""
    
    def __init__(
        self,
        env: Environment,
        action_cls: Type[Action],
        observation_cls: Type[Observation],
    ):
        self.env = env
        self.action_cls = action_cls
        self.observation_cls = observation_cls
        self.episode_state = EpisodeState(
            episode_id=None,
            step_count=0,
            current_observation=None,
            action_logs=[]
        )
        self.connected_clients: List[WebSocket] = []
    
    async def connect_websocket(self, websocket: WebSocket):
        """Connect a new WebSocket client."""
        await websocket.accept()
        self.connected_clients.append(websocket)
        
        # Send current state to the new client
        await self._send_state_update()
    
    async def disconnect_websocket(self, websocket: WebSocket):
        """Disconnect a WebSocket client."""
        if websocket in self.connected_clients:
            self.connected_clients.remove(websocket)
    
    async def _send_state_update(self):
        """Send current state to all connected clients."""
        if not self.connected_clients:
            return
            
        state_data = {
            "type": "state_update",
            "episode_state": asdict(self.episode_state)
        }
        
        # Send to all connected clients
        disconnected_clients = []
        for client in self.connected_clients:
            try:
                await client.send_text(json.dumps(state_data))
            except:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.connected_clients.remove(client)
    
    async def reset_environment(self) -> Dict[str, Any]:
        """Reset the environment and update state."""
        observation = self.env.reset()
        state = self.env.state
        
        # Update episode state
        self.episode_state.episode_id = state.episode_id
        self.episode_state.step_count = 0
        self.episode_state.current_observation = asdict(observation)
        self.episode_state.action_logs = []
        self.episode_state.is_reset = True
        
        # Send state update
        await self._send_state_update()
        
        return {
            "observation": asdict(observation),
            "reward": observation.reward,
            "done": observation.done,
        }
    
    async def step_environment(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a step in the environment and update state."""
        # Deserialize action
        action = self._deserialize_action(action_data)
        
        # Execute step
        observation = self.env.step(action)
        state = self.env.state
        
        # Create action log
        action_log = ActionLog(
            timestamp=datetime.now().isoformat(),
            action=asdict(action),
            observation=asdict(observation),
            reward=observation.reward,
            done=observation.done,
            step_count=state.step_count
        )
        
        # Update episode state
        self.episode_state.episode_id = state.episode_id
        self.episode_state.step_count = state.step_count
        self.episode_state.current_observation = asdict(observation)
        self.episode_state.action_logs.append(action_log)
        self.episode_state.is_reset = False
        
        # Send state update
        await self._send_state_update()
        
        return {
            "observation": asdict(observation),
            "reward": observation.reward,
            "done": observation.done,
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current environment state."""
        state = self.env.state
        return asdict(state)
    
    def _deserialize_action(self, action_data: Dict[str, Any]) -> Action:
        """Convert JSON dict to Action instance."""
        metadata = action_data.pop("metadata", {})
        action = self.action_cls(**action_data)
        action.metadata = metadata
        return action


def create_web_interface_app(
    env: Environment,
    action_cls: Type[Action],
    observation_cls: Type[Observation],
) -> FastAPI:
    """
    Create a FastAPI application with web interface for the given environment.
    
    Args:
        env: The Environment instance to serve
        action_cls: The Action subclass this environment expects
        observation_cls: The Observation subclass this environment returns
        
    Returns:
        FastAPI application instance with web interface
    """
    from .http_server import create_fastapi_app
    
    # Create the base environment app
    app = create_fastapi_app(env, action_cls, observation_cls)
    
    # Create web interface manager
    web_manager = WebInterfaceManager(env, action_cls, observation_cls)
    
    # Add web interface routes
    @app.get("/web", response_class=HTMLResponse)
    async def web_interface():
        """Serve the web interface."""
        return get_web_interface_html(action_cls)
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await web_manager.connect_websocket(websocket)
        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            await web_manager.disconnect_websocket(websocket)
    
    @app.post("/web/reset")
    async def web_reset():
        """Reset endpoint for web interface."""
        return await web_manager.reset_environment()
    
    @app.post("/web/step")
    async def web_step(request: Dict[str, Any]):
        """Step endpoint for web interface."""
        action_data = request.get("action", {})
        return await web_manager.step_environment(action_data)
    
    @app.get("/web/state")
    async def web_state():
        """State endpoint for web interface."""
        return web_manager.get_state()
    
    return app


def get_web_interface_html(action_cls: Type[Action]) -> str:
    """Generate the HTML for the web interface."""
    
    # Get action fields for dynamic form generation
    action_fields = []
    if hasattr(action_cls, '__dataclass_fields__'):
        for field_name, field_info in action_cls.__dataclass_fields__.items():
            if field_name != 'metadata':
                field_type = field_info.type
                if field_type == str:
                    input_type = "text"
                elif field_type == int:
                    input_type = "number"
                elif field_type == float:
                    input_type = "number"
                elif field_type == bool:
                    input_type = "checkbox"
                else:
                    input_type = "text"
                
                action_fields.append({
                    'name': field_name,
                    'type': input_type,
                    'required': field_info.default is field_info.default_factory
                })
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenEnv Web Interface</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            height: 100vh;
            overflow: hidden;
        }}
        
        .container {{
            display: flex;
            height: 100vh;
        }}
        
        .left-pane {{
            width: 50%;
            background: white;
            border-right: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
        }}
        
        .right-pane {{
            width: 50%;
            background: #fafafa;
            display: flex;
            flex-direction: column;
        }}
        
        .pane-header {{
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
            background: #f8f9fa;
            font-weight: 600;
            font-size: 16px;
        }}
        
        .pane-content {{
            flex: 1;
            padding: 20px;
            overflow-y: auto;
        }}
        
        .action-form {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .form-group {{
            margin-bottom: 15px;
        }}
        
        .form-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #333;
        }}
        
        .form-group input, .form-group textarea {{
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        .form-group input:focus, .form-group textarea:focus {{
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }}
        
        .btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
            margin-bottom: 10px;
        }}
        
        .btn:hover {{
            background: #0056b3;
        }}
        
        .btn:disabled {{
            background: #6c757d;
            cursor: not-allowed;
        }}
        
        .btn-secondary {{
            background: #6c757d;
        }}
        
        .btn-secondary:hover {{
            background: #545b62;
        }}
        
        .state-display {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        
        .state-item {{
            margin-bottom: 8px;
        }}
        
        .state-label {{
            font-weight: 500;
            color: #666;
        }}
        
        .state-value {{
            color: #333;
            font-family: monospace;
        }}
        
        .logs-container {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .log-entry {{
            border-bottom: 1px solid #f0f0f0;
            padding: 10px 0;
        }}
        
        .log-entry:last-child {{
            border-bottom: none;
        }}
        
        .log-timestamp {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .log-action {{
            background: #e3f2fd;
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 5px;
            font-family: monospace;
            font-size: 12px;
        }}
        
        .log-observation {{
            background: #f3e5f5;
            padding: 8px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
        }}
        
        .log-reward {{
            font-weight: 600;
            color: #28a745;
        }}
        
        .log-done {{
            font-weight: 600;
            color: #dc3545;
        }}
        
        .status-indicator {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .status-connected {{
            background: #28a745;
        }}
        
        .status-disconnected {{
            background: #dc3545;
        }}
        
        .json-display {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 10px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Left Pane: HumanAgent Interface -->
        <div class="left-pane">
            <div class="pane-header">
                <span class="status-indicator status-disconnected" id="connection-status"></span>
                HumanAgent Interface
            </div>
            <div class="pane-content">
                <!-- Action Form -->
                <div class="action-form">
                    <h3>Take Action</h3>
                    <form id="action-form">
                        {_generate_action_form_fields(action_fields)}
                        <button type="submit" class="btn" id="step-btn">Step</button>
                    </form>
                </div>
                
                <!-- Control Buttons -->
                <div style="margin-bottom: 20px;">
                    <button class="btn btn-secondary" id="reset-btn">Reset Environment</button>
                    <button class="btn btn-secondary" id="state-btn">Get State</button>
                </div>
                
                <!-- Current State Display -->
                <div class="state-display">
                    <h3>Current State</h3>
                    <div id="current-state">
                        <div class="state-item">
                            <span class="state-label">Status:</span>
                            <span class="state-value" id="env-status">Not initialized</span>
                        </div>
                        <div class="state-item">
                            <span class="state-label">Episode ID:</span>
                            <span class="state-value" id="episode-id">-</span>
                        </div>
                        <div class="state-item">
                            <span class="state-label">Step Count:</span>
                            <span class="state-value" id="step-count">0</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Right Pane: State Observer -->
        <div class="right-pane">
            <div class="pane-header">
                State Observer
            </div>
            <div class="pane-content">
                <!-- Current Observation -->
                <div class="state-display">
                    <h3>Current Observation</h3>
                    <div id="current-observation" class="json-display">
                        No observation yet
                    </div>
                </div>
                
                <!-- Action Logs -->
                <div class="logs-container">
                    <h3>Action History</h3>
                    <div id="action-logs">
                        No actions taken yet
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        class OpenEnvWebInterface {{
            constructor() {{
                this.ws = null;
                this.isConnected = false;
                this.init();
            }}
            
            init() {{
                this.connectWebSocket();
                this.setupEventListeners();
            }}
            
            connectWebSocket() {{
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${{protocol}}//${{window.location.host}}/ws`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {{
                    this.isConnected = true;
                    this.updateConnectionStatus(true);
                    console.log('WebSocket connected');
                }};
                
                this.ws.onmessage = (event) => {{
                    const data = JSON.parse(event.data);
                    if (data.type === 'state_update') {{
                        this.updateUI(data.episode_state);
                    }}
                }};
                
                this.ws.onclose = () => {{
                    this.isConnected = false;
                    this.updateConnectionStatus(false);
                    console.log('WebSocket disconnected');
                    // Attempt to reconnect after 3 seconds
                    setTimeout(() => this.connectWebSocket(), 3000);
                }};
                
                this.ws.onerror = (error) => {{
                    console.error('WebSocket error:', error);
                }};
            }}
            
            setupEventListeners() {{
                // Action form submission
                document.getElementById('action-form').addEventListener('submit', (e) => {{
                    e.preventDefault();
                    this.submitAction();
                }});
                
                // Reset button
                document.getElementById('reset-btn').addEventListener('click', () => {{
                    this.resetEnvironment();
                }});
                
                // State button
                document.getElementById('state-btn').addEventListener('click', () => {{
                    this.getState();
                }});
            }}
            
            async submitAction() {{
                const formData = new FormData(document.getElementById('action-form'));
                const action = {{}};
                
                // Collect form data
                for (const [key, value] of formData.entries()) {{
                    if (value !== '') {{
                        action[key] = value;
                    }}
                }}
                
                try {{
                    const response = await fetch('/web/step', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ action }})
                    }});
                    
                    if (!response.ok) {{
                        throw new Error(`HTTP error! status: ${{response.status}}`);
                    }}
                    
                    const result = await response.json();
                    console.log('Step result:', result);
                }} catch (error) {{
                    console.error('Error submitting action:', error);
                    alert('Error submitting action: ' + error.message);
                }}
            }}
            
            async resetEnvironment() {{
                try {{
                    const response = await fetch('/web/reset', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }}
                    }});
                    
                    if (!response.ok) {{
                        throw new Error(`HTTP error! status: ${{response.status}}`);
                    }}
                    
                    const result = await response.json();
                    console.log('Reset result:', result);
                }} catch (error) {{
                    console.error('Error resetting environment:', error);
                    alert('Error resetting environment: ' + error.message);
                }}
            }}
            
            async getState() {{
                try {{
                    const response = await fetch('/web/state');
                    const state = await response.json();
                    console.log('Current state:', state);
                    alert('Current state: ' + JSON.stringify(state, null, 2));
                }} catch (error) {{
                    console.error('Error getting state:', error);
                    alert('Error getting state: ' + error.message);
                }}
            }}
            
            updateConnectionStatus(connected) {{
                const indicator = document.getElementById('connection-status');
                if (connected) {{
                    indicator.className = 'status-indicator status-connected';
                }} else {{
                    indicator.className = 'status-indicator status-disconnected';
                }}
            }}
            
            updateUI(episodeState) {{
                // Update current state
                document.getElementById('env-status').textContent = 
                    episodeState.is_reset ? 'Reset' : 'Running';
                document.getElementById('episode-id').textContent = 
                    episodeState.episode_id || '-';
                document.getElementById('step-count').textContent = 
                    episodeState.step_count.toString();
                
                // Update current observation
                const observationDiv = document.getElementById('current-observation');
                if (episodeState.current_observation) {{
                    observationDiv.textContent = JSON.stringify(
                        episodeState.current_observation, null, 2
                    );
                }} else {{
                    observationDiv.textContent = 'No observation yet';
                }}
                
                // Update action logs
                const logsDiv = document.getElementById('action-logs');
                if (episodeState.action_logs.length === 0) {{
                    logsDiv.innerHTML = 'No actions taken yet';
                }} else {{
                    logsDiv.innerHTML = episodeState.action_logs.map(log => `
                        <div class="log-entry">
                            <div class="log-timestamp">${{log.timestamp}} (Step ${{log.step_count}})</div>
                            <div class="log-action">Action: ${{JSON.stringify(log.action, null, 2)}}</div>
                            <div class="log-observation">Observation: ${{JSON.stringify(log.observation, null, 2)}}</div>
                            <div>
                                <span class="log-reward">Reward: ${{log.reward !== null ? log.reward : 'None'}}</span>
                                ${{log.done ? '<span class="log-done">DONE</span>' : ''}}
                            </div>
                        </div>
                    `).join('');
                }}
            }}
        }}
        
        // Initialize the web interface when the page loads
        document.addEventListener('DOMContentLoaded', () => {{
            new OpenEnvWebInterface();
        }});
    </script>
</body>
</html>
    """.replace('{_generate_action_form_fields(action_fields)}', _generate_action_form_fields(action_fields))


def _generate_action_form_fields(action_fields: List[Dict[str, Any]]) -> str:
    """Generate HTML form fields for action input."""
    if not action_fields:
        return '<p>No action fields available</p>'
    
    fields_html = []
    for field in action_fields:
        if field['type'] == 'checkbox':
            fields_html.append(f'''
                <div class="form-group">
                    <label>
                        <input type="checkbox" name="{field['name']}" value="true">
                        {field['name']}
                    </label>
                </div>
            ''')
        elif field['type'] == 'text' and 'message' in field['name'].lower():
            fields_html.append(f'''
                <div class="form-group">
                    <label for="{field['name']}">{field['name']}:</label>
                    <textarea name="{field['name']}" id="{field['name']}" rows="3" placeholder="Enter {field['name']}..."></textarea>
                </div>
            ''')
        else:
            fields_html.append(f'''
                <div class="form-group">
                    <label for="{field['name']}">{field['name']}:</label>
                    <input type="{field['type']}" name="{field['name']}" id="{field['name']}" placeholder="Enter {field['name']}..." {"required" if field['required'] else ""}>
                </div>
            ''')
    
    return '\n'.join(fields_html)
