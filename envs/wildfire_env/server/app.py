# server/app.py
import os
from fastapi.responses import HTMLResponse

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.web_interface import load_environment_metadata
from ..models import WildfireAction, WildfireObservation
from .wildfire_environment import WildfireEnvironment
from .wildfire_web_interface import get_wildfire_web_interface_html

W = int(os.getenv("WILDFIRE_WIDTH", "16"))
H = int(os.getenv("WILDFIRE_HEIGHT", "16"))

# Factory function to create WildfireEnvironment instances
def create_wildfire_environment():
    """Factory function that creates WildfireEnvironment with config."""
    return WildfireEnvironment(width=W, height=H)

# Create the app with web interface support
# create_app handles ENABLE_WEB_INTERFACE automatically
app = create_app(
    create_wildfire_environment,
    WildfireAction,
    WildfireObservation,
    env_name="wildfire_env",
)

# Check if web interface should be enabled for custom routes
enable_web = (
    os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")
)

if enable_web:
    # Load metadata for custom wildfire interface
    env_instance = create_wildfire_environment()
    metadata = load_environment_metadata(env_instance, "wildfire_env")

    # Add our custom wildfire interface route (overrides default /web)
    @app.get("/web", response_class=HTMLResponse)
    async def wildfire_web_interface():
        """Custom wildfire-specific web interface."""
        return get_wildfire_web_interface_html(metadata)


def main():
    """Main entry point for running the server."""
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
