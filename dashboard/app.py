"""HuggingFace Spaces entry point.

Spaces looks for `app.py` at the repo root of the Space. This file simply
delegates to dashboard/gradio_app.py.
"""

from gradio_app import main

if __name__ == "__main__":
    main()
