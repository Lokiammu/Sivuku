#!/usr/bin/env python3
import json

# Read notebook
with open('examples/OpenEnv_Tutorial.ipynb', 'r') as f:
    nb = json.load(f)

# Insert TOC after cell 1
toc_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "## 📑 Table of Contents\n",
        "\n",
        "<div style=\"background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;\">\n",
        "\n",
        "**Quick Navigation** - Click any section to jump right there! 🎯\n",
        "\n",
        "### Foundation\n",
        "- [Part 1: RL in 60 Seconds ⏱️](#part-1)\n",
        "- [Part 2: The Problem with Traditional RL 😤](#part-2)\n",
        "- [Part 3: Setup 🛠️](#part-3)\n",
        "\n",
        "### Architecture\n",
        "- [Part 4: The OpenEnv Pattern 🏗️](#part-4)\n",
        "- [Part 5: Example Integration - OpenSpiel 🎮](#part-5)\n",
        "\n",
        "### Hands-On Demo\n",
        "- [Part 6: Interactive Demo 🎮](#part-6)\n",
        "- [Part 7: Four Policies 🤖](#part-7)\n",
        "- [Part 8: Policy Competition! 🏆](#part-8)\n",
        "\n",
        "### Advanced\n",
        "- [Part 9: Using Real OpenSpiel 🎮](#part-9)\n",
        "- [Part 10: Create Your Own Integration 🛠️](#part-10)\n",
        "\n",
        "### Wrap Up\n",
        "- [Summary: Your Journey 🎓](#summary)\n",
        "- [Resources 📚](#resources)\n",
        "\n",
        "</div>\n",
        "\n",
        "---"
    ]
}

# Insert setup code cell after Part 3 header
setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Detect environment\n",
        "try:\n",
        "    import google.colab\n",
        "    IN_COLAB = True\n",
        "    print(\"🌐 Running in Google Colab - Perfect!\")\n",
        "except ImportError:\n",
        "    IN_COLAB = False\n",
        "    print(\"💻 Running locally - Nice!\")\n",
        "\n",
        "if IN_COLAB:\n",
        "    print(\"\\n📦 Cloning OpenEnv repository...\")\n",
        "    !git clone https://github.com/meta-pytorch/OpenEnv.git > /dev/null 2>&1\n",
        "    %cd OpenEnv\n",
        "    \n",
        "    print(\"📚 Installing dependencies (this takes ~10 seconds)...\")\n",
        "    !pip install -q fastapi uvicorn requests\n",
        "    \n",
        "    import sys\n",
        "    sys.path.insert(0, './src')\n",
        "    print(\"\\n✅ Setup complete! Everything is ready to go! 🎉\")\n",
        "else:\n",
        "    import sys\n",
        "    from pathlib import Path\n",
        "    sys.path.insert(0, str(Path.cwd().parent / 'src'))\n",
        "    print(\"✅ Using local OpenEnv installation\")\n",
        "\n",
        "print(\"\\n🚀 Ready to explore OpenEnv and build amazing things!\")\n",
        "print(\"💡 Tip: Run cells top-to-bottom for the best experience.\\n\")"
    ]
}

# Insert architecture diagram after Part 2
arch_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### The Architecture\n",
        "\n",
        "```\n",
        "┌────────────────────────────────────────────────────────────┐\n",
        "│  YOUR TRAINING CODE                                        │\n",
        "│                                                            │\n",
        "│  env = OpenSpielEnv(...)        ← Import the client      │\n",
        "│  result = env.reset()           ← Type-safe!             │\n",
        "│  result = env.step(action)      ← Type-safe!             │\n",
        "│                                                            │\n",
        "└─────────────────┬──────────────────────────────────────────┘\n",
        "                  │\n",
        "                  │  HTTP/JSON (Language-Agnostic)\n",
        "                  │  POST /reset, POST /step, GET /state\n",
        "                  │\n",
        "┌─────────────────▼──────────────────────────────────────────┐\n",
        "│  DOCKER CONTAINER                                          │\n",
        "│                                                            │\n",
        "│  ┌──────────────────────────────────────────────┐         │\n",
        "│  │  FastAPI Server                              │         │\n",
        "│  │  └─ Environment (reset, step, state)         │         │\n",
        "│  │     └─ Your Game/Simulation Logic            │         │\n",
        "│  └──────────────────────────────────────────────┘         │\n",
        "│                                                            │\n",
        "│  Isolated • Reproducible • Secure                          │\n",
        "└────────────────────────────────────────────────────────────┘\n",
        "```\n",
        "\n",
        "<div style=\"background-color: #e7f3ff; padding: 15px; border-left: 5px solid #0366d6; margin: 20px 0;\">\n",
        "\n",
        "**🎯 Key Insight**: You never see HTTP details - just clean Python methods!\n",
        "\n",
        "```python\n",
        "env.reset()    # Under the hood: HTTP POST to /reset\n",
        "env.step(...)  # Under the hood: HTTP POST to /step\n",
        "env.state()    # Under the hood: HTTP GET to /state\n",
        "```\n",
        "\n",
        "The magic? OpenEnv handles all the plumbing. You focus on RL! ✨\n",
        "\n",
        "</div>"
    ]
}

# Check which cells exist
has_toc = any('Table of Contents' in ''.join(cell.get('source', [])) for cell in nb['cells'])
has_setup = any('IN_COLAB' in ''.join(cell.get('source', [])) for cell in nb['cells'])
has_arch = any('┌───' in ''.join(cell.get('source', [])) and 'YOUR TRAINING CODE' in ''.join(cell.get('source', [])) for cell in nb['cells'])

print(f"Current state:")
print(f"  TOC present: {has_toc}")
print(f"  Setup code present: {has_setup}")
print(f"  Architecture diagram present: {has_arch}")

# Insert TOC after cell 1 if missing
if not has_toc:
    nb['cells'].insert(2, toc_cell)
    print("✅ Added TOC")

# Find Part 3 header and add setup cell after it if missing
if not has_setup:
    for i, cell in enumerate(nb['cells']):
        if 'Part 3: Setup' in ''.join(cell.get('source', [])) and cell['cell_type'] == 'markdown':
            nb['cells'].insert(i + 1, setup_cell)
            print("✅ Added setup code cell")
            break

# Find Part 2 and add architecture diagram if missing
if not has_arch:
    for i, cell in enumerate(nb['cells']):
        if 'Part 2:' in ''.join(cell.get('source', [])) and 'The OpenEnv Philosophy' in ''.join(cell.get('source', [])):
            nb['cells'].insert(i + 1, arch_cell)
            print("✅ Added architecture diagram")
            break

# Save
with open('examples/OpenEnv_Tutorial.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n✅ Notebook fixed! Total cells: {len(nb['cells'])}")
