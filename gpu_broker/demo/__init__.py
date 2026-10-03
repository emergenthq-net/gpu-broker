"""`gpu-broker demo`: the real broker, API and dashboard on a simulated GPU.

Nothing here touches hardware or starts a process. A simulated driver (driver.py) and
simulated model servers (backends.py) stand in for systemd/Docker units, llama.cpp and
ComfyUI; everything above them (queue, residency switches, idle restore, events, the
dashboard) is the production code, unchanged. A traffic generator (traffic.py) plays a small
team using the card so the dashboard has something to show.

Every timing and size the simulation uses is in tuning.py; the text it says is in content.py.
"""
