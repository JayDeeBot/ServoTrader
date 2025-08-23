# -------- Runtime image for live agent --------
FROM python:3.11-slim

# System deps:
# - tzdata: timezone
# - python3-tk + tk: Tkinter for GUI
# - libgl1: common for image displays; x11-utils optional for debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata python3-tk tk libgl1 xauth x11-apps \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Australia/Sydney \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    # Install Torch CPU wheels explicitly from the official CPU index
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 && \
    # Install remaining dependencies from requirements.txt
    pip install --no-cache-dir -r requirements.txt

# Copy only the minimal code needed to run live (no training)
# If your GUI lives under /gui, copy it too.
COPY servo_trader /app/servo_trader
COPY scripts /app/scripts
COPY gui /app/gui

# Create standard runtime dirs (bind-mounted at runtime)
RUN mkdir -p /app/data /app/logs /app/models /app/config

# ----- Compatibility symlink for legacy absolute paths -----
# Map /home/jarred/git/ServoTrader -> /app so code that references
# /home/jarred/git/ServoTrader/data (etc) transparently hits /app/data.
RUN mkdir -p /home/jarred/git && \
    ln -sfn /app /home/jarred/git/ServoTrader

# Non-root for nicer file perms
RUN useradd -ms /bin/bash trader && chown -R trader:trader /app /home/jarred
USER trader

# Default command (overridden by docker-compose)
CMD ["python", "scripts/web_live_ppo_agent.py"]