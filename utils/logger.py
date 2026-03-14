import json
import os

LOG_FILE = "logs/agent_run.json"


def log_run(data):

    os.makedirs("logs", exist_ok=True)

    # If file does not exist, create it
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([], f)

    # Try reading JSON safely
    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except:
        logs = []

    logs.append(data)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)