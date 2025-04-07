import os

def load_settings(filename):
    settings = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()

                # Remove trailing inline comments after `#`
                value = value.split("#", 1)[0].strip()

                # Expand user (~) for paths
                if any(kw in key.lower() for kw in ["folder", "file", "path"]):
                    value = os.path.expanduser(value)

                settings[key] = value

    return settings


