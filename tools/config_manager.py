import argparse
import json
import sys

class ConfigManager:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Config Manager")
        parser.add_argument(
            "--config", required=True,
            help="Path to JSON configuration file"
        )
        args = parser.parse_args()

        try:
            with open(args.config, "r") as f:
                self._config = json.load(f)
        except Exception as e:
            print(f"Error reading config file {args.config}: {e}", file=sys.stderr)
            sys.exit(1)

    def __getitem__(self, key):
        return self._config[key]

    def get(self, key, default=None):
        return self._config.get(key, default)

    def __contains__(self, key):
        return key in self._config
