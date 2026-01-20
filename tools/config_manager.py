import argparse
import json
import sys


class ConfigManager:
    """
    Simple configuration manager for loading JSON-based input files.

    This class parses a command-line argument ``--config`` pointing to a JSON
    configuration file, loads the file into memory, and provides dictionary-like
    access to its contents.

    Notes
    -----
    - The configuration file is parsed at object construction time.
    - If the file cannot be read or parsed, the program exits immediately.
    - This class is intentionally lightweight and does not perform schema
      validation.
    """

    def __init__(self):
        """
        Parse command-line arguments and load the JSON configuration file.

        Raises
        ------
        SystemExit
            If the configuration file cannot be read or parsed.
        """
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
            print(
                f"Error reading config file {args.config}: {e}",
                file=sys.stderr
            )
            sys.exit(1)

    def __getitem__(self, key):
        """
        Return a configuration value using dictionary-style indexing.

        Parameters
        ----------
        key : str
            Configuration key.

        Returns
        -------
        object
            Value associated with `key`.

        Raises
        ------
        KeyError
            If `key` does not exist in the configuration.
        """
        return self._config[key]

    def get(self, key, default=None):
        """
        Return a configuration value with a default fallback.

        Parameters
        ----------
        key : str
            Configuration key.
        default : object, optional
            Value to return if `key` is not present.

        Returns
        -------
        object
            Configuration value or `default`.
        """
        return self._config.get(key, default)

    def __contains__(self, key):
        """
        Check whether a configuration key exists.

        Parameters
        ----------
        key : str
            Configuration key.

        Returns
        -------
        bool
            True if `key` exists in the configuration, False otherwise.
        """
        return key in self._config
