import sys
import parsl


class ExaPdLogger:
    """
    Lightweight logging utility for ExaPD workflows.

    This class provides a minimal logging interface with five severity levels:
    DEBUG, INFO, WARNING, ERROR, and CRITICAL. Messages are written to stdout
    or stderr depending on severity, and CRITICAL messages terminate the
    program after attempting to clean up Parsl resources.

    Parameters
    ----------
    level_name : str, optional
        Initial logging level (default: "INFO").
    logger_name : str, optional
        Name prefix for all log messages (default: "exa-pd").

    Notes
    -----
    - DEBUG and INFO messages are written to stdout.
    - WARNING and ERROR messages are written to stderr.
    - CRITICAL messages are written to stderr and then cause program exit.
    - If Parsl is active, a cleanup is attempted before exit.
    """

    LEVEL_MAP = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50
    }

    def __init__(self, level_name="INFO", logger_name="exa-pd"):
        """
        Initialize the logger with a given severity level and name.

        Parameters
        ----------
        level_name : str, optional
            Logging level name.
        logger_name : str, optional
            Logger name prefix.
        """
        self.logger_name = logger_name
        self.configure(level_name)

    def configure(self, level_name="INFO"):
        """
        Configure the logging level.

        If an unsupported logging level is provided, the logger falls back
        to INFO and prints a warning message to stdout.

        Parameters
        ----------
        level_name : str, optional
            Logging level name.
        """
        level_name = level_name.upper()
        if level_name in self.LEVEL_MAP:
            self._current_level = self.LEVEL_MAP[level_name]
        else:
            self._current_level = self.LEVEL_MAP["INFO"]
            sys.stdout.write(
                f"Unsupported log level '{level_name}'. Falling back to INFO.\n"
            )

    def _log(self, level_name, message):
        """
        Internal logging dispatcher.

        Parameters
        ----------
        level_name : {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            Severity level.
        message : str
            Log message.

        Notes
        -----
        - DEBUG/INFO → stdout
        - WARNING/ERROR → stderr
        - CRITICAL → stderr + Parsl cleanup + program exit
        """
        numeric_level = self.LEVEL_MAP[level_name]

        if numeric_level < self._current_level:
            return

        formatted_message = f"[{level_name}] {self.logger_name}: {message}"

        if numeric_level <= self.LEVEL_MAP["INFO"]:
            sys.stdout.write(formatted_message + "\n")
        elif numeric_level < self.LEVEL_MAP["CRITICAL"]:
            sys.stderr.write(formatted_message + "\n")
        else:
            sys.stderr.write(formatted_message + "\n")
            try:
                parsl.dfk().cleanup()
            except BaseException:
                pass
            sys.exit(1)

    def debug(self, message):
        """
        Log a DEBUG-level message.

        Parameters
        ----------
        message : str
            Message to log.
        """
        self._log("DEBUG", message)

    def info(self, message):
        """
        Log an INFO-level message.

        Parameters
        ----------
        message : str
            Message to log.
        """
        self._log("INFO", message)

    def warning(self, message):
        """
        Log a WARNING-level message.

        Parameters
        ----------
        message : str
            Message to log.
        """
        self._log("WARNING", message)

    def error(self, message):
        """
        Log an ERROR-level message.

        Parameters
        ----------
        message : str
            Message to log.
        """
        self._log("ERROR", message)

    def critical(self, message):
        """
        Log a CRITICAL-level message and terminate the program.

        Parameters
        ----------
        message : str
            Message to log.

        Notes
        -----
        This method always results in program termination.
        """
        self._log("CRITICAL", message)


# Global logger instance
exapd_logger = ExaPdLogger()
