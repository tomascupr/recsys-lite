"""Simple test for the logging module."""

import unittest

from recsys_lite.utils.logging import LogLevel, configure_logging, get_logger


class TestLogging(unittest.TestCase):
    """Test suite for the logging module."""

    def test_get_logger(self):
        """Test the get_logger function."""
        # Test with a simple name
        logger = get_logger("test")
        self.assertEqual(logger.name, "recsys-lite.test")

        # Test with a name that already has the prefix
        logger = get_logger("recsys-lite.test")
        self.assertEqual(logger.name, "recsys-lite.test")

        # Test with __main__
        logger = get_logger("__main__")
        self.assertEqual(logger.name, "recsys-lite")

    def test_log_levels(self):
        """Test the LogLevel enum."""
        self.assertEqual(LogLevel.DEBUG, "DEBUG")
        self.assertEqual(LogLevel.INFO, "INFO")
        self.assertEqual(LogLevel.WARNING, "WARNING")
        self.assertEqual(LogLevel.ERROR, "ERROR")
        self.assertEqual(LogLevel.CRITICAL, "CRITICAL")


if __name__ == "__main__":
    # Configure logging
    configure_logging(level=LogLevel.DEBUG)

    # Run the tests
    unittest.main()
