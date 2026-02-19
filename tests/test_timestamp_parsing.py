"""Edge case tests for timestamp normalization."""
from __future__ import annotations

import math
import unittest

from wash_detector.ingest import parse_timestamp_ms


class TimestampParsingTests(unittest.TestCase):
    """Tests for parse_timestamp_ms edge cases."""

    def test_milliseconds_passthrough(self):
        """Timestamps already in ms (13 digits) pass through."""
        ts = 1707600000000
        self.assertEqual(ts, parse_timestamp_ms(ts))
        self.assertEqual(ts, parse_timestamp_ms(float(ts)))
        self.assertEqual(ts, parse_timestamp_ms(str(ts)))

    def test_seconds_to_milliseconds(self):
        """Unix seconds (10 digits) get multiplied by 1000."""
        ts_seconds = 1707600000
        expected_ms = 1707600000000
        self.assertEqual(expected_ms, parse_timestamp_ms(ts_seconds))
        self.assertEqual(expected_ms, parse_timestamp_ms(float(ts_seconds)))

    def test_microseconds_to_milliseconds(self):
        """Microseconds (16 digits) get divided by 1000."""
        ts_micros = 1707600000000000
        expected_ms = 1707600000000
        self.assertEqual(expected_ms, parse_timestamp_ms(ts_micros))

    def test_iso_string_utc(self):
        """ISO timestamp with +00:00 timezone."""
        iso_str = "2024-02-11T00:00:00+00:00"
        result = parse_timestamp_ms(iso_str)
        self.assertEqual(1707609600000, result)

    def test_iso_string_z_suffix(self):
        """ISO timestamp with Z suffix (Zulu time)."""
        iso_str = "2024-02-11T00:00:00Z"
        result = parse_timestamp_ms(iso_str)
        self.assertEqual(1707609600000, result)

    def test_iso_string_naive_treated_as_utc(self):
        """ISO timestamp without timezone treated as UTC."""
        iso_str = "2024-02-11T00:00:00"
        result = parse_timestamp_ms(iso_str)
        self.assertEqual(1707609600000, result)

    def test_numeric_string(self):
        """Numeric timestamp encoded as string."""
        ts_str = "1707600000000"
        self.assertEqual(1707600000000, parse_timestamp_ms(ts_str))

    def test_null_raises(self):
        """None value raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            parse_timestamp_ms(None)
        self.assertIn("null", str(ctx.exception))

    def test_empty_string_raises(self):
        """Empty string raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            parse_timestamp_ms("")
        self.assertIn("empty", str(ctx.exception))

    def test_whitespace_string_raises(self):
        """Whitespace-only string raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            parse_timestamp_ms("   ")
        self.assertIn("empty", str(ctx.exception))

    def test_nan_raises(self):
        """NaN value raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            parse_timestamp_ms(float("nan"))
        self.assertIn("not finite", str(ctx.exception))

    def test_infinity_raises(self):
        """Infinity raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            parse_timestamp_ms(float("inf"))
        self.assertIn("not finite", str(ctx.exception))

    def test_too_large_timestamp_raises(self):
        """Extremely large timestamp raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            parse_timestamp_ms(1e17)
        self.assertIn("too large", str(ctx.exception))

    def test_too_small_timestamp_raises(self):
        """Very small timestamp raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            parse_timestamp_ms(1000)
        self.assertIn("too small", str(ctx.exception))

    def test_negative_timestamp_accepted(self):
        """Negative timestamps (before 1970) are accepted."""
        # Uses absolute value for magnitude classification
        ts_neg_ms = -1707600000000  # Well before 1970
        result = parse_timestamp_ms(ts_neg_ms)
        self.assertEqual(ts_neg_ms, result)


if __name__ == "__main__":
    unittest.main()
