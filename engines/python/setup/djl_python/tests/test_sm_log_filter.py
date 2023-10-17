import unittest
from unittest.mock import MagicMock
from djl_python.sm_log_filter import SMLogFilter
import logging


class TestSMLogFilter(unittest.TestCase):

    def test_filter_hit(self):
        filter = SMLogFilter()

        record = MagicMock()
        record.msg = f"SysHealth: LLM sharding and compilation latency: 845.62 secs"
        actual = filter.filter(record).msg
        expected = "SysHealth.Count:1|#DJLServing:0.24.0,LLMShardingAndCompilationLatency:845.62 secs"
        self.assertEqual(actual.split('|')[0], expected.split('|')[0])
        self.assertEqual(actual.split(':')[1], expected.split(':')[1])
        self.assertEqual(actual.split(',')[1], expected.split(',')[1])

        record = MagicMock()
        record.msg = f"SysHealth: LLM sharding and compilation latency: 845.62 secs"
        actual = filter.filter(record).msg
        expected = "SysHealth.Count:2|#DJLServing:0.24.0,LLMShardingAndCompilationLatency:845.62 secs"
        self.assertEqual(actual.split('|')[0], expected.split('|')[0])
        self.assertEqual(actual.split(':')[1], expected.split(':')[1])
        self.assertEqual(actual.split(',')[1], expected.split(',')[1])

    def test_filter_warning(self):
        filter = SMLogFilter()
        record = MagicMock()
        record.msg = f"SysHealth: LLM sharding and compilation latency: 845.62 : secs"
        actual = filter.filter(record)

        with self.assertLogs(level=logging.WARNING):
            filter.filter(record)

    def test_filter_miss(self):
        filter = SMLogFilter()
        record = MagicMock()
        record.msg = f"LLM sharding and compilation latency: 845.62 : secs"
        actual = filter.filter(record)
        self.assertFalse(actual)

