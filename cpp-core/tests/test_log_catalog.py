"""The log catalog's gzip member index must not depend on the Python version that wrote it."""
import gzip
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import build_log_catalog as catalog

DOCS = Path(__file__).resolve().parents[1]/'docs'


class MemberIndexBytes(unittest.TestCase):
    def test_header_is_pinned(self):
        data = catalog.compress_members({'members': [{'path': 'stage1/a.json'}]})
        self.assertEqual(data[4:8], b'\0\0\0\0')
        self.assertEqual(data[9], 0xff)
        self.assertEqual(json.loads(gzip.decompress(data)), {'members': [{'path': 'stage1/a.json'}]})

    def test_committed_index_reproduces(self):
        committed = (DOCS/'log-archive-members.json.gz').read_bytes()
        self.assertEqual(catalog.compress_members(json.loads(gzip.decompress(committed))), committed)


if __name__ == '__main__':
    unittest.main()
