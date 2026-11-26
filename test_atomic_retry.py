"""Preserve the previous artifact on persistent errors; retry only Windows locks."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.runtime import atomic_json


class AtomicRetryTests(unittest.TestCase):
    def test_transient_windows_lock_eventually_replaces(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'comparison.json';path.write_text('{"old":true}')
            original=os.replace;calls=[]
            def locked_then_replace(source,target):
                calls.append(1)
                if len(calls)<3:
                    self.assertEqual(json.loads(path.read_text()),{'old':True})
                    error=PermissionError('simulated transient Windows lock');error.winerror=5
                    raise error
                return original(source,target)
            with patch('utils.runtime.os.replace',side_effect=locked_then_replace),patch('utils.runtime.time.sleep'):
                atomic_json(path,{'new':True})
            self.assertEqual(len(calls),3)
            self.assertEqual(json.loads(path.read_text()),{'new':True})

    def test_permanent_lock_keeps_old_artifact_and_completed_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'comparison.json';path.write_text('{"old":true}')
            error=PermissionError('simulated persistent Windows lock');error.winerror=32
            with patch('utils.runtime.os.replace',side_effect=error) as replace,patch('utils.runtime.time.sleep'):
                with self.assertRaises(PermissionError):atomic_json(path,{'new':True})
            self.assertEqual(replace.call_count,6)
            self.assertEqual(json.loads(path.read_text()),{'old':True})
            self.assertEqual(json.loads(path.with_suffix('.json.tmp').read_text()),{'new':True})

    def test_other_permission_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('utils.runtime.os.replace',side_effect=PermissionError('not a Windows sharing error')) as replace:
                with self.assertRaises(PermissionError):atomic_json(Path(folder)/'file.json',{'x':1})
            self.assertEqual(replace.call_count,1)


if __name__=='__main__':unittest.main(verbosity=2)
