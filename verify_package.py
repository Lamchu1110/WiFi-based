"""Verify packaged source checksums using only the Python standard library."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
    errors = []
    for item in manifest['files']:
        path = root/item['path']
        if not path.is_file():
            errors.append('Missing: '+item['path'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            errors.append('Changed: '+item['path'])
    for error in errors:
        print(error)
    print(f"Checked {len(manifest['files'])} source/document files: {'FAIL' if errors else 'PASS'}")
    return int(bool(errors))


if __name__=='__main__':
    raise SystemExit(main())
