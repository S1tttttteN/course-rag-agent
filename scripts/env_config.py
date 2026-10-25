"""Check the public template or align local .env layout without logging values.

Use the backend Python environment (python-dotenv is required).
"""
import argparse
from datetime import datetime, timezone
from io import StringIO
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from dotenv.parser import parse_stream


ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT = re.compile(r'^([A-Z][A-Z0-9_]*)=.*$', re.MULTILINE)
PRIVATE_FIELDS = re.compile(r'(?:API_KEY|TOKEN|SECRET|PASSWORD)$')
CHOICES = {'CHAT_PROVIDER', 'CHAT_MODEL', 'EMBEDDING_PROVIDER', 'EMBEDDING_MODEL'}


def read_values(text):
    values = {}
    for binding in parse_stream(StringIO(text)):
        if binding.error:
            raise ValueError(f'Invalid configuration syntax at line {binding.original.line}')
        if binding.key is None:
            continue
        if binding.key in values:
            raise ValueError(f'Duplicate configuration key: {binding.key}')
        values[binding.key] = binding.value
    return values


def quote(value):
    if not value:
        return ''
    if re.fullmatch(r'[A-Za-z0-9_./:@+,-]+', value):
        return value
    return "'" + value.replace('\\', '\\\\').replace("'", "\\'") + "'"


def render(template, values):
    defaults = read_values(template)
    unknown = sorted(set(values) - set(defaults))
    if unknown:
        raise ValueError('Unknown local keys require manual review; no values removed: ' + ', '.join(unknown))
    return ASSIGNMENT.sub(lambda match: match.group(1) + '=' + quote(values.get(match.group(1), defaults[match.group(1)])), template)


def template_issues(template):
    values = read_values(template)
    return [key for key, value in values.items() if value and (key in CHOICES or PRIVATE_FIELDS.search(key))]


def run(mode):
    template = (ROOT / '.env.example').read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    issues = template_issues(template)
    if issues:
        raise ValueError('Public template must leave model choices and credentials empty: ' + ', '.join(issues))
    if mode == 'template':
        return {'template': 'valid', 'fields': len(read_values(template))}, 0
    path = ROOT / '.env'
    if path.is_symlink():
        raise ValueError('Refusing to replace a symbolic-link .env')
    original = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n') if path.exists() else ''
    values = read_values(original)
    expected = render(template, values)
    # Parse before writing to confirm every existing value survives serialization.
    rendered_values = read_values(expected)
    if any(rendered_values[key] != (value or '') for key, value in values.items()):
        raise ValueError('Value preservation check failed; no file was changed')
    missing = sorted(set(read_values(template)) - set(values))
    result = {'layout_matches': original == expected, 'missing_keys': missing,
              'existing_values_preserved': len(values)}
    if mode == 'check':
        return result, int(original != expected)
    if original == expected:
        return result, 0
    if path.exists():
        backup_dir = ROOT / '.cache' / 'env-backups'
        if not backup_dir.resolve().is_relative_to(ROOT):
            raise ValueError('Backup directory must stay inside the workspace')
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / ('.env-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        shutil.copy2(path, backup)
        result['backup'] = str(backup.relative_to(ROOT))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', prefix='.env.', dir=ROOT, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(expected)
        if path.exists():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    result.update(layout_matches=True, missing_keys=[], synchronized=True)
    return result, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-template', action='store_const', const='template', dest='mode')
    group.add_argument('--check', action='store_const', const='check', dest='mode')
    group.add_argument('--sync', action='store_const', const='sync', dest='mode')
    args = parser.parse_args()
    try:
        report, code = run(args.mode)
    except ValueError as error:
        report, code = {'error': str(error)}, 1
    print(json.dumps(report, ensure_ascii=True, indent=2))
    raise SystemExit(code)
