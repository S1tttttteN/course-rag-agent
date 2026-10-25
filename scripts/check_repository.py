"""Bounded credential checks for tracked/new files and reachable Git blobs.

Only locations and rule names are printed. No matching credential is logged.
This is a supplementary check, not a comprehensive security or rights audit.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RULES = {
    'private key': re.compile(rb'-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----'),
    'GitHub token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'AWS access key': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'API token': re.compile(rb'\bsk-[A-Za-z0-9_-]{24,}\b'),
    'credential in URL': re.compile(rb'https?://[^\s/@:]+:[^\s/@]+@'),
}


def sensitive_path(name):
    path = PurePosixPath(name.replace('\\', '/'))
    return (
        (path.name == '.env' or path.name.startswith('.env.')) and path.name != '.env.example'
        or 'data' in path.parts[:-1]
        or path.name in {'id_rsa', 'id_ed25519', 'id_ecdsa'}
        or path.suffix.lower() in {'.p12', '.pfx', '.key', '.pem', '.sqlite', '.sqlite3'}
    )


def local_credentials():
    path = ROOT / '.env'
    values = set()
    if path.exists():
        # Only --local-env needs the backend dependency; CI's history scan stays stdlib-only.
        from dotenv import dotenv_values
        for key, value in dotenv_values(path, encoding='utf-8-sig', interpolate=False).items():
            value = (value or '').strip()
            if re.search(r'(API_KEY|TOKEN|SECRET|PASSWORD|PRIVATE_KEY)$', key) and len(value) >= 12 and '你的' not in value:
                values.add(value.encode())
    return values


def content_findings(data, credentials=()):
    hits = [name for name, pattern in RULES.items() if pattern.search(data)]
    if any(value in data for value in credentials):
        hits.append('local credential value')
    return hits


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def audit(history=False, compare_local=False):
    credentials = local_credentials() if compare_local else ()
    findings = []
    names = set(git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')) - {''}
    for name in sorted(names):
        path = ROOT / name
        if not path.is_file():  # Deleted files are checked in history below.
            continue
        if path.is_symlink():
            hits = ['symlink requires manual review']
        else:
            hits = content_findings(path.read_bytes(), credentials)
        if sensitive_path(name):
            hits.append('sensitive path')
        if hits:
            findings.append({'location': name, 'rules': hits})
    blobs = 0
    metadata = 0
    if history:
        # Stream one object at a time: never put all historical contents in logs or memory.
        objects = git('rev-list', '--objects', '--all').decode().splitlines()
        with subprocess.Popen(['git', 'cat-file', '--batch'], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
            for entry in objects:
                oid, _, name = entry.partition(' ')
                process.stdin.write((oid + '\n').encode())
                process.stdin.flush()
                header = process.stdout.readline().decode().strip().split()
                if len(header) != 3:
                    raise RuntimeError('Cannot inspect a Git object')
                size = int(header[2])
                if size > 50_000_000:
                    raise RuntimeError('Large Git object requires manual review')
                data = process.stdout.read(size)
                process.stdout.read(1)
                if header[1] in {'commit', 'tag'}:
                    metadata += 1
                    hits = content_findings(data, credentials)
                    if hits:
                        findings.append({'location': f'{header[1]}:{oid[:12]}', 'rules': hits})
                    continue
                if header[1] != 'blob':
                    continue
                blobs += 1
                hits = content_findings(data, credentials)
                if sensitive_path(name):
                    hits.append('sensitive historical path')
                if hits:
                    findings.append({'location': f'{oid[:12]}:{name}', 'rules': hits})
            process.stdin.close()
    return {'worktree_paths': len(names), 'historical_blobs': blobs, 'historical_metadata_objects': metadata,
            'local_value_comparison': compare_local, 'findings': findings}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true', help='Scan objects reachable from all local Git refs')
    parser.add_argument('--local-env', action='store_true', help='Compare local .env credential values without printing them')
    arguments = parser.parse_args()
    result = audit(arguments.history, arguments.local_env)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(bool(result['findings']))
