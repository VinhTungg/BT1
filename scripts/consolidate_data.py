"""Consolidate the active, already cleaned split without changing its benchmark.

Run once to move historical data under archive/, or use --check for read-only checks.
The merged CSV is for managing the corpus, never for fitting the benchmark model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import clean_text

FIELDS = ['id', 'post_id', 'text', 'text_clean', 'text_tokenized', 'label']
SPLITS = ('train', 'val', 'test')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(path: str | Path) -> Path:
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT.resolve()) or result == ROOT.resolve():
        raise ValueError(f'Path must stay within the project: {path}')
    return result


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f'Missing training columns: {path}')
        return list(reader)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate(rows_by_split: dict[str, list[dict]]) -> dict:
    """Reject ambiguous data rather than guessing labels or changing held-out rows."""
    seen = {key: {} for key in ('id', 'text_clean', 'text_tokenized')}
    posts = {}
    summary = {}
    for split, rows in rows_by_split.items():
        if not rows:
            raise ValueError(f'Empty split: {split}')
        for row in rows:
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in FIELDS):
                raise ValueError(f'Empty or missing field: {split}/{row.get("id")}')
            if row['label'] not in ('0', '1'):
                raise ValueError(f'Invalid label: {row["id"]}')
            if clean_text(row['text']) != row['text_clean']:
                raise ValueError(f'Stale text_clean: {row["id"]}')
            if clean_text(row['text_clean']) != row['text_clean'] or clean_text(row['text_tokenized']) != row['text_tokenized']:
                raise ValueError(f'Non-normalized model input: {row["id"]}')
            for key in seen:
                value = clean_text(row[key]).casefold() if key != 'id' else row[key]
                if value in seen[key]:
                    previous = seen[key][value]
                    reason = 'Conflicting labels' if previous['label'] != row['label'] else 'Duplicate'
                    raise ValueError(f'{reason} on {key}: {previous["id"]}, {row["id"]}')
                seen[key][value] = row
            # Group prefixes can differ even though the Facebook post number is identical.
            post = row['post_id'].rsplit(':', 1)[-1]
            if post in posts and posts[post] != split:
                raise ValueError(f'Post leakage: {post}')
            posts[post] = split
        labels = Counter(r['label'] for r in rows)
        if set(labels) != {'0', '1'}:
            raise ValueError(f'Both labels required: {split}')
        summary[split] = {'samples': len(rows), 'non_toxic': labels['0'], 'toxic': labels['1'],
                          'posts': len({r['post_id'] for r in rows})}
    return summary


def check() -> dict:
    pointer = json.loads(local('data/current_dataset.json').read_text(encoding='utf-8'))
    manifest_path = local(pointer['manifest'])
    if sha(manifest_path) != pointer['manifest_sha256']:
        raise ValueError('Active manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows = {}
    for split in SPLITS:
        metadata = manifest['splits'][split]
        path = local(metadata['path'])
        if sha(path) != metadata['sha256']:
            raise ValueError(f'{split} hash mismatch')
        rows[split] = read_csv(path)
        if len(rows[split]) != metadata['samples']:
            raise ValueError(f'{split} count mismatch')
    summary = validate(rows)
    merged_meta = manifest.get('merged')
    if merged_meta:
        path = local(merged_meta['path'])
        if sha(path) != merged_meta['sha256']:
            raise ValueError('Merged CSV hash mismatch')
        merged = read_csv(path)
        actual = sorted(tuple(r[k] for k in FIELDS) for part in rows.values() for r in part)
        if sorted(tuple(r[k] for k in FIELDS) for r in merged) != actual:
            raise ValueError('Merged CSV differs from the split union')
        provenance_meta = manifest['provenance']
        if sha(local(provenance_meta['path'])) != provenance_meta['sha256']:
            raise ValueError('Provenance hash mismatch')
        with local(provenance_meta['path']).open(encoding='utf-8-sig', newline='') as stream:
            provenance = list(csv.DictReader(stream))
        expected = {(r['id'], split) for split, part in rows.items() for r in part}
        if len(provenance) != len(expected) or {(p['id'], p['split']) for p in provenance} != expected:
            raise ValueError('Provenance IDs or split assignments differ from the corpus')
        if dict(Counter(p['label_origin'] for p in provenance)) != provenance_meta['counts']:
            raise ValueError('Label provenance counts differ from the manifest')
    return {'dataset_id': pointer['dataset_id'], 'splits': summary,
            'total_samples': sum(len(part) for part in rows.values()), 'checks_passed': True}


def consolidate() -> dict:
    current = local('data/current_dataset.json')
    pointer = json.loads(current.read_text(encoding='utf-8'))
    if pointer['manifest'] == 'data/clean/manifest.json':
        return check()
    check()  # Complete all data checks before any move.
    old_manifest = local(pointer['manifest'])
    manifest = json.loads(old_manifest.read_text(encoding='utf-8'))
    rows_by_split = {s: read_csv(local(manifest['splits'][s]['path'])) for s in SPLITS}
    merged = sorted((r for part in rows_by_split.values() for r in part), key=lambda r: r['id'])
    old_merged = read_csv(old_manifest.parent / 'all_labeled.csv')
    if sorted(tuple(r[k] for k in FIELDS) for r in old_merged) != sorted(tuple(r[k] for k in FIELDS) for r in merged):
        raise ValueError('Previous merged corpus differs from active splits')
    ai_rows = {r['id']: r for r in read_csv(local('data/expanded/round_2/new_comments_labeled.csv'))}
    human_rows = {}
    for path in local('annotations').glob('thanh_vien_*.csv'):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                human_rows[row['id']] = (row, path.relative_to(ROOT).as_posix())
    provenance = []
    for split, rows in rows_by_split.items():
        for row in rows:
            if row['id'] in ai_rows:
                source = ai_rows[row['id']]
                kind = 'ai_silver'
                source_file = 'data/expanded/round_2/new_comments_labeled.csv'
            elif row['id'] in human_rows:
                source, source_file = human_rows[row['id']]
                kind = 'original_approved'
            else:
                raise ValueError(f'Unknown label provenance: {row["id"]}')
            if source['label'] != row['label']:
                raise ValueError(f'Source label mismatch: {row["id"]}')
            provenance.append({'id': row['id'], 'split': split, 'label_origin': kind,
                               'source_file_before_archive': source_file,
                               'reviewer': source.get('reviewer', ''),
                               'decision_file': source.get('decision_file', '')})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    archive = local(f'archive/data_before_consolidation_{stamp}')
    clean = local('data/clean')
    if clean.exists() or archive.exists():
        raise ValueError('Destination already exists; refusing to overwrite')
    # Resolve and validate every recursive move before creating destinations.
    move_names = ['data/splits', 'data/expanded', 'data/incoming', 'data/experiments',
                  'data/manifest.json', 'data/dataset_revision.json']
    moves = [(local(name), local(archive.relative_to(ROOT) / name)) for name in move_names]
    backup_names = ['data/current_dataset.json', 'data/README.md', 'models/current_run.json',
                    'reports/training_run.json']
    archive.mkdir(parents=True)
    for name in backup_names:
        source = local(name)
        if source.exists():
            destination = local(archive.relative_to(ROOT) / name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    report = local('reports/data_cleanup')
    report.mkdir(parents=True, exist_ok=True)
    clean.mkdir()
    for split in SPLITS:
        metadata = manifest['splits'][split]
        shutil.copy2(local(metadata['path']), clean / f'{split}.csv')
        metadata['path'] = f'data/clean/{split}.csv'
    shutil.copy2(old_manifest.parent / 'all_labeled.csv', clean / 'all_labeled.csv')
    provenance_path = report / 'provenance.csv'
    with provenance_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(provenance[0]))
        writer.writeheader()
        writer.writerows(provenance)
    manifest['merged'] = {'path': 'data/clean/all_labeled.csv', 'sha256': sha(clean / 'all_labeled.csv'),
                          'samples': len(merged), 'usage': 'Corpus management only; use train.csv for training.'}
    manifest['provenance'] = {'path': 'reports/data_cleanup/provenance.csv', 'sha256': sha(provenance_path),
                              'counts': dict(Counter(p['label_origin'] for p in provenance))}
    manifest['consolidation'] = {'archive': archive.relative_to(ROOT).as_posix(),
                                 'previous_manifest': str(old_manifest.relative_to(ROOT)),
                                 'split_bytes_and_assignments_unchanged': True,
                                 'additional_rows_removed': 0, 'schema': FIELDS}
    write_json(clean / 'manifest.json', manifest)
    write_json(archive / 'relocation.json', {str(s.relative_to(ROOT)): str(d.relative_to(ROOT)) for s, d in moves})
    moved = []
    try:
        for source, destination in moves:
            if source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                moved.append((source, destination))
        pointer.update(manifest='data/clean/manifest.json', manifest_sha256=sha(clean / 'manifest.json'))
        write_json(current, pointer)
        result = check()
    except Exception:
        for source, destination in reversed(moved):
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(source))
        shutil.copy2(archive / 'data/current_dataset.json', current)
        raise
    for name in ('models/current_run.json', 'reports/training_run.json'):
        path = local(name)
        metadata = json.loads(path.read_text(encoding='utf-8'))
        metadata.update(dataset_manifest=pointer['manifest'], dataset_manifest_sha256=pointer['manifest_sha256'],
                        train_path='data/clean/train.csv', data_relocated_without_retraining=True)
        write_json(path, metadata)
    result.update(archive=archive.relative_to(ROOT).as_posix(),
                  additional_rows_removed=0, label_origins=manifest['provenance']['counts'])
    write_json(report / 'audit.json', result)
    (archive / 'README.md').write_text(
        '# Dữ liệu trước khi gom\n\nGiữ nguyên byte và nhãn của các phiên bản cũ. '
        'Bộ đang dùng nằm tại `data/clean/`. Bảng `relocation.json` ánh xạ đường dẫn cũ sang bản lưu. '
        'Các báo cáo lịch sử ghi đường dẫn tại thời điểm chạy; muốn tái lập lượt cũ, dùng bản lưu '
        'hoặc khôi phục các đường dẫn theo bảng này. Không gộp CSV trong archive vào train.\n', encoding='utf-8')
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Read-only validation of the active corpus')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(check() if args.check else consolidate(), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        print(f'Data consolidation failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
