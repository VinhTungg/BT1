"""Audit AI decisions and prepare a versioned train with 6000 usable new texts.

The audit-only mode is safe while collection appends new rows. It never
creates a training set. Validation/test are checked against their frozen
fingerprints and are not rewritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collect_comments import FIELDS as RAW_FIELDS, canonical_url, normalize_text, post_key
from src.post_discovery import canonical_post, comment_count, read_groups
from src.preprocessing import clean_text, segment_vietnamese

TRAIN_FIELDS = ['id', 'post_id', 'text', 'text_clean', 'text_tokenized', 'label']
MINIMUM_NEW = 6000
DECIDED_STATUSES = {'ai_semantic_decided', 'ai_context_reviewed', 'human_approved'}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv_bytes(data: bytes, required: set[str], name: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
    if not reader.fieldnames or required - set(reader.fieldnames):
        raise ValueError(f'{name}: missing required columns')
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f'{name}: malformed CSV row')
    return rows


def raw_snapshot(path: Path) -> tuple[bytes, list[dict]]:
    data = path.read_bytes()
    # Collector normalizes message newlines before writing each CSV row.
    # Discard only a partial final write; do not read an incomplete record.
    if not data.endswith(b'\n'):
        data = data[:data.rfind(b'\n') + 1]
    rows = read_csv_bytes(data, set(RAW_FIELDS), str(path))
    if not rows:
        raise ValueError('No complete collected rows')
    return data, rows


def load_frozen(manifest_path: Path, root: Path = ROOT) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    splits = {}
    for name in ('train', 'val', 'test'):
        metadata = manifest['splits'][name]
        path = root / metadata['path']
        data = path.read_bytes()
        if sha256(data) != metadata['sha256']:
            raise ValueError(f'Frozen {name} fingerprint changed: {path}')
        rows = read_csv_bytes(data, set(TRAIN_FIELDS), str(path))
        if len(rows) != metadata['samples'] or any(row['label'] not in {'0', '1'} for row in rows):
            raise ValueError(f'Frozen {name} schema/count/labels changed')
        splits[name] = rows
    for name, expected in manifest.get('annotation_sha256', {}).items():
        if sha256((root / name).read_bytes()) != expected:
            raise ValueError(f'Approved annotation changed: {name}')
    return manifest, splits


def read_decisions(paths: list[Path], root: Path = ROOT) -> dict[str, dict]:
    decisions = {}
    for path in paths:
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get('id') or row['id'] in decisions:
                raise ValueError(f'{path}:{number}: missing or duplicate decision id')
            label = row.get('label')
            if label is None:
                if row.get('review_status') != 'needs_context':
                    raise ValueError(f'{path}:{number}: unresolved label must need context')
            elif type(label) is not int or label not in (0, 1) or row.get('review_status') not in DECIDED_STATUSES:
                raise ValueError(f'{path}:{number}: invalid label/review status')
            if not row.get('reason') or not row.get('reviewer'):
                raise ValueError(f'{path}:{number}: missing label provenance')
            if row.get('context_file'):
                context = (root / str(row['context_file']).replace('\\', '/')).resolve()
                if not context.is_relative_to(root.resolve()):
                    raise ValueError('Context evidence must be within the project')
                context_data = context.read_bytes()
                if sha256(context_data) != row.get('context_sha256'):
                    raise ValueError(f'Context evidence changed for {row["id"]}')
                evidence = json.loads(context_data)
                if (not isinstance(evidence, dict) or not row.get('post_id')
                        or not row.get('source_url')
                        or evidence.get('post_id') != row['post_id']
                        or evidence.get('source_url') != row['source_url']):
                    raise ValueError(f'Context source differs for {row["id"]}')
            decisions[row['id']] = {**row, '_decision_file': str(path)}
    return decisions


def read_sources(path: Path, groups_path: Path) -> dict[str, dict]:
    groups = {url.split('/groups/', 1)[1].strip('/') for url in read_groups(groups_path)}
    rows = read_csv_bytes(path.read_bytes(), {'post_id', 'url', 'observed_comments', 'count_evidence'}, str(path))
    proof = {}
    for row in rows:
        post = canonical_post(row['url'], groups)
        count = int(row['observed_comments'])
        if (post is None or post[0] != row['post_id'] or count < 10
                or comment_count(row['count_evidence']) != count or post[0] in proof):
            raise ValueError(f'Invalid/duplicate source proof: {row["post_id"]}')
        proof[post[0]] = row
    return proof


def select_ready(raw: list[dict], decisions: dict[str, dict], proof: dict[str, dict],
                 old: list[dict], *, tokenizer=segment_vietnamese) -> tuple[list[dict], list[dict], dict]:
    """Exclude unresolved, duplicate, conflicting, and leaking new text."""
    old_posts = {row['post_id'].rsplit(':', 1)[-1] for row in old}
    old_ids = {row['id'] for row in old}
    old_clean = {clean_text(row['text_clean']).casefold() for row in old}
    old_tokens = {normalize_text(row['text_tokenized']).casefold() for row in old}
    status, indexed, candidates = {}, {}, {}
    for row in raw:
        identifier = row['id']
        if not identifier or identifier in indexed or identifier in old_ids:
            raise ValueError('Collected comment id missing, duplicate, or from old data')
        if sha256(normalize_text(row['text']).encode())[:24] != identifier:
            raise ValueError(f'Collected text/id no longer match: {identifier}')
        source = canonical_url(row['source_url'])
        if post_key(source) != row['post_id'] or row.get('label'):
            raise ValueError(f'Raw source/label changed: {identifier}')
        post = row['post_id'].rsplit(':', 1)[-1]
        if post in old_posts or post not in proof:
            raise ValueError(f'Old or unverified source post: {post}')
        indexed[identifier] = row
        status[identifier] = 'unreviewed'
        decision = decisions.get(identifier)
        if decision is None:
            continue
        if (decision.get('post_id') != row['post_id']
                or canonical_url(decision.get('source_url', '')) != source):
            raise ValueError(f'Decision/source mismatch: {identifier}')
        if decision['label'] is None:
            status[identifier] = 'needs_context'
            continue
        # Facebook can expose a scheme-relative URL as text rather than an anchor.
        cleaned = clean_text(re.sub(r'(?<!\S)//(?:[a-z0-9-]+\.)+[a-z]{2,}(?:[/?#]\S*)?',
                                    '[URL]', row['text'], flags=re.I))
        linguistic = re.sub(r'\[(?:MENTION|URL)\]', '', cleaned)
        linguistic = re.sub(r'^\s*(?:Người tham gia ẩn danh|Anonymous participant)(?:\s+\d+)?\b',
                            '', linguistic, flags=re.I)
        if not any(char.isalpha() for char in linguistic):
            status[identifier] = 'no_linguistic_content'
            continue
        if cleaned.casefold() in old_clean:
            status[identifier] = 'old_text_overlap'
            continue
        candidates[identifier] = {**row, 'text_clean': cleaned, 'label': str(decision['label']),
                                 'label_reason': decision['reason'], 'review_status': decision['review_status'],
                                 'reviewer': decision['reviewer'], 'decision_file': decision.get('_decision_file', ''),
                                 'context_file': decision.get('context_file', '')}
        status[identifier] = 'candidate'
    if set(decisions) - set(indexed):
        raise ValueError('Decisions reference comments absent from the collected snapshot')

    def deduplicate(field: str, conflict: str, duplicate: str) -> None:
        groups = defaultdict(list)
        for identifier, row in candidates.items():
            groups[normalize_text(row[field]).casefold()].append(identifier)
        for identifiers in groups.values():
            if len({candidates[key]['label'] for key in identifiers}) > 1:
                removed, reason = identifiers, conflict
            else:
                removed, reason = identifiers[1:], duplicate
            for identifier in removed:
                candidates.pop(identifier)
                status[identifier] = reason

    deduplicate('text_clean', 'conflicting_clean_labels', 'duplicate_clean')
    # The same long comment may be captured with an anchor marked URL or MENTION.
    # Restrict this extra normalization to one post; short replies retain their target.
    for identifier, row in candidates.items():
        reference_text = normalize_text(re.sub(r'\[(?:MENTION|URL)\]', '[REF]', row['text'], flags=re.I))
        row['_reference_key'] = f'{row["post_id"]}\n{reference_text}' if len(reference_text) >= 80 else identifier
    deduplicate('_reference_key', 'conflicting_reference_labels', 'duplicate_reference_variation')
    for row in candidates.values():
        row.pop('_reference_key')
    for identifier, row in list(candidates.items()):
        tokenized = tokenizer(row['text_clean'])
        if not isinstance(tokenized, str) or not tokenized.strip():
            status[identifier] = 'empty_model_input'
            candidates.pop(identifier)
        elif normalize_text(tokenized).casefold() in old_tokens:
            status[identifier] = 'old_model_input_overlap'
            candidates.pop(identifier)
        else:
            row['text_tokenized'] = tokenized
    deduplicate('text_tokenized', 'conflicting_token_labels', 'duplicate_tokens')
    for identifier in candidates:
        status[identifier] = 'ready'
    excluded = [{'id': identifier, 'post_id': indexed[identifier]['post_id'],
                 'text': indexed[identifier]['text'], 'source_url': indexed[identifier]['source_url'],
                 'status': value, 'reason': decisions.get(identifier, {}).get('reason', '')}
                for identifier, value in status.items() if value != 'ready']
    counts = dict(Counter(status.values()))
    assert sum(counts.values()) == len(raw)
    return list(candidates.values()), excluded, counts


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT/'data/incoming/comments_round_2.csv')
    parser.add_argument('--decisions', type=Path, action='append')
    parser.add_argument('--sources', type=Path, default=ROOT/'reports/discovery/round_2_all_candidates.csv')
    parser.add_argument('--groups', type=Path, default=ROOT/'data/incoming/groups.txt')
    parser.add_argument('--frozen-manifest', type=Path, default=ROOT/'data/manifest.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/expanded/round_2')
    parser.add_argument('--report-dir', type=Path, default=ROOT/'reports/data_expansion/dataset_audit')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        paths = args.decisions or [ROOT/'reports/data_expansion/label_decisions_first_100.jsonl',
                                  *sorted((ROOT/'reports/data_expansion/review_batches').glob('*.decisions.jsonl'))]
        raw_data, raw = raw_snapshot(args.raw)
        frozen, splits = load_frozen(args.frozen_manifest)
        decisions = read_decisions(paths)
        proof = read_sources(args.sources, args.groups)
        ready, excluded, counts = select_ready(raw, decisions, proof, sum(splits.values(), []))
        report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'raw_path': str(args.raw),
                  'raw_snapshot_sha256': sha256(raw_data), 'raw_snapshot_bytes': len(raw_data),
                  'raw_rows': len(raw), 'reviewed_rows': len(decisions), 'status_counts': counts,
                  'ready_new_rows': len(ready), 'minimum_required_new_rows': MINIMUM_NEW,
                  'ready_new_labels': dict(Counter(row['label'] for row in ready)),
                  'frozen_splits': {name: frozen['splits'][name] for name in ('train', 'val', 'test')},
                  'decision_files': {str(path): sha256(path.read_bytes()) for path in paths},
                  'source_proof_sha256': sha256(args.sources.read_bytes()),
                  'preprocessing_code_sha256': sha256((ROOT/'src/preprocessing.py').read_bytes()),
                  'preparation_code_sha256': sha256(Path(__file__).read_bytes()),
                  'underthesea_version': version('underthesea'),
                  'label_provenance': 'AI semantic/context review; not independently human-verified.',
                  'audit_only': args.audit_only}
        args.report_dir.mkdir(parents=True, exist_ok=True)
        (args.report_dir/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        write_csv(args.report_dir/'excluded.csv', ['id', 'post_id', 'text', 'source_url', 'status', 'reason'], excluded)
        context_review = [{**row, 'context_file': decisions.get(row['id'], {}).get('context_file', ''),
                           'decision_file': decisions.get(row['id'], {}).get('_decision_file', '')}
                          for row in excluded if row['status'] == 'needs_context'
                          or row['status'].startswith('conflicting_')]
        write_csv(args.report_dir/'can_ngu_canh.csv',
                  ['id', 'post_id', 'text', 'source_url', 'status', 'reason', 'context_file', 'decision_file'], context_review)
        print(json.dumps({key: report[key] for key in ('raw_rows', 'reviewed_rows', 'ready_new_rows', 'status_counts')}, ensure_ascii=False))
        if args.audit_only:
            return 0
        if len(ready) < MINIMUM_NEW:
            raise ValueError(f'Need {MINIMUM_NEW} usable new comments; currently {len(ready)}. No train created.')
        if args.output_dir.exists():
            raise ValueError('Output directory already exists; choose a new version')
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.prepare_expansion_', dir=args.output_dir.parent) as temporary:
            staging = Path(temporary)
            write_csv(staging/'train.csv', TRAIN_FIELDS, splits['train'] + ready)
            write_csv(staging/'new_comments_labeled.csv', RAW_FIELDS + ['text_clean', 'text_tokenized', 'label_reason', 'review_status', 'reviewer', 'decision_file', 'context_file'], ready)
            (staging/'raw_snapshot.csv').write_bytes(raw_data)
            (staging/'labels_snapshot.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in decisions.values()), encoding='utf-8')
            (staging/'sources_snapshot.csv').write_bytes(args.sources.read_bytes())
            load_frozen(args.frozen_manifest)  # Detect changes during preprocessing.
            report.update(audit_only=False, train_rows=len(splits['train']) + len(ready),
                          train_sha256=sha256((staging/'train.csv').read_bytes()),
                          fit_policy='New decided comments augment train only; validation/test retain frozen files.')
            (staging/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            staging.replace(args.output_dir)
        print(f'Prepared {len(ready)} new comments: {args.output_dir}')
        return 0
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as exc:
        print(f'Expansion preparation failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
