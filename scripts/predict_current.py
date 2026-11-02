"""Predict comments with the verified model selected by models/current_run.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.preprocessing import clean_text, segment_vietnamese


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('texts', nargs='+', help='Quote each comment, including an empty string.')
    args = parser.parse_args(argv)
    try:
        run = json.loads((ROOT/'models/current_run.json').read_text(encoding='utf-8'))
        metadata = next(item for item in run['models'] if item['model'] == run['best_on_validation'])
        path = ROOT/metadata['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
            raise ValueError('Current model artifact fingerprint changed')
        model = joblib.load(path)
        if metadata.get('input_contract') == 'raw_text':
            features = args.texts
        else:
            cleaned = [clean_text(text) for text in args.texts]
            tokenized = [segment_vietnamese(text) for text in cleaned]
            features = (pd.DataFrame({'text_clean': cleaned, 'text_tokenized': tokenized})
                        if 'features' in model.named_steps else tokenized)
        predictions = model.predict(features)
        for text, label in zip(args.texts, predictions):
            print(json.dumps({'text': text, 'label': int(label),
                              'label_name': 'toxic' if int(label) else 'non-toxic',
                              'model': metadata['model'], 'run_id': run['run_id']}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        print(f'Prediction failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
