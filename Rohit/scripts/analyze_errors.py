"""Reproduce the report's error counts and review-length analysis."""
import argparse
from pathlib import Path
from rohit_dl.io import read_jsonl, save_json
from rohit_dl.text import tokenize


def analyze(suite):
    result = {}
    for condition in ('clean', 'ood'):
        path = Path(suite)/condition/'predictions.jsonl'
        if not path.exists():
            continue
        rows = read_jsonl(path)
        bins = {name: {'n': 0, 'correct': 0} for name in
                ('up_to_100_tokens', '101_to_400_tokens', 'over_400_tokens')}
        for row in rows:
            length = len(tokenize(row['text']))
            key = 'up_to_100_tokens' if length <= 100 else '101_to_400_tokens' if length <= 400 else 'over_400_tokens'
            bins[key]['n'] += 1
            bins[key]['correct'] += row['label'] == row['prediction']
        for stats in bins.values():
            stats['accuracy'] = stats['correct']/stats['n'] if stats['n'] else None
        wrong = [row for row in rows if row['label'] != row['prediction']]
        result[condition] = {
            'by_original_token_length': bins,
            'false_positive_count': sum(row['prediction'] == 1 for row in wrong),
            'false_negative_count': sum(row['prediction'] == 0 for row in wrong)}
    if not result:
        raise ValueError('No clean or OOD predictions found')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='runs/robustness_seed42')
    parser.add_argument('--output', default='reports/textcnn_seed42/error_summary.json')
    args = parser.parse_args()
    save_json(args.output, analyze(args.suite))
