#!/usr/bin/env python3
"""User-level cold-start preprocessing: 8 train folds / 1 valid / 1 test.

"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

PathLike = Union[str, Path]
FORMAT_VERSION = 1


def _write_json(value: dict, path: Path) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def _mapping_to_file(mapping: dict, path: Path) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for key, value in mapping.items():
            handle.write(f'{key}\t{value}\n')


def cold_processed(
    seq_length: int,
    raw_path: PathLike = './data/App_usage_trace.txt',
    output_dir: PathLike = './data/cold_10fold',
    n_splits: int = 10,
    split_seed: int = 42,
    min_app_records: int = 5,
    min_user_samples: int = 50,
    gap_seconds: int = 300,
    force: bool = False,
    quiet: bool = False,
) -> list[Path]:
    """Create all user folds and return fold directories (zero-based numbering).

    Fold k: test = user_fold[k], valid = user_fold[(k+1) % n_splits],
    train = all remaining user folds. Fold membership is fixed across models.
    For n_splits=10, each user is tested once, validated once and trained eight
    times. n_splits other than 10 is supported for smoke tests only.
    """
    if seq_length < 1 or n_splits < 3 or min_app_records < 1 or min_user_samples < 1:
        raise ValueError('Positive lengths/counts and at least 3 folds are required.')
    if gap_seconds < 0:
        raise ValueError('gap_seconds cannot be negative.')
    raw_path, output_dir = Path(raw_path), Path(output_dir)
    if not raw_path.is_file():
        raise FileNotFoundError(f'Raw usage file not found: {raw_path}')

    stat = raw_path.stat()
    config = dict(
        format_version=FORMAT_VERSION, raw_name=raw_path.name,
        raw_size=stat.st_size, raw_mtime_ns=stat.st_mtime_ns,
        seq_length=int(seq_length), n_splits=int(n_splits), split_seed=int(split_seed),
        min_app_records=int(min_app_records), min_user_samples=int(min_user_samples),
        gap_seconds=int(gap_seconds), hour_source='last_observed_event',
        user_feature='constant_zero', vocabulary='closed_corpus_inputs_and_targets',
    )
    manifest_path = output_dir / 'manifest.json'
    fold_dirs = [output_dir / f'fold_{k:02d}' for k in range(n_splits)]
    required = ['train.txt', 'valid.txt', 'test.txt', 'app2id.txt',
                'user2id.txt', 'users.json', 'stats.json']
    if manifest_path.exists() and not force:
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous['config'] != config:
            raise ValueError('Existing preprocessing configuration differs. Use a new '
                             'output directory or force=True / --force_preprocess.')
        if all((d / name).is_file() for d in fold_dirs for name in required):
            if not quiet:
                print(f'Using existing user folds: {output_dir}')
            return fold_dirs
        raise ValueError('Incomplete cached preprocessing configuration. Rebuild with force=True.')
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise ValueError('Output directory is nonempty without a reusable configuration; '
                         'use a new directory or force=True.')
    output_dir.mkdir(parents=True, exist_ok=True)
    # Only remove the completion marker; never recursively delete user files.
    if manifest_path.exists():
        manifest_path.unlink()

    df = pd.read_csv(
        raw_path, sep=r'\s+', header=None,
        names=['user', 'time', 'location', 'app', 'traffic'],
        usecols=['user', 'time', 'app', 'traffic'],
        dtype={'user': 'int64', 'time': str, 'app': 'int64', 'traffic': 'float64'},
    )
    if df.empty or df.isna().any().any():
        raise ValueError('Input is empty or contains missing user/time/app/traffic fields.')
    if not df['time'].str.fullmatch(r'\d{14}').all():
        raise ValueError('Expected timestamps in YYYYMMDDHHMMSS format.')

    # Preserve the original minute-resolution preprocessing. This removes exact
    # duplicate rows; it is NOT traffic aggregation or merging by app alone.
    df['time'] = df['time'].str[:-2]
    df = df.drop_duplicates(subset=['user', 'time', 'app', 'traffic'])
    df = df[df.groupby('app')['app'].transform('count').ge(min_app_records)].copy()
    if df.empty:
        raise ValueError('No records remain after the corpus-level app filter.')
    df['timestamp'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M', errors='raise')
    df = df.sort_values(['user', 'timestamp'], kind='stable').reset_index(drop=True)

    # The original code normalized traffic but never wrote traffic_seq to output.
    # Preserve its actual feature interface instead of inventing a duration feature
    # or fitting an unused normalization statistic on held-out users.
    rows = []
    previous_user, previous_time = None, None
    apps, times = [], []
    iterator = df[['user', 'app', 'timestamp']].itertuples(index=False, name=None)
    for user, app, timestamp in tqdm(iterator, total=len(df),
                                     desc='Building windows', disable=quiet):
        same_session = (user == previous_user and
                        (timestamp - previous_time).total_seconds() <= gap_seconds)
        if not same_session:
            apps, times = [int(app)], [timestamp]
        else:
            if len(apps) == seq_length:
                # Use the LAST OBSERVED event's time, not the target's time.
                rows.append((int(user), apps.copy(),
                             [(previous_time - t).total_seconds() // 60 for t in times],
                             int(app), f'{previous_time.weekday()}_{previous_time.hour}'))
                apps = apps[1:] + [int(app)]
                times = times[1:] + [timestamp]
            else:
                apps.append(int(app))
                times.append(timestamp)
        previous_user, previous_time = user, timestamp
    del df

    processed = pd.DataFrame(rows, columns=['user_id', 'app_seq', 'time_seq', 'next_app', 'time'])
    del rows
    if processed.empty:
        raise ValueError('No windows were produced. Check seq_length and session-gap settings.')
    # As in the supplied code: threshold counts prediction WINDOWS, not raw rows.
    processed = processed[processed.groupby('user_id')['user_id'].transform('count')
                          .ge(min_user_samples)].copy().reset_index(drop=True)
    users = np.sort(processed['user_id'].unique())
    if len(users) < n_splits:
        raise ValueError(f'Only {len(users)} eligible users; at least {n_splits} users required.')

    # Include target-only apps. A missing target must never silently become app 0.
    app_set = set(processed['next_app'].tolist())
    for sequence in processed['app_seq']:
        app_set.update(sequence)
    app2id = {int(app): i for i, app in enumerate(sorted(app_set))}
    processed['app_seq'] = processed['app_seq'].apply(lambda seq: [app2id[a] for a in seq])
    processed['app'] = processed['next_app'].map(app2id).astype('int64')
    processed['user'] = 0  # compatibility field, NOT a user identity
    _mapping_to_file(app2id, output_dir / 'app2id.txt')

    # KFold is applied to unique USERS, never to individual windows.
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    user_folds = [users[idx] for _, idx in splitter.split(users)]
    user_to_fold = {int(u): k for k, members in enumerate(user_folds) for u in members}
    fold_index = processed['user_id'].map(user_to_fold)
    pd.DataFrame(sorted(user_to_fold.items()), columns=['user_id', 'user_fold']) \
        .to_csv(output_dir / 'user_folds.csv', index=False)

    stats_rows = []
    output_columns = ['app_seq', 'time_seq', 'next_app', 'time', 'user', 'app']
    for k, fold_dir in enumerate(fold_dirs):
        fold_dir.mkdir(parents=True, exist_ok=True)
        v = (k + 1) % n_splits
        masks = {'train': (fold_index != k) & (fold_index != v),
                 'valid': fold_index == v, 'test': fold_index == k}
        members = {
            'train': [int(u) for u in users if user_to_fold[int(u)] not in (k, v)],
            'valid': [int(u) for u in user_folds[v]],
            'test': [int(u) for u in user_folds[k]],
        }
        train_u, valid_u, test_u = map(set, (members['train'], members['valid'], members['test']))
        if train_u & valid_u or train_u & test_u or valid_u & test_u:
            raise RuntimeError('Internal error: overlapping user partitions.')
        if train_u | valid_u | test_u != set(map(int, users)):
            raise RuntimeError('Internal error: incomplete user coverage.')

        # Existing readers expect user2id.txt; one constant slot is intentional.
        _mapping_to_file({0: 0}, fold_dir / 'user2id.txt')
        _mapping_to_file(app2id, fold_dir / 'app2id.txt')
        _write_json(members, fold_dir / 'users.json')
        stats = dict(fold_id=k, valid_user_fold=v, num_apps=len(app2id),
                     num_model_users=1)
        train_targets = set(processed.loc[masks['train'], 'app'])
        train_inputs = set()
        for sequence in processed.loc[masks['train'], 'app_seq']:
            train_inputs.update(sequence)
        observed_apps = train_targets | train_inputs
        for split, mask in masks.items():
            part = processed.loc[mask, output_columns]
            if part.empty:
                raise ValueError(f'Empty {split} data in fold {k}.')
            part.to_csv(fold_dir / f'{split}.txt', sep='\t', index=False)
            stats[f'{split}_users'] = len(members[split])
            stats[f'{split}_samples'] = len(part)
            # Diagnostics only. Do NOT delete these evaluation cases.
            stats[f'{split}_targets_not_in_train_targets'] = int((~part['app'].isin(train_targets)).sum())
            stats[f'{split}_targets_not_observed_in_train'] = int((~part['app'].isin(observed_apps)).sum())
        _write_json(stats, fold_dir / 'stats.json')
        stats_rows.append(stats)
        if not quiet:
            print(f'Fold {k:02d} | users train/valid/test: '
                  f"{stats['train_users']}/{stats['valid_users']}/{stats['test_users']} | "
                  f"samples: {stats['train_samples']}/{stats['valid_samples']}/{stats['test_samples']}")

    pd.DataFrame(stats_rows).to_csv(output_dir / 'fold_statistics.csv', index=False)
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    _write_json(dict(config=config, fingerprint=fingerprint, num_users=len(users),
                     num_samples=len(processed), num_apps=len(app2id),
                     notes=['Corpus-level cohort filtering is retained.',
                            'Closed app catalog includes all eligible inputs and targets.',
                            'time_seq contains elapsed minutes, not duration.',
                            'No traffic feature is exported, matching the original CSV interface.',
                            'User IDs are stored only in audit files; model-facing user is zero.']),
                manifest_path)
    return fold_dirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seq_length', type=int, default=8)
    parser.add_argument('--raw_file', default='./data/App_usage_trace.txt')
    parser.add_argument('--data_root', default='./data/cold_10fold')
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--min_app_records', type=int, default=5)
    parser.add_argument('--min_user_samples', type=int, default=50)
    parser.add_argument('--gap_seconds', type=int, default=300)
    parser.add_argument('--force_preprocess', action='store_true')
    args = parser.parse_args()
    cold_processed(args.seq_length, args.raw_file, args.data_root, args.n_splits,
                   args.split_seed, args.min_app_records, args.min_user_samples,
                   args.gap_seconds, force=args.force_preprocess)
