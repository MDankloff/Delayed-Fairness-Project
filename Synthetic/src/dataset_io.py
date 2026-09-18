"""Read selected BAF columns, with an optional source-validated Parquet cache."""
import importlib.util
import json
from pathlib import Path
import os
import warnings

import pandas as pd

BAF_COLUMNS = ('fraud_bool', 'customer_age', 'income',
               'credit_risk_score', 'proposed_credit_limit')


def read_baf_dataset(path, columns=BAF_COLUMNS, use_cache=True):
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path, columns=list(columns))
    if path.suffix != '.csv':
        raise ValueError(f'Unsupported dataset format: {path.suffix}')
    if not use_cache or importlib.util.find_spec('pyarrow') is None:
        return pd.read_csv(path, usecols=list(columns)).loc[:, list(columns)]
    stat = path.stat()
    signature = dict(source=str(path.resolve()), size=stat.st_size,
                     mtime_ns=stat.st_mtime_ns, columns=list(columns), version=1)
    cache_dir = path.parent / '.baf_cache'
    parquet = cache_dir / f'{path.stem}.parquet'
    metadata = cache_dir / f'{path.stem}.json'
    if parquet.exists() and metadata.exists():
        try:
            if json.loads(metadata.read_text()) == signature:
                return pd.read_parquet(parquet, columns=list(columns), engine='pyarrow')
        except (OSError, ValueError):
            pass  # Rebuild an incomplete/stale cache from the source CSV.
    frame = pd.read_csv(path, usecols=list(columns)).loc[:, list(columns)]
    temporary = parquet.with_suffix(f'.{os.getpid()}.tmp.parquet')
    meta_tmp = metadata.with_suffix(f'.{os.getpid()}.tmp.json')
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(temporary, index=False, engine='pyarrow')
        meta_tmp.write_text(json.dumps(signature, indent=2))
        temporary.replace(parquet)
        meta_tmp.replace(metadata)
    except OSError as error:
        warnings.warn(f'Parquet cache could not be written; using CSV data: {error}', RuntimeWarning)
    finally:
        temporary.unlink(missing_ok=True)
        meta_tmp.unlink(missing_ok=True)
    return frame
