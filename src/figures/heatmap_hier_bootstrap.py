"""
Heatmap — Major Brain Regions — Hierarchical Bootstrap (matches distribution script)
=====================================================================================
Same pipeline as heatmap_drug_effects_baseline.py, but uses the identical
hierarchical bootstrap engine as distribution_major_regions_RT_STR_35_45.py:
  _sample_hier → resample sessions, then units within sessions.

This makes the heatmap asterisks directly comparable to the forest plot p-values.

Outputs (per metric × comparison × normalization):
  heatmap_{metric}_{comparison}_baseline_uncorrected_unitpooled.png
  heatmap_{metric}_{comparison}_t0_uncorrected_unitpooled.png

Usage
-----
  python heatmap_hier_bootstrap.py --base_dir output_prestim_burst_new
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import glob
import os
import argparse
import warnings
import multiprocessing as mp
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

METRICS = ['burst_fraction', 'burst_rate', 'mean_burst_size']

METRIC_LABELS = {
    'burst_fraction':  'Burst Fraction',
    'burst_rate':      'Burst Rate',
    'mean_burst_size': 'Mean Burst Size',
    'prop_bursting':   'Prop. Bursting Units',
}

# prop_bursting is computed on ALL units (including non-bursters in baseline)
# and uses no per-unit normalization — it is handled separately from METRICS
METRICS_RAW = ['prop_bursting']

CONDITION_COLORS = {
    'Saline':     '#030aa7',
    'Psilocybin': '#a90308',
    'Ketanserin': '#F3CA4B',
}

AREA_LABEL_COLORS = {
    'Isocortex': '#2ecc71',
    'Thalamus':  '#e91e8c',
    'OLF':       '#1abc9c',
    'HPF':       '#3498db',
    'STR':       '#e74c3c',
    'RT':        '#9b59b6',
}

COMPARISONS = [
    ('Psilocybin', 'Saline',    'Psilocybin_vs_Saline'),
    ('Ketanserin', 'Saline',    'Ketanserin_vs_Saline'),
]

MAJOR_REGION_ORDER = ['Isocortex', 'Thalamus', 'OLF', 'HPF', 'STR']

LAYER_NORMALIZE = {'1': '1/2/3', '2/3': '1/2/3', '6a': '6', '6b': '6'}

ISOCORTEX_ID = 315
TH_ID        = 549
OLF_ID       = 698
HPF_ID       = 1089
STR_ID       = 477
RT_ID        = 262
TH_ID        = 549
OLF_ID       = 698
HPF_ID       = 1089
STR_ID       = 477
RT_ID        = 262

# ============================================================================
# ALLEN SDK SETUP
# ============================================================================

_base_area_to_isocortex_child_cache = {}
_base_area_to_major_region_cache    = {}

ALLEN_SDK_AVAILABLE = False
structure_tree      = None


class ManualStructureTree:
    def __init__(self, structures):
        self.structures        = structures
        self.structures_by_id  = {s['id']: s for s in structures}
        self.structures_by_acr = defaultdict(list)
        for s in structures:
            acr = s.get('acronym')
            if acr:
                self.structures_by_acr[acr].append(s)

    def get_structures_by_acronym(self, acronyms):
        if isinstance(acronyms, str):
            acronyms = [acronyms]
        out = []
        for a in acronyms:
            out.extend(self.structures_by_acr.get(a, []))
        return out

    def get_structures_by_id(self, ids):
        if isinstance(ids, (int, np.integer)):
            ids = [ids]
        return [self.structures_by_id[i] for i in ids if i in self.structures_by_id]


try:
    print("Loading Allen Brain Atlas structure tree...")
    from allensdk.api.queries.ontologies_api import OntologiesApi
    oapi            = OntologiesApi()
    structure_graph = oapi.get_structures_with_sets([1])
    print(f"  Downloaded {len(structure_graph)} Allen structures")
    try:
        from allensdk.core.structure_tree import StructureTree
        structure_tree = StructureTree(structure_graph)
        if structure_tree.get_structures_by_acronym(['VPM']):
            ALLEN_SDK_AVAILABLE = True
            print("  Official Allen SDK StructureTree working")
        else:
            raise ValueError("empty result")
    except Exception as e:
        print(f"  Official StructureTree failed ({e}), using manual workaround")
        structure_tree = ManualStructureTree(structure_graph)
        if structure_tree.get_structures_by_acronym(['VPM']):
            ALLEN_SDK_AVAILABLE = True
            print(f"  Manual structure tree working")
        else:
            raise ValueError("manual tree failed too")
except Exception as e:
    print(f"  Allen SDK unavailable: {e}")
    ALLEN_SDK_AVAILABLE = False


def _get_path(structure):
    p = structure.get('structure_id_path', [])
    if isinstance(p, list):
        return [int(x) for x in p]
    elif isinstance(p, str):
        return [int(x) for x in p.strip('/').split('/') if x]
    return []


_ISOCORTEX_AREA_MAP = {
    'MOp': 'MO',  'MOs': 'MO',
    'SSp': 'SS',  'SSp-bfd': 'SS', 'SSp-tr': 'SS', 'SSp-ll': 'SS',
    'SSp-n': 'SS','SSp-m': 'SS',   'SSp-ul': 'SS', 'SSp-un': 'SS',
    'SSs': 'SS',
    'VISp': 'VIS','VISl': 'VIS',   'VISam': 'VIS', 'VISpm': 'VIS',
    'VISrl': 'VIS','VISa': 'VIS',  'VISal': 'VIS', 'VISli': 'VIS',
    'VISpor': 'VIS',
    'AUDp': 'AUD','AUDd': 'AUD',   'AUDv': 'AUD',  'AUDpo': 'AUD',
    'RSPagl': 'RSP','RSPd': 'RSP', 'RSPv': 'RSP',
    'ACAd': 'ACA','ACAv': 'ACA',
    'PL': 'PL',   'ILA': 'ILA',
    'ORBl': 'ORB','ORBm': 'ORB',   'ORBvl': 'ORB', 'ORBv': 'ORB',
    'AIp': 'AI',  'AIv': 'AI',     'AId': 'AI',
    'GU': 'GU',   'VISC': 'VISC',  'PTLp': 'PTLp',
    'TEa': 'TEa', 'ECT': 'ECT',    'PERI': 'PERI', 'FRP': 'FRP',
}

# RT now included (part of Thalamus)
_THALAMIC_PREFIXES = [
    'VPM','VPL','VAL','VM','PO','LP','LD','CL','MD',
    'MGM','MGD','MGV','LGD','LGV','IAD','IAM','POL','PT',
    'SPF','CM','PCN','IMD','PVT','PF','PIL','SGN',
    'AD','AM','AV','RE','RH','MG','LG','PP','SPA',
    'PoT','MGm','SPFp','SPFm','RT',
]

_OLF_FALLBACK = ['MOB','AOB','AON','TT','DP','PIR','NLOT','PAA','COA','OLF','EP','EPd','EPv']
_HPF_FALLBACK = ['CA1','CA2','CA3','DG','SUB','PRE','POST','PARA','FC','IG','HATA','APR','HPF']
_STR_FALLBACK = ['CP','ACB','FS','LSX','LS','CEA','BST','LA','BLA','BMA','AA','OT','MEA','SI','STR','isl','islm']


def _isocortex_area_fallback(base_area):
    if base_area in _ISOCORTEX_AREA_MAP:
        return _ISOCORTEX_AREA_MAP[base_area]
    for key, val in _ISOCORTEX_AREA_MAP.items():
        if base_area.startswith(key):
            return val
    return None


def _olf_hpf_str_fallback(base_area):
    loc = base_area.upper()
    for p in _OLF_FALLBACK:
        if loc == p.upper() or loc.startswith(p.upper()): return 'OLF'
    for p in _HPF_FALLBACK:
        if loc == p.upper() or loc.startswith(p.upper()): return 'HPF'
    for p in _STR_FALLBACK:
        if loc == p.upper() or loc.startswith(p.upper()): return 'STR'
    return None


def get_isocortex_area(base_area):
    if base_area in _base_area_to_isocortex_child_cache:
        return _base_area_to_isocortex_child_cache[base_area]
    result = None
    if ALLEN_SDK_AVAILABLE:
        structs = structure_tree.get_structures_by_acronym([base_area])
        if structs:
            path = _get_path(structs[0])
            if ISOCORTEX_ID in path:
                iso_idx = path.index(ISOCORTEX_ID)
                if iso_idx + 1 < len(path):
                    child = structure_tree.get_structures_by_id([path[iso_idx + 1]])
                    if child:
                        result = child[0]['acronym']
    if result is None:
        result = _isocortex_area_fallback(base_area)
    _base_area_to_isocortex_child_cache[base_area] = result
    return result


def get_major_region(base_area, brain_area):
    """UPDATED: RT and STR handled via brain_area column."""
    if brain_area == 'Motor':         return 'Isocortex'
    if brain_area == 'Somatosensory': return 'Isocortex'
    if brain_area == 'Thalamus':      return 'Thalamus'
    if brain_area == 'RT':            return 'Thalamus'
    if brain_area == 'STR':           return 'STR'
    if base_area in _base_area_to_major_region_cache:
        return _base_area_to_major_region_cache[base_area]
    result = None
    if get_isocortex_area(base_area) is not None:
        result = 'Isocortex'
    if result is None and ALLEN_SDK_AVAILABLE:
        structs = structure_tree.get_structures_by_acronym([base_area])
        if structs:
            path = _get_path(structs[0])
            if   OLF_ID       in path: result = 'OLF'
            elif HPF_ID       in path: result = 'HPF'
            elif STR_ID       in path: result = 'STR'
            elif ISOCORTEX_ID in path: result = 'Isocortex'
            elif TH_ID        in path: result = 'Thalamus'
    if result is None:
        result = _olf_hpf_str_fallback(base_area) or 'Other'
    _base_area_to_major_region_cache[base_area] = result
    return result


# ============================================================================
# LOCATION PARSING
# ============================================================================


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data(base_dir='.'):
    print("\nLoading data...")
    for candidate in [base_dir, '.', '..', 'output_prestim_burst_new4_silence50ms',
                      'output_prestim_burst_new4_silence50ms',
                      '../output_prestim_burst_new4_silence50ms', '../output_prestim_burst_new4_silence50ms']:
        if (glob.glob(f'{candidate}/psilocybin/*_timecourse.csv') or
                glob.glob(f'{candidate}/saline/*_timecourse.csv')):
            base_dir = candidate
            break
    else:
        raise FileNotFoundError(f"No data found under: {base_dir}")

    dfs = []
    for treatment, folder in [('Psilocybin', 'psilocybin'),
                               ('Saline',     'saline'),
                               ('Ketanserin', 'ketanserin')]:
        files = glob.glob(f'{base_dir}/{folder}/*_timecourse.csv')
        print(f"  {treatment}: {len(files)} files")
        for f in files:
            df = pd.read_csv(f)
            df['session_id'] = os.path.basename(f).replace(
                '_prestim_burst_timecourse.csv', '')
            df['treatment'] = treatment
            dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data['period'] = data['bin'].apply(
        lambda x: 'Baseline' if 'baseline' in str(x) else 'Effect')

    def _extract_time(b):
        return int(str(b).split('_')[1].split('-')[0])

    data['time_bin_start'] = data['bin'].apply(_extract_time)
    data['relative_time']  = data.apply(
        lambda r: -r['time_bin_start'] if r['period'] == 'Baseline'
                  else  r['time_bin_start'], axis=1)

    print(f"  Total rows : {len(data):,}")
    print(f"  Units      : {data[['session_id','unit_id']].drop_duplicates().shape[0]:,}")
    print(f"  Sessions   : {data['session_id'].nunique()}")
    for col in ['burst_fraction', 'burst_rate', 'mean_burst_size']:
        data[col] = data[col].where(data[col] > 0, np.nan)
    # Can't compute burst fraction from 0 spikes — recording artifact, not real zero
    if 'total_spikes' in data.columns:
        data.loc[data['total_spikes'] == 0, 'burst_fraction'] = np.nan
    return data
# ============================================================================
# LOCATION CLASSIFICATION — uses location string directly with Allen SDK
# ============================================================================

_LAYER_SUFFIXES = ['2/3', '6a', '6b', '1', '4', '5', '6']
_location_cache = {}


def _try_strip_layer(location):
    """Strip cortical layer suffix. Returns (base, raw_layer) or (location, None)."""
    for suffix in _LAYER_SUFFIXES:
        if str(location).endswith(suffix):
            base = location[:-len(suffix)]
            if len(base) >= 2:
                return base, suffix
    return location, None


def _query_allen(acronym):
    """
    Query Allen SDK for acronym.
    Returns (major_region, isocortex_area) or None if not found.
    """
    if not ALLEN_SDK_AVAILABLE:
        return None
    structs = structure_tree.get_structures_by_acronym([acronym])
    if not structs:
        return None
    path = _get_path(structs[0])
    if STR_ID in path:
        return 'STR', None
    if ISOCORTEX_ID in path:
        iso_idx = path.index(ISOCORTEX_ID)
        if iso_idx + 1 < len(path):
            child = structure_tree.get_structures_by_id([path[iso_idx + 1]])
            if child:
                return 'Isocortex', child[0]['acronym']
        return 'Isocortex', acronym
    if OLF_ID in path:
        return 'OLF', None
    if HPF_ID in path:
        return 'HPF', None
    if TH_ID in path:
        return 'Thalamus', None
    return None


def _classify_location(location):
    """
    Classify a location string into (base_area, layer_norm, major_region, isocortex_area).

    Strategy:
    1. Try Allen SDK with full location string
    2. If not found, strip layer suffix and retry
    3. Fall back to string matching on full string, then stripped
    """
    if location in _location_cache:
        return _location_cache[location]

    # Step 1: try full location string
    result = _query_allen(location)
    if result is not None:
        major_region, iso_area = result
        if major_region == 'Isocortex':
            # Full string is a valid Allen isocortex acronym (e.g. MOs2/3).
            # Still strip layer suffix so we get base_area and layer_norm.
            base_full, layer_raw_full = _try_strip_layer(location)
            if base_full != location:
                r2 = _query_allen(base_full)
                iso2 = r2[1] if r2 and r2[0] == 'Isocortex' else iso_area
                out = (base_full, LAYER_NORMALIZE.get(layer_raw_full, layer_raw_full),
                       'Isocortex', iso2 or iso_area)
            else:
                out = (location, None, 'Isocortex', iso_area)
        else:
            out = (location, None, major_region, None)
        _location_cache[location] = out
        return out

    # Step 2: strip layer suffix and retry
    base, layer_raw = _try_strip_layer(location)
    if base != location:
        result = _query_allen(base)
        if result is not None:
            major_region, iso_area = result
            if major_region == 'Isocortex':
                layer_norm = LAYER_NORMALIZE.get(layer_raw, layer_raw)
                out = (base, layer_norm, 'Isocortex', iso_area)
            else:
                # Layer suffix was a false positive (e.g., CA1 → CA+1, CA→HPF)
                out = (base, None, major_region, None)
            _location_cache[location] = out
            return out

    # Step 3: fallback string matching on full location
    mr = _olf_hpf_str_fallback(location)
    if mr:
        out = (location, None, mr, None)
        _location_cache[location] = out
        return out

    # Step 4: fallback on stripped base
    if base != location:
        mr = _olf_hpf_str_fallback(base)
        if mr:
            out = (base, None, mr, None)
            _location_cache[location] = out
            return out
        iso = _isocortex_area_fallback(base)
        if iso:
            layer_norm = LAYER_NORMALIZE.get(layer_raw, layer_raw)
            out = (base, layer_norm, 'Isocortex', iso)
            _location_cache[location] = out
            return out

    iso = _isocortex_area_fallback(location)
    if iso:
        out = (location, None, 'Isocortex', iso)
        _location_cache[location] = out
        return out

    out = (location, None, 'Other', None)
    _location_cache[location] = out
    return out


def annotate_locations(data):
    """
    Annotate each unit with major_region, isocortex_area, layer_norm.
    Uses location string directly with Allen SDK for ALL units.
    brain_area column is NOT used — it was unreliable in the extraction script.
    """
    print("\nAnnotating locations...")
    data = data.copy()

    loc_cache = {}
    base_areas    = []
    layer_norms   = []
    major_regions = []
    iso_areas     = []

    for loc in data['location']:
        if loc not in loc_cache:
            loc_cache[loc] = _classify_location(loc)
        b, ln, mr, ia = loc_cache[loc]
        base_areas.append(b)
        layer_norms.append(ln)
        major_regions.append(mr)
        iso_areas.append(ia)

    data['base_area']      = base_areas
    data['layer_norm']     = layer_norms
    data['major_region']   = major_regions
    data['isocortex_area'] = iso_areas
    data['is_thalamus']    = data['major_region'] == 'Thalamus'

    mr_counts = data.groupby('major_region').apply(lambda g: g[['session_id','unit_id']].drop_duplicates().shape[0])
    print(f"  Unit counts per major_region:\n{mr_counts.to_string()}")
    # Treat 0 as NaN for burst_rate/mean_burst_size — zero means no bursts detected
    for col in ['burst_fraction', 'burst_rate', 'mean_burst_size']:
        if col in data.columns:
            data[col] = data[col].where(data[col] > 0, np.nan)
    return data



# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_to_baseline(data):
    print("\nNormalising to baseline (absolute delta)...")
    data = data.copy()

    baseline_means = (data[data['period'] == 'Baseline']
                      .groupby(['session_id', 'unit_id'])[METRICS]
                      .mean()
                      .reset_index()
                      .rename(columns={m: f'baseline_{m}' for m in METRICS}))
    data = data.merge(baseline_means, on=['session_id', 'unit_id'], how='left')

    for metric in METRICS:
        bc = f'baseline_{metric}'
        data[f'{metric}_delta'] = data[metric] - data[bc]

    effect     = data[data['period'] == 'Effect'].copy()
    delta_cols = [f'{m}_delta' for m in METRICS]
    effect     = effect[effect[delta_cols].notna().any(axis=1)].copy()
    print(f"  Units after cleaning: {effect[['session_id','unit_id']].drop_duplicates().shape[0]:,}")
    return effect


# ============================================================================
# GROUP STATS
# ============================================================================

def compute_group_stats(data, group_col):
    delta_cols = [f'{m}_delta' for m in METRICS]
    needed     = [group_col, 'relative_time', 'treatment', 'session_id', 'unit_id'] + delta_cols
    df         = data[needed].dropna(subset=[group_col]).copy()

    rows = []
    for grp_val, grp_data in df.groupby(group_col):
        for t, t_data in grp_data.groupby('relative_time'):
            row = {group_col: grp_val, 'relative_time': t}
            for cond in ['Psilocybin', 'Saline', 'Ketanserin']:
                cond_data = t_data[t_data['treatment'] == cond]
                label     = cond.lower()
                row[f'n_{label}'] = cond_data[['session_id','unit_id']].drop_duplicates().shape[0]
                for metric in METRICS:
                    pc   = f'{metric}_delta'
                    vals = cond_data[pc].dropna()
                    row[f'{pc}_{label}']     = vals.mean() if len(vals) > 0 else np.nan
                    row[f'{pc}_{label}_sem'] = vals.sem()  if len(vals) > 1 else np.nan
            rows.append(row)

    return pd.DataFrame(rows).sort_values('relative_time').reset_index(drop=True)


def compute_prop_bursting_stats(annotated_data, group_col):
    """
    Region-level prop_bursting: fraction of ALL units bursting per
    (session, group, timepoint), expressed as absolute change from session's baseline fraction.
    All units contribute (including those that start/stop bursting).
    """
    data = annotated_data.copy()
    data = data[data[group_col].isin(MAJOR_REGION_ORDER)].copy()
    if 'rest_duration' in data.columns:
        data = data[data['rest_duration'] > 0].copy()
    data['prop_bursting'] = (data['burst_fraction'].fillna(0) > 0).astype(float)

    session_baseline = (data[data['period'] == 'Baseline']
                        .groupby([group_col, 'session_id', 'treatment'])['prop_bursting']
                        .mean()
                        .reset_index()
                        .rename(columns={'prop_bursting': 'baseline_frac'}))

    effect = data[data['period'] == 'Effect'].copy()
    session_effect = (effect
                      .groupby([group_col, 'session_id', 'treatment', 'relative_time'])['prop_bursting']
                      .mean()
                      .reset_index()
                      .rename(columns={'prop_bursting': 'effect_frac'}))

    session_effect = session_effect.merge(
        session_baseline, on=[group_col, 'session_id', 'treatment'], how='left')

    # Absolute change — no division, no 1e-4 filter needed
    session_effect['prop_bursting_abs_change'] = (
        session_effect['effect_frac'] - session_effect['baseline_frac']
    )

    rows = []
    col = 'prop_bursting_abs_change'
    for grp_val, grp_data in session_effect.groupby(group_col):
        for t, t_data in grp_data.groupby('relative_time'):
            row = {group_col: grp_val, 'relative_time': t}
            for cond in ['Psilocybin', 'Saline', 'Ketanserin']:
                cond_data = t_data[t_data['treatment'] == cond]
                label     = cond.lower()
                vals      = cond_data[col].dropna()
                row[f'n_{label}']         = len(vals)
                row[f'{col}_{label}']     = vals.mean() if len(vals) > 0 else np.nan
                row[f'{col}_{label}_sem'] = vals.sem()  if len(vals) > 1 else np.nan
            rows.append(row)

    return pd.DataFrame(rows).sort_values('relative_time').reset_index(drop=True)








# ============================================================================
# BOOTSTRAP  — same hierarchical sampler as distribution script
# ============================================================================

from collections import Counter

def _sample_hier(df, metric, levels, num_samples, nboots):
    """Recursive hierarchical bootstrap (verbatim from distribution script)."""
    if len(levels) == 0:
        sums   = np.zeros(nboots)
        counts = np.zeros(nboots)
        for i in range(nboots):
            if num_samples[i] > 0:
                n = len(df) * int(num_samples[i])
                sums[i]   = df[metric].sample(n=n, replace=True).sum()
                counts[i] = n
        return sums, counts
    items   = df[levels[0]].unique()
    samples = np.zeros((len(items), nboots))
    for i in range(nboots):
        chosen = Counter(np.random.choice(
            items, size=len(items) * int(num_samples[i])))
        for idx, item in enumerate(items):
            samples[idx, i] = chosen[item]
    sums   = np.zeros(nboots)
    counts = np.zeros(nboots)
    for idx, item in enumerate(items):
        temp = df[df[levels[0]] == item]
        ts, tc = _sample_hier(temp, metric, levels[1:], samples[idx, :], nboots)
        sums   += ts
        counts += tc
    return sums, counts


def _bootstrap_mean(df, metric, levels, nboots=1000):
    sums, counts = _sample_hier(df, metric, levels, [1] * nboots, nboots)
    valid  = counts > 0
    result = np.full(nboots, np.nan)
    result[valid] = sums[valid] / counts[valid]
    return result


def _p_from_boots(db, cb):
    valid = ~(np.isnan(db) | np.isnan(cb))
    if valid.sum() < 10:
        return None
    diff   = db[valid] - cb[valid]
    effect = np.mean(diff)
    p1     = np.mean(diff < 0) if effect >= 0 else np.mean(diff > 0)
    return min(2 * p1, 1.0)


def _bootstrap_job(args):
    """
    Unit-pooled hierarchical bootstrap at one (region, timepoint).
    levels = ['session_id', 'unit_id']  — matches distribution script.
    """
    area, rt, drug_cond, ctrl_cond, subset, pct_col, nboots = args

    p_pooled = None
    for cond in [drug_cond, ctrl_cond]:
        cd = (subset[(subset['relative_time'] == rt) &
                     (subset['treatment'] == cond)]
              .dropna(subset=[pct_col]))
        if cd['session_id'].nunique() < 1 or len(cd) < 2:
            return area, rt, None, None

    boots = {}
    for cond in [drug_cond, ctrl_cond]:
        cd = (subset[(subset['relative_time'] == rt) &
                     (subset['treatment'] == cond)]
              .dropna(subset=[pct_col]))
        # Average per unit per session first (same as distribution delta)
        um = (cd.groupby(['session_id', 'unit_id'])[pct_col]
                .mean()
                .reset_index())
        boots[cond] = _bootstrap_mean(
            um, pct_col, levels=['session_id', 'unit_id'], nboots=nboots)

    p_pooled = _p_from_boots(boots[drug_cond], boots[ctrl_cond])
    # Return (area, rt, p_pooled, p_hier=None) — hier not used
    return area, rt, p_pooled, None


def run_bootstrap(effect_valid, drug_cond, ctrl_cond, metric,
                  group_col='major_region', regions=None,
                  pct_col_override=None, nboots=1000, n_workers=1):
    pct_col   = pct_col_override or ('prop_bursting_abs_change' if metric == 'prop_bursting' else f'{metric}_delta')
    regions   = regions or MAJOR_REGION_ORDER
    rel_times = sorted(effect_valid['relative_time'].unique())

    jobs = [
        (region, rt, drug_cond, ctrl_cond,
         effect_valid[effect_valid[group_col] == region],
         pct_col, nboots)
        for region in regions
        for rt     in rel_times
        if len(effect_valid[effect_valid[group_col] == region]) > 0
    ]
    print(f"  [{metric}/{pct_col}] {len(jobs)} jobs on {n_workers} workers...")

    pooled_records = []

    def _collect(region, rt, p_pooled, _):
        if p_pooled is not None:
            pooled_records.append({group_col: region,
                                   'relative_time': rt,
                                   'p_value': p_pooled})

    try:
        with mp.Pool(processes=n_workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_bootstrap_job, jobs, chunksize=4)
            ):
                _collect(*result)
                if i % max(1, len(jobs) // 10) == 0:
                    print(f"    {i}/{len(jobs)} done...", flush=True)
    except Exception as e:
        print(f"  Pool failed ({e}), single-process fallback...")
        for job in jobs:
            _collect(*_bootstrap_job(job))

    return pd.DataFrame(pooled_records), pd.DataFrame()


# ============================================================================
# FDR CORRECTION — per timepoint across regions
# ============================================================================

def _bh(p_arr):
    """Benjamini-Hochberg on a 1-D array, returns adjusted p-values."""
    n = len(p_arr)
    try:
        from scipy.stats import false_discovery_control
        return false_discovery_control(np.clip(p_arr, 1e-10, 1.0), method='bh')
    except Exception:
        si    = np.argsort(p_arr)
        sp    = p_arr[si]
        p_adj = np.minimum(1.0, sp * n / np.arange(1, n + 1))
        for i in range(n - 2, -1, -1):
            p_adj[i] = min(p_adj[i], p_adj[i + 1])
        reord     = np.empty(n)
        reord[si] = p_adj
        return reord


def apply_fdr_per_timepoint(df, group_col='major_region'):
    """
    FDR correction applied independently at each timepoint.
    Family = regions at that timepoint (not regions × timepoints).
    Adds a 'p_fdr' column.
    """
    if df.empty or 'p_value' not in df.columns:
        return df
    df = df.copy()
    df['p_fdr'] = np.nan
    for t, grp in df.groupby('relative_time'):
        idx = grp.index
        p   = grp['p_value'].values.astype(float)
        df.loc[idx, 'p_fdr'] = _bh(p)
    return df


# ============================================================================
# PLOTTING
# ============================================================================

def plot_heatmap(stats, pvals_df, metric, drug_cond, ctrl_cond,
                 comp_label, t0_norm, sig_col, sig_threshold,
                 output_dir, correction_label,
                 group_col='major_region', regions=None,
                 label_map=None, fname_prefix='heatmap',
                 vmax=None):
    # Color = (drug absolute Δ) − (control absolute Δ).
    # This is the quantity the asterisk tests, so colored magnitude and
    # significance now correspond to the same contrast.
    _base = 'prop_bursting_abs_change' if metric == 'prop_bursting' else f'{metric}_delta'
    pct_col_drug = f'{_base}_{drug_cond.lower()}'
    pct_col_ctrl = f'{_base}_{ctrl_cond.lower()}'
    rel_times = sorted(stats['relative_time'].unique())

    region_order = regions or MAJOR_REGION_ORDER
    areas        = [a for a in region_order if a in stats[group_col].values]

    n_areas = len(areas)
    n_times = len(rel_times)
    area_idx = {a: i for i, a in enumerate(areas)}
    rt_idx   = {t: j for j, t in enumerate(rel_times)}

    # ── Colour: (drug − control) difference of %Δ from baseline ───────────────
    colour_matrix = np.full((n_areas, n_times), np.nan)
    for _, row in stats.iterrows():
        a = row[group_col]
        t = row['relative_time']
        v_drug = row.get(pct_col_drug, np.nan)
        v_ctrl = row.get(pct_col_ctrl, np.nan)
        if (a in area_idx and t in rt_idx
                and not pd.isna(v_drug) and not pd.isna(v_ctrl)):
            colour_matrix[area_idx[a], rt_idx[t]] = v_drug - v_ctrl

    if t0_norm:
        for i in range(n_areas):
            t0_val = colour_matrix[i, 0]
            if not np.isnan(t0_val):
                colour_matrix[i, :] -= t0_val

    # ── Significance ──────────────────────────────────────────────────────────
    sig_matrix = np.zeros((n_areas, n_times), dtype=bool)
    if not pvals_df.empty and sig_col in pvals_df.columns and group_col in pvals_df.columns:
        for _, row in pvals_df.iterrows():
            a = row[group_col]
            t = row['relative_time']
            if a in area_idx and t in rt_idx:
                if t0_norm and t == rel_times[0]:
                    continue
                sig_matrix[area_idx[a], rt_idx[t]] = row[sig_col] < sig_threshold

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_w = max(8, n_times * 0.9 + 2.5)
    fig_h = max(2, n_areas * 0.9 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax_data = np.nanmax(np.abs(colour_matrix))
    if np.isnan(vmax_data) or vmax_data == 0:
        vmax_data = 1.0
    vmax_use = vmax if vmax is not None else vmax_data

    if metric == 'prop_bursting':
        # Use same diverging PRGn as other metrics — centered at 0
        norm = TwoSlopeNorm(vmin=-vmax_use, vcenter=0, vmax=vmax_use)
        im   = ax.imshow(colour_matrix, cmap='PRGn', norm=norm,
                         aspect='auto', interpolation='none', origin='upper')
    else:
        norm = TwoSlopeNorm(vmin=-vmax_use, vcenter=0, vmax=vmax_use)
        im   = ax.imshow(colour_matrix, cmap='PRGn', norm=norm,
                         aspect='auto', interpolation='none', origin='upper')
    for i in range(n_areas):
        for j in range(n_times):
            if sig_matrix[i, j]:
                ax.text(j, i, '*', ha='center', va='center',
                        fontsize=14, fontweight='bold', color='black')
            if np.isnan(colour_matrix[i, j]):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, color='lightgrey', zorder=0))

    ax.set_xticks(range(n_times))
    ax.set_xticklabels([str(t) for t in rel_times], fontsize=9)
    ax.set_xlabel('Time post-injection (min)', fontsize=11)

    ax.set_yticks(range(n_areas))
    display_labels = [(label_map or {}).get(a, a) for a in areas]
    ax.set_yticklabels(display_labels, fontsize=12, fontweight='bold')
    for lbl in ax.get_yticklabels():
        lbl.set_color(AREA_LABEL_COLORS.get(lbl.get_text(), '#333333'))

    drug_color = CONDITION_COLORS.get(drug_cond, '#333333')

    if metric == 'prop_bursting':
        ylabel_str = 'Absolute change'
    else:
        ylabel_str = 'Absolute Δ from t=0' if t0_norm else 'Absolute Δ'

    ax.set_title(
        f'{METRIC_LABELS[metric]}: {drug_cond} − {ctrl_cond} ({ylabel_str})\n'
        f'* = sig., {correction_label}, p < {sig_threshold}',
        fontsize=12, color=drug_color, fontweight='bold', pad=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f'{drug_cond} − {ctrl_cond}\n({ylabel_str})', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()

    norm_suffix = 't0' if t0_norm else 'baseline'
    fname  = (f'{fname_prefix}_{metric}_{comp_label}'
               f'_{norm_suffix}_{correction_label}.png')
    fpath  = os.path.join(output_dir, fname)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")
    return fpath


# ============================================================================
# MAIN
# ============================================================================


def filter_valid_groups(data, group_col, min_units=10, min_sessions=3,
                       drug_cond=None):
    """
    Return groups meeting: >= min_units total, >= min_sessions PER condition.

    drug_cond=None  → union across both comparisons (for lineplots: show if
                      valid in ANY comparison)
    drug_cond='Psilocybin' or 'Ketanserin' → filter for that comparison only
                      (for heatmaps: one comparison per plot)
    """
    comparisons = [drug_cond] if drug_cond else ['Psilocybin', 'Ketanserin']
    all_valid = set()
    for dc in comparisons:
        pair_data = data[data['treatment'].isin([dc, 'Saline'])]
        for grp, gd in pair_data.groupby(group_col):
            if pd.isna(grp):
                continue
            n_drug_units = gd[gd['treatment'] == dc][['session_id','unit_id']].drop_duplicates().shape[0]
            n_ctrl_units = gd[gd['treatment'] == 'Saline'][['session_id','unit_id']].drop_duplicates().shape[0]
            n_drug_sess  = gd[gd['treatment'] == dc]['session_id'].nunique()
            n_ctrl_sess  = gd[gd['treatment'] == 'Saline']['session_id'].nunique()
            if (n_drug_units >= min_units and n_ctrl_units >= min_units and
                    n_drug_sess >= min_sessions and n_ctrl_sess >= min_sessions):
                all_valid.add(grp)

    valid = sorted(all_valid)
    mode = drug_cond or 'union'
    print(f"  Valid {group_col} ({mode}, >={min_units} units, >={min_sessions} sessions/cond): {valid}")
    return valid



# ============================================================================
# SHARED COLOUR-SCALE HELPER
# Share ONE colour bar per metric across both comparisons of the figure.
# Different metrics keep their own scale. Only colour scaling changes — stats,
# bootstrap p-values and asterisks are untouched.
# ============================================================================

def _panel_absmax(stats, metric, drug_cond, ctrl_cond, group_col, regions=None):
    """Max |drug_delta - ctrl_delta| that plot_heatmap would colour for this
    panel. Same columns plot_heatmap reads, so this equals
    np.nanmax(np.abs(colour_matrix)). 0.0 if nothing plottable."""
    if stats is None or len(stats) == 0:
        return 0.0
    _base = 'prop_bursting_abs_change' if metric == 'prop_bursting' else f'{metric}_delta'
    cd = f'{_base}_{drug_cond.lower()}'
    cc = f'{_base}_{ctrl_cond.lower()}'
    if cd not in stats.columns or cc not in stats.columns:
        return 0.0
    sub = stats if regions is None else stats[stats[group_col].isin(list(regions))]
    diff = (sub[cd] - sub[cc]).abs()
    m = diff.max(skipna=True)
    return float(m) if pd.notna(m) else 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Heatmap — major brain regions — RT & STR separated version')
    parser.add_argument('--base_dir',      default='../data_extraction')
    parser.add_argument('--output_dir',    default='../plots/heatmaps/heatmap_output_hier')
    parser.add_argument('--nboots',        type=int,   default=10000)
    parser.add_argument('--sig_threshold', type=float, default=0.05)
    parser.add_argument('--vmax',          type=float, default=None,
                        help='Fixed colour scale max (e.g. 100 for -100 to +100). '
                             'Default: auto per figure.')
    parser.add_argument('--n_workers',     type=int,   default=None)
    args = parser.parse_args()
    args.base_dir   = os.path.abspath(args.base_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    # Capped rather than scaled to cpu_count(): each worker reloads its own
    # copy of the Allen SDK structure tree and data, so memory scales with
    # worker count, not CPU count — an unbounded default can OOM on
    # high-core-count machines.
    n_workers = args.n_workers or max(1, min(os.cpu_count() - 1, 8))
    np.random.seed(42)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    raw    = load_all_data(args.base_dir)
    data   = annotate_locations(raw)
    effect = normalize_to_baseline(data)

    # Keep only recognised major regions
    effect_valid = effect[
        effect['major_region'].isin(MAJOR_REGION_ORDER) &
        (effect['relative_time'] <= 60)
    ].copy()

    # Filter: ≥10 units total, ≥3 sessions per condition
    valid_major = filter_valid_groups(effect_valid, 'major_region')
    effect_valid = effect_valid[effect_valid['major_region'].isin(valid_major)].copy()


    # Group stats — same values as lineplots
    stats = compute_group_stats(effect_valid, 'major_region')

    # ── prop_bursting: computed on ALL units before normalization filter ───────
    prop_stats = compute_prop_bursting_stats(data, 'major_region')
    # Add prop_bursting columns into stats by merging on (major_region, relative_time)
    stats = stats.merge(
        prop_stats[['major_region', 'relative_time'] +
                   [c for c in prop_stats.columns
                    if c.startswith('prop_bursting')]],
        on=['major_region', 'relative_time'], how='left')

    # Effect-period data for prop_bursting bootstrap — region-level normalization
    prop_data = data[
        (data['major_region'].isin(MAJOR_REGION_ORDER))
    ].copy()
    if 'rest_duration' in prop_data.columns:
        prop_data = prop_data[prop_data['rest_duration'] > 0].copy()
    prop_data['prop_bursting'] = (prop_data['burst_fraction'].fillna(0) > 0).astype(float)
    # Per-session baseline fraction
    _bsl_frac = (prop_data[prop_data['period'] == 'Baseline']
                 .groupby(['major_region', 'session_id', 'treatment'])['prop_bursting']
                 .mean().reset_index()
                 .rename(columns={'prop_bursting': 'baseline_frac'}))
    # Per-session effect fraction at each timepoint
    _eff = prop_data[
        (prop_data['period'] == 'Effect') &
        (prop_data['relative_time'] <= 60)
    ].copy()
    prop_effect = (_eff
                   .groupby(['major_region', 'session_id', 'treatment', 'relative_time'])['prop_bursting']
                   .mean().reset_index()
                   .rename(columns={'prop_bursting': 'effect_frac'}))
    prop_effect = prop_effect.merge(_bsl_frac, on=['major_region', 'session_id', 'treatment'], how='left')
    # Absolute change — no division, no 1e-4 filter needed
    prop_effect['prop_bursting_abs_change'] = (
        prop_effect['effect_frac'] - prop_effect['baseline_frac']
    )
    # Bootstrap resamples sessions — set unit_id = session_id
    prop_effect['unit_id'] = prop_effect['session_id']

    os.makedirs(args.output_dir, exist_ok=True)
    stats.to_csv(os.path.join(args.output_dir, 'group_stats.csv'), index=False)
    print(f"\n  group_stats.csv saved")

    # ── Bootstrap p-values + plots ────────────────────────────────────────────
    # Defer plotting so a shared per-metric colour scale can be computed across
    # both comparisons before anything is drawn.
    plot_jobs = []
    vmax_by_metric = {}
    for drug_cond, ctrl_cond, comp_label in COMPARISONS:
        print(f"\n{'='*60}")
        print(f"  {drug_cond} vs {ctrl_cond}")
        print(f"{'='*60}")

        # Per-comparison region filter
        valid_comp = filter_valid_groups(effect_valid, 'major_region',
                                         drug_cond=drug_cond)
        valid_comp = [r for r in MAJOR_REGION_ORDER if r in valid_comp]
        if not valid_comp:
            print("  No valid regions for this comparison — skipping")
            continue

        for metric in METRICS + METRICS_RAW:
            print(f"\n  Metric: {metric}")

            # Use prop_effect for prop_bursting, effect_valid for all others
            boot_data = prop_effect if metric == 'prop_bursting' else effect_valid

            print(f"  Running bootstrap...")
            pvals_pooled, _ = run_bootstrap(
                boot_data, drug_cond, ctrl_cond, metric,
                regions=valid_comp,
                nboots=args.nboots, n_workers=n_workers)

            # Save CSV
            pvals_pooled.to_csv(os.path.join(
                args.output_dir,
                f'pvals_{metric}_{comp_label}_unitpooled.csv'), index=False)

            # Defer per-timepoint heatmap (uncorrected only)
            if 'p_value' in pvals_pooled.columns:
                plot_jobs.append(dict(
                    stats=stats, pvals=pvals_pooled, metric=metric,
                    drug_cond=drug_cond, ctrl_cond=ctrl_cond,
                    comp_label=comp_label, regions=valid_comp))
                if args.vmax is None:
                    _m = _panel_absmax(stats, metric, drug_cond, ctrl_cond,
                                       'major_region', valid_comp)
                    vmax_by_metric[metric] = max(vmax_by_metric.get(metric, 0.0), _m)

    # ── Draw everything with a shared per-metric colour scale ────────────────
    for _j in plot_jobs:
        if args.vmax is not None:
            _vm = args.vmax
        else:
            _vm = vmax_by_metric.get(_j['metric']) or None
        plot_heatmap(
            _j['stats'], _j['pvals'], _j['metric'],
            _j['drug_cond'], _j['ctrl_cond'], _j['comp_label'],
            t0_norm=False,
            sig_col='p_value',
            sig_threshold=args.sig_threshold,
            output_dir=args.output_dir,
            correction_label='uncorrected',
            vmax=_vm,
            regions=_j['regions'],
        )

    print(f"\nAll done. Outputs in: {args.output_dir}/")


def _run_heatmap_for_group(effect_data, stats, group_col, regions,
                            label_map, metric, drug_cond, ctrl_cond,
                            comp_label, sig_threshold, output_dir,
                            nboots, n_workers, fname_prefix, vmax=None):
    # Add per-unit t0 column if not present (not applicable to prop_bursting)
    if metric != 'prop_bursting' and f'{metric}_delta_from_t0' not in effect_data.columns:
        t0_ref = (effect_data[effect_data['relative_time'] == 0]
                  .groupby(['session_id', 'unit_id'])
                  [f'{metric}_delta']
                  .mean().reset_index()
                  .rename(columns={f'{metric}_delta': f'{metric}_t0_ref'}))
        effect_data = effect_data.merge(
            t0_ref, on=['session_id', 'unit_id'], how='left')
        effect_data[f'{metric}_delta_from_t0'] = (
            effect_data[f'{metric}_delta'] -
            effect_data[f'{metric}_t0_ref'])

    print(f"  Running bootstrap (baseline pct)...")
    p_pool, _ = run_bootstrap(
        effect_data, drug_cond, ctrl_cond, metric,
        group_col=group_col, regions=regions,
        pct_col_override=None, nboots=nboots, n_workers=n_workers)

    for t0_norm in [False]:
        if 'p_value' not in p_pool.columns:
            continue
        plot_heatmap(
            stats, p_pool, metric,
            drug_cond, ctrl_cond, comp_label,
            t0_norm=t0_norm,
            sig_col='p_value',
            sig_threshold=sig_threshold,
            output_dir=output_dir,
            correction_label='uncorrected_unitpooled',
            group_col=group_col,
            regions=regions,
            label_map=label_map,
            fname_prefix=fname_prefix,
            vmax=vmax,
        )


if __name__ == '__main__':
    main()