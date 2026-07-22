"""
Firing Rate Distribution Plots with Half-Violin Overlay
========================================================
Mirrors distribution_layers_thalamus_hpf_30_50.py (burst version) but for
firing_rate (Hz). Same statistical pipeline: hierarchical bootstrap
(session → unit), BH-FDR per panel, min_units=10 / min_sessions=3 per condition.

Reads *_prestim_firing_rate_timecourse.csv files from
  <data_dir>/{psilocybin,saline,ketanserin}/

Three plot configurations:
  1. Cortical layer rows (14): Prefrontal/Somatosensory/Visual/RSP x layers
  2. Thalamic functional groups (6)
  3. Hippocampal nuclei (5)

Single metric: firing_rate (absolute Δ from per-unit baseline, averaged across
the effect window — default 25–50 min post-injection).

NaN/zero handling (DIFFERS from burst pipeline by design):
  - Firing rate of 0 Hz is a REAL measurement (silenced unit) and is kept as 0.
  - Only NaN / Inf values are dropped.

Usage
-----
  python distribution_layers_thalamus_hpf_firing_rate.py \\
      --data_dir <data_dir> --nboots 1000
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import argparse
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc='', **kw):
        if desc:
            print(f"    {desc}...")
        return iterable

try:
    from scipy.stats import gaussian_kde
except ImportError:
    raise ImportError("scipy is required for KDE half-violins")


# ============================================================================
# CONSTANTS
# ============================================================================

CONDITION_COLORS = {
    'Saline':     '#030aa7',
    'Psilocybin': '#a90308',
    'Ketanserin': '#F3CA4B',
}

METRIC_LABELS = {
    'firing_rate': 'Firing Rate (Hz)',
}

COMPARISONS = [
    ('Psilocybin', 'Saline'),
    ('Ketanserin', 'Saline'),
]

# Effect window — collapse across these 5-min bins
EFFECT_WINDOW = {'effect_25-30', 'effect_30-35', 'effect_35-40', 'effect_40-45', 'effect_45-50'}
WINDOW_LABEL  = '25-50 min post-injection'

# Allen IDs
ISOCORTEX_ID = 315
TH_ID        = 549
OLF_ID       = 698
HPF_ID       = 1089
STR_ID       = 477
RT_ID        = 262
CA_ID        = 375

LAYER_NORMALIZE = {'1': '1/2/3', '2/3': '1/2/3', '6a': '6', '6b': '6'}
LAYER_ORDER     = ['1/2/3', '4', '5', '6']

# ── Plot 1: cortical layer rows ──────────────────────────────────────────────
CORTICAL_AREA_TO_GROUP = {
    'MO': 'Prefrontal',  'ORB': 'Prefrontal', 'ACA': 'Prefrontal',
    'ILA': 'Prefrontal', 'PL':  'Prefrontal', 'FRP': 'Prefrontal',
    'SS':  'Somatosensory',
    'VIS': 'Visual',
    'RSP': 'RSP',
}

CORTICAL_ROWS = [
    ('Prefrontal',    '1/2/3'),
    ('Prefrontal',    '5'),
    ('Prefrontal',    '6'),
    ('Somatosensory', '1/2/3'),
    ('Somatosensory', '4'),
    ('Somatosensory', '5'),
    ('Somatosensory', '6'),
    ('Visual',        '1/2/3'),
    ('Visual',        '4'),
    ('Visual',        '5'),
    ('Visual',        '6'),
    ('RSP',           '1/2/3'),
    ('RSP',           '5'),
    ('RSP',           '6'),
]

def _cortical_key(group, layer):
    return f'{group} — L{layer}'

CORTICAL_ROW_ORDER  = [_cortical_key(g, l) for g, l in CORTICAL_ROWS]
CORTICAL_ROW_LABELS = {k: k for k in CORTICAL_ROW_ORDER}

# ── Plot 2: thalamic functional groups (from heatmap_hpf_thalamic_nuclei.py) ─
THALAMIC_GROUP_MAP = {
    'AD': 'Anterior', 'AV': 'Anterior', 'AMd': 'Anterior', 'AMv': 'Anterior',
    'LD': 'Anterior', 'IAD': 'Anterior',
    'MD': 'Higher-order', 'LP': 'Higher-order', 'PO': 'Higher-order',
    'PoT': 'Higher-order', 'SGN': 'Higher-order',
    'VAL': 'First-order Somatomotor', 'VPL': 'First-order Somatomotor',
    'VPM': 'First-order Somatomotor',
    'CL': 'Intralaminar', 'CM': 'Intralaminar', 'PCN': 'Intralaminar',
    'PF': 'Intralaminar', 'SPFp': 'Intralaminar',
    'LGd': 'First-order Sensory Geniculate', 'LGco': 'First-order Sensory Geniculate',
    'LGip': 'First-order Sensory Geniculate', 'LGsh': 'First-order Sensory Geniculate',
    'LGv': 'First-order Sensory Geniculate', 'IGL': 'First-order Sensory Geniculate',
    'MGd': 'First-order Sensory Geniculate', 'MGm': 'First-order Sensory Geniculate',
    'MGv': 'First-order Sensory Geniculate',
    'RT': 'Reticular nucleus',
}

THALAMIC_GROUP_ORDER = [
    'Anterior', 'Higher-order', 'First-order Somatomotor',
    'Intralaminar', 'First-order Sensory Geniculate', 'Reticular nucleus',
]

# ── Plot 3: hippocampal nuclei ───────────────────────────────────────────────
HPF_NUCLEUS_MERGE = {'ProS': 'Subicular complex', 'SUB': 'Subicular complex'}
HPF_ROW_ORDER = ['CA1', 'CA2', 'CA3', 'DG', 'Subicular complex']


# ============================================================================
# ALLEN SDK SETUP
# ============================================================================

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
            print("  Allen SDK StructureTree working")
        else:
            raise ValueError("empty result")
    except Exception as e:
        print(f"  StructureTree fallback ({e}); using manual workaround")
        structure_tree = ManualStructureTree(structure_graph)
        if structure_tree.get_structures_by_acronym(['VPM']):
            ALLEN_SDK_AVAILABLE = True
            print("  Manual structure tree working")
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


# ── Fallback maps (same as upstream scripts) ─────────────────────────────────
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

_OLF_FALLBACK = ['MOB','AOB','AON','TT','DP','PIR','NLOT','PAA','COA','OLF','EP','EPd','EPv']
_HPF_FALLBACK = ['CA1','CA2','CA3','DG','SUB','PRE','POST','PARA','FC','IG','HATA','APR','HPF','ProS']
_STR_FALLBACK = ['CP','ACB','FS','LSX','LS','CEA','BST','LA','BLA','BMA','AA','OT','MEA','SI','STR','isl','islm']

_LAYER_SUFFIXES = ['2/3', '6a', '6b', '1', '4', '5', '6']
_location_cache = {}
_hpf_nucleus_cache = {}
_thalamic_group_cache = {}


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


def _try_strip_layer(location):
    for suffix in _LAYER_SUFFIXES:
        if str(location).endswith(suffix):
            base = location[:-len(suffix)]
            if len(base) >= 2:
                return base, suffix
    return location, None


def _query_allen(acronym):
    if not ALLEN_SDK_AVAILABLE:
        return None
    structs = structure_tree.get_structures_by_acronym([acronym])
    if not structs:
        return None
    path = _get_path(structs[0])
    if RT_ID in path:
        return 'Thalamus', None
    if STR_ID in path:
        return 'STR', None
    if ISOCORTEX_ID in path:
        iso_idx = path.index(ISOCORTEX_ID)
        if iso_idx + 1 < len(path):
            child = structure_tree.get_structures_by_id([path[iso_idx + 1]])
            if child:
                return 'Isocortex', child[0]['acronym']
        return 'Isocortex', acronym
    if OLF_ID in path: return 'OLF', None
    if HPF_ID in path: return 'HPF', None
    if TH_ID  in path: return 'Thalamus', None
    return None


def _classify_location(location):
    """Returns (base_area, layer_norm, major_region, isocortex_area)."""
    if location in _location_cache:
        return _location_cache[location]

    result = _query_allen(location)
    if result is not None:
        major_region, iso_area = result
        if major_region == 'Isocortex':
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

    base, layer_raw = _try_strip_layer(location)
    if base != location:
        result = _query_allen(base)
        if result is not None:
            major_region, iso_area = result
            if major_region == 'Isocortex':
                out = (base, LAYER_NORMALIZE.get(layer_raw, layer_raw),
                       'Isocortex', iso_area)
            else:
                out = (base, None, major_region, None)
            _location_cache[location] = out
            return out

    mr = _olf_hpf_str_fallback(location)
    if mr:
        out = (location, None, mr, None)
        _location_cache[location] = out
        return out

    if base != location:
        mr = _olf_hpf_str_fallback(base)
        if mr:
            out = (base, None, mr, None)
            _location_cache[location] = out
            return out
        iso = _isocortex_area_fallback(base)
        if iso:
            out = (base, LAYER_NORMALIZE.get(layer_raw, layer_raw),
                   'Isocortex', iso)
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


def get_hpf_nucleus(location):
    """For an HPF location, resolve the nucleus (CA1, CA2, CA3, DG, Subicular complex, etc.)."""
    if location in _hpf_nucleus_cache:
        return _hpf_nucleus_cache[location]

    result = location
    if ALLEN_SDK_AVAILABLE:
        structs = structure_tree.get_structures_by_acronym([location])
        if structs:
            path = _get_path(structs[0])
            if HPF_ID in path:
                hpf_idx = path.index(HPF_ID)
                if hpf_idx + 2 < len(path):
                    depth2_id = path[hpf_idx + 2]
                    if depth2_id == CA_ID and hpf_idx + 3 < len(path):
                        child = structure_tree.get_structures_by_id([path[hpf_idx + 3]])
                        if child:
                            result = child[0]['acronym']
                    else:
                        child = structure_tree.get_structures_by_id([depth2_id])
                        if child:
                            result = child[0]['acronym']
                elif hpf_idx + 1 < len(path):
                    child = structure_tree.get_structures_by_id([path[hpf_idx + 1]])
                    if child:
                        result = child[0]['acronym']

    result = HPF_NUCLEUS_MERGE.get(result, result)
    _hpf_nucleus_cache[location] = result
    return result


def get_thalamic_group(location):
    """Map a thalamic location to one of THALAMIC_GROUP_ORDER, or None."""
    if location in _thalamic_group_cache:
        return _thalamic_group_cache[location]

    if location in THALAMIC_GROUP_MAP:
        _thalamic_group_cache[location] = THALAMIC_GROUP_MAP[location]
        return THALAMIC_GROUP_MAP[location]

    result = None
    if ALLEN_SDK_AVAILABLE:
        structs = structure_tree.get_structures_by_acronym([location])
        if structs:
            path = _get_path(structs[0])
            if TH_ID in path:
                th_idx = path.index(TH_ID)
                for pid in path[th_idx + 1:]:
                    node = structure_tree.get_structures_by_id([pid])
                    if node:
                        acr = node[0]['acronym']
                        if acr in THALAMIC_GROUP_MAP:
                            result = THALAMIC_GROUP_MAP[acr]
                            break

    _thalamic_group_cache[location] = result
    return result


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data(base_dir='.'):
    print("\nLoading firing-rate data...")
    for candidate in [base_dir, '.', '..',
                      'output_prestim_firing_rate', 'output_prestim_firing_rate_new',
                      '../output_prestim_firing_rate', '../output_prestim_firing_rate_new']:
        if (glob.glob(f'{candidate}/psilocybin/*_firing_rate_timecourse.csv') or
                glob.glob(f'{candidate}/saline/*_firing_rate_timecourse.csv')):
            base_dir = candidate
            break
    else:
        # Don't fail if user passed an explicit data_dir — just use it
        pass
    print(f"  Using base_dir: {base_dir}")

    dfs = []
    for treatment, folder in [('Psilocybin', 'psilocybin'),
                               ('Saline',     'saline'),
                               ('Ketanserin', 'ketanserin')]:
        files = glob.glob(f'{base_dir}/{folder}/*_firing_rate_timecourse.csv')
        print(f"  {treatment}: {len(files)} files")
        for f in files:
            df = pd.read_csv(f)
            df['session_id'] = (os.path.basename(f)
                                .replace('_prestim_firing_rate_timecourse.csv', '')
                                .replace('_firing_rate_timecourse.csv', ''))
            df['treatment'] = treatment
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No *_firing_rate_timecourse.csv files found under {base_dir}/"
            "{psilocybin,saline,ketanserin}/")

    data = pd.concat(dfs, ignore_index=True)
    data['period'] = data['bin'].apply(
        lambda x: 'Baseline' if 'baseline' in str(x) else 'Effect')

    data['time_bin_start'] = data['bin'].apply(
        lambda b: int(str(b).split('_')[1].split('-')[0]))

    # NaN/zero handling — DIFFERS from burst pipeline:
    # firing_rate of 0 Hz is a real measurement (silenced unit). Keep zeros.
    # Drop only NaN / +-Inf (e.g. from rest_duration == 0 division).
    if 'firing_rate' in data.columns:
        data['firing_rate'] = pd.to_numeric(data['firing_rate'], errors='coerce')
        data.loc[~np.isfinite(data['firing_rate']), 'firing_rate'] = np.nan

    print(f"  Total rows: {len(data):,}")
    print(f"  Units     : {data[['session_id','unit_id']].drop_duplicates().shape[0]:,}")
    print(f"  Sessions  : {data['session_id'].nunique()}")
    return data


# ============================================================================
# ANNOTATION
# ============================================================================

def annotate(data):
    print("\nAnnotating locations...")
    data = data.copy()

    loc_cache = {}
    base_areas, layer_norms, major_regions, iso_areas = [], [], [], []
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

    # Cortical row label  (e.g. "Prefrontal — L5")
    def _cortical_row(r):
        if r['major_region'] != 'Isocortex':
            return None
        grp = CORTICAL_AREA_TO_GROUP.get(r['isocortex_area'])
        if grp is None or r['layer_norm'] not in LAYER_ORDER:
            return None
        return _cortical_key(grp, r['layer_norm'])
    data['cortical_row'] = data.apply(_cortical_row, axis=1)

    # Thalamic group
    def _th_group(r):
        if r['major_region'] != 'Thalamus':
            return None
        for cand in [r['location'], r['base_area']]:
            g = get_thalamic_group(cand)
            if g is not None:
                return g
        return None
    data['thalamic_group'] = data.apply(_th_group, axis=1)

    # HPF nucleus
    def _hpf_nuc(r):
        if r['major_region'] != 'HPF':
            return None
        for cand in [r['location'], r['base_area']]:
            n = get_hpf_nucleus(cand)
            if n is not None:
                return n
        return None
    data['hpf_nucleus'] = data.apply(_hpf_nuc, axis=1)

    # Diagnostic counts
    print("\n  Unit counts by cortical_row:")
    cc = data.dropna(subset=['cortical_row']).groupby(['cortical_row', 'treatment'])['unit_id'].nunique().unstack(fill_value=0)
    if not cc.empty:
        print(cc.to_string())

    print("\n  Unit counts by thalamic_group:")
    tc = data.dropna(subset=['thalamic_group']).groupby(['thalamic_group', 'treatment'])['unit_id'].nunique().unstack(fill_value=0)
    if not tc.empty:
        print(tc.to_string())

    print("\n  Unit counts by hpf_nucleus:")
    hc = data.dropna(subset=['hpf_nucleus']).groupby(['hpf_nucleus', 'treatment'])['unit_id'].nunique().unstack(fill_value=0)
    if not hc.empty:
        print(hc.to_string())

    return data


# ============================================================================
# DELTA COMPUTATION  (per-unit pct change from baseline, 30–50 min window)
# ============================================================================

def compute_unit_deltas(data, metrics):
    print(f"\nComputing per-unit ABSOLUTE deltas ({WINDOW_LABEL})...")

    # Match heatmap exactly — group baseline by (session_id, unit_id)
    baseline_means = (data[data['period'] == 'Baseline']
                      .groupby(['session_id', 'unit_id'])[metrics]
                      .mean()
                      .rename(columns={m: f'baseline_{m}' for m in metrics})
                      .reset_index())
    enriched = data.merge(baseline_means, on=['session_id', 'unit_id'], how='left')

    delta_cols = []
    for m in metrics:
        bc = f'baseline_{m}'
        dc = f'{m}_delta'
        enriched[dc] = enriched[m] - enriched[bc]
        delta_cols.append(dc)

    effect = enriched[enriched['period'] == 'Effect'].copy()
    effect = effect[effect[delta_cols].notna().any(axis=1)].copy()

    meta_cols = ['session_id', 'unit_id', 'treatment', 'location',
                 'base_area', 'layer_norm', 'isocortex_area',
                 'major_region', 'cortical_row', 'thalamic_group', 'hpf_nucleus']
    meta_cols = [c for c in meta_cols if c in effect.columns]

    # Average across the effect window
    eff_win = (effect[effect['bin'].isin(EFFECT_WINDOW)]
               .groupby(['session_id', 'unit_id'])[delta_cols]
               .mean()
               .reset_index())

    meta = (effect[meta_cols]
            .drop_duplicates(subset=['session_id', 'unit_id', 'treatment']))

    merged = meta.merge(eff_win, on=['session_id', 'unit_id'], how='inner')
    merged = merged[merged[delta_cols].notna().any(axis=1)].copy()

    print(f"  Units with delta: {merged['unit_id'].nunique():,}")
    print(f"  Sessions        : {merged['session_id'].nunique()}")
    return merged


# ============================================================================
# HIERARCHICAL BOOTSTRAP  (session → unit)
# ============================================================================

def _sample_hier(df, metric, levels, num_samples, nboots):
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


def bootstrap_mean(df, metric, levels, nboots=1000):
    sums, counts = _sample_hier(df, metric, levels, [1] * nboots, nboots)
    valid  = counts > 0
    result = np.full(nboots, np.nan)
    result[valid] = sums[valid] / counts[valid]
    return result


# ============================================================================
# REGION-LEVEL STATS  (returns stats DataFrame + raw bootstrap arrays)
# ============================================================================

def compute_region_stats(deltas, group_col, regions, conditions,
                         delta_col, nboots=1000):
    levels = ['session_id', 'unit_id']  # hierarchical
    drug_cond, ctrl_cond = conditions
    records = []
    boots_by_region = {}

    for region in tqdm(regions, desc=f"{delta_col}"):
        rdata = deltas[deltas[group_col] == region].copy()
        if len(rdata) == 0:
            continue

        cond_boots = {}
        for cond in conditions:
            cd = rdata[rdata['treatment'] == cond].dropna(subset=[delta_col])
            if len(cd) < 2:
                continue
            boot = bootstrap_mean(
                cd[['session_id', 'unit_id', delta_col]],
                metric=delta_col, levels=levels, nboots=nboots)
            cond_boots[cond] = boot

        if drug_cond not in cond_boots or ctrl_cond not in cond_boots:
            continue

        db    = cond_boots[drug_cond]
        cb    = cond_boots[ctrl_cond]
        valid = ~(np.isnan(db) | np.isnan(cb))
        if valid.sum() < 10:
            continue

        diff   = db[valid] - cb[valid]
        effect = np.mean(diff)
        p1     = np.mean(diff < 0) if effect >= 0 else np.mean(diff > 0)
        p_val  = min(2 * p1, 1.0)

        dv = rdata[rdata['treatment'] == drug_cond][delta_col].dropna()
        cv = rdata[rdata['treatment'] == ctrl_cond][delta_col].dropna()
        if len(dv) > 1 and len(cv) > 1:
            pooled_sd = np.sqrt((dv.std(ddof=1)**2 + cv.std(ddof=1)**2) / 2)
            cohens_d  = (dv.mean() - cv.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        else:
            cohens_d = np.nan

        row = {group_col: region, 'p_value': p_val, 'cohens_d': cohens_d}
        for cond in conditions:
            if cond in cond_boots:
                b = cond_boots[cond]
                row[f'{cond}_mean']  = np.nanmean(b)
                row[f'{cond}_ci_lo'] = np.nanpercentile(b, 2.5)
                row[f'{cond}_ci_hi'] = np.nanpercentile(b, 97.5)
                cd_cond = rdata[rdata['treatment'] == cond].dropna(subset=[delta_col])
                row[f'{cond}_n_units']    = int(cd_cond['unit_id'].nunique())
                row[f'{cond}_n_sessions'] = int(cd_cond['session_id'].nunique())
        records.append(row)
        boots_by_region[region] = cond_boots

    return pd.DataFrame(records), boots_by_region


# ============================================================================
# FDR (Benjamini–Hochberg)
# ============================================================================

def apply_fdr(stats_df):
    if len(stats_df) == 0 or 'p_value' not in stats_df.columns:
        return stats_df
    p = stats_df['p_value'].values.copy()
    n = len(p)
    try:
        from scipy.stats import false_discovery_control
        p_adj = false_discovery_control(np.clip(p, 1e-10, 1.0), method='bh')
    except Exception:
        si    = np.argsort(p)
        sp    = p[si]
        p_adj = np.minimum(1.0, sp * n / np.arange(1, n + 1))
        for i in range(n - 2, -1, -1):
            p_adj[i] = min(p_adj[i], p_adj[i + 1])
        reord     = np.empty(n)
        reord[si] = p_adj
        p_adj     = reord
    out = stats_df.copy()
    out['p_fdr'] = p_adj
    return out


def stars(p):
    if   p < 0.001: return '***'
    elif p < 0.01:  return '**'
    elif p < 0.05:  return '*'
    return ''


# ============================================================================
# FILTERING
# ============================================================================

def filter_regions(deltas, group_col, conditions, min_units=10, min_sessions=3, verbose=True):
    valid = []
    rejected = []
    for region in deltas[group_col].dropna().unique():
        rdata  = deltas[deltas[group_col] == region]
        passes = True
        reasons = []
        for cond in conditions:
            cd = rdata[rdata['treatment'] == cond]
            n_u = cd['unit_id'].nunique()
            n_s = cd['session_id'].nunique()
            if n_u < min_units or n_s < min_sessions:
                passes = False
                reasons.append(f'{cond}: {n_u}u/{n_s}s')
        if passes:
            valid.append(region)
        else:
            rejected.append((region, '; '.join(reasons)))
    if verbose:
        print(f"  filter [{'+'.join(conditions)}] min_units={min_units}, min_sessions={min_sessions}:")
        print(f"    KEEP ({len(valid)}): {valid}")
        if rejected:
            print(f"    DROP ({len(rejected)}):")
            for region, reason in rejected:
                print(f"      {region}  →  {reason}")
    return valid


# ============================================================================
# PLOT — half-violin + point + CI  (matches screenshot style)
# ============================================================================

def _plot_half_violin(ax, x, density_max, y_base, direction, color,
                      half_height=0.38, alpha=0.35, fill=False):
    """Draw half-violin from y_base in `direction` (+1 up / -1 down).
    fill=True  → filled polygon
    fill=False → outline curve only (closed back to the baseline)
    """
    y_top = y_base + direction * density_max * half_height
    if fill:
        xs = np.concatenate([x, x[::-1]])
        ys = np.concatenate([np.full_like(x, y_base), y_top[::-1]])
        ax.fill(xs, ys, color=color, alpha=alpha, lw=0, zorder=2)
        ax.plot(x, y_top, color=color, lw=0.9, alpha=0.85, zorder=2)
    else:
        # Outline only — curve from baseline up through density and back
        xs = np.concatenate([[x[0]], x, [x[-1]]])
        ys = np.concatenate([[y_base], y_top, [y_base]])
        ax.plot(xs, ys, color=color, lw=1.1, alpha=0.95, zorder=2)


def forest_panel_with_violins(ax, stats_df, boots_dict, group_col, regions,
                              drug_cond, ctrl_cond, label_map,
                              sig_col='p_fdr'):
    drug_color = CONDITION_COLORS[drug_cond]
    ctrl_color = CONDITION_COLORS[ctrl_cond]
    point_off  = 0.14
    half_h     = 0.38
    n          = len(regions)
    x_all      = []
    sig_annots = []

    for i, region in enumerate(regions):
        rows = stats_df[stats_df[group_col] == region]
        if len(rows) == 0:
            continue
        row   = rows.iloc[0]
        boots = boots_dict.get(region, {})

        for cond, direction, color, yoff_sign in [
            # direction is in DATA y-space; ax.invert_yaxis() flips the
            # display, so visually drug ends up BELOW the row label and
            # control above — matching the screenshot reference.
            (drug_cond, +1, drug_color, +1),
            (ctrl_cond, -1, ctrl_color, -1),
        ]:
            # Half-violin from bootstrap distribution
            b = boots.get(cond)
            if b is not None:
                bv = b[~np.isnan(b)]
                if len(bv) > 5 and np.std(bv) > 1e-10:
                    try:
                        kde = gaussian_kde(bv)
                        lo, hi = np.percentile(bv, [0.5, 99.5])
                        pad = (hi - lo) * 0.05 + 1e-9
                        x_grid = np.linspace(lo - pad, hi + pad, 200)
                        dens   = kde(x_grid)
                        if dens.max() > 0:
                            dens = dens / dens.max()
                            _plot_half_violin(
                                ax, x_grid, dens, y_base=i,
                                direction=direction, color=color,
                                half_height=half_h)
                            x_all.extend([x_grid[0], x_grid[-1]])
                    except Exception:
                        pass

            # Point only (no horizontal CI bar — the violin shows the spread)
            m_key  = f'{cond}_mean'
            if m_key in row.index and not pd.isna(row[m_key]):
                m  = row[m_key]
                y_pt = i + yoff_sign * point_off
                ax.scatter(m, y_pt, color=color, s=42, zorder=5,
                           edgecolors='white', linewidths=0.5)
                # Still gather x extent from the CI for axis scaling
                lo_key = f'{cond}_ci_lo'
                hi_key = f'{cond}_ci_hi'
                if lo_key in row.index and not pd.isna(row.get(lo_key)):
                    x_all.extend([row[lo_key], row[hi_key]])

        p_unc  = row.get('p_value', 1.0)
        p_corr = row.get(sig_col, 1.0)
        d_val  = row.get('cohens_d', np.nan)
        star_corr = stars(p_corr if not pd.isna(p_corr) else 1.0)
        star_unc  = stars(p_unc  if not pd.isna(p_unc)  else 1.0)
        sig_annots.append((i, star_corr, star_unc, p_unc, d_val))

    ax.axvline(0, color='black', lw=1.0, ls='--', zorder=1, alpha=0.7)
    for i in range(n):
        ax.axhline(i, color='grey', lw=0.4, alpha=0.3, zorder=0)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([label_map.get(r, r) for r in regions], fontsize=9)
    ax.set_ylim(-0.6, n - 0.4)
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    return x_all, sig_annots


def apply_xlim_and_annotations(ax, xlim, sig_annots, drug_color):
    xmin, xmax = xlim
    span = max(abs(xmax - xmin), 1e-6)
    pad  = max(span * 0.05, abs(xmax) * 0.03, 0.001)
    ax.set_xlim(xmin - pad, xmax + pad * 14)  # extra room for annotations

    xr   = ax.get_xlim()[1]
    xl   = ax.get_xlim()[0]
    xpos = xr - (xr - xl) * 0.005

    for (i, star_corr, star_unc, p_unc, d_val) in sig_annots:
        # Tier 1: FDR-corrected star (bold, large) if present
        # Tier 2: uncorrected star only (lighter, smaller, in parens) if uncorr<.05 but FDR>=.05
        if star_corr:
            marker = star_corr
            fw, fs, alpha = 'bold', 10, 1.0
        elif star_unc:
            marker = f'({star_unc})'    # parens = uncorrected only
            fw, fs, alpha = 'normal', 7.5, 0.85
        else:
            marker = ''
            fw, fs, alpha = 'normal', 7, 0.7
        bits = []
        if marker:
            bits.append(marker)
        if not pd.isna(p_unc):
            bits.append(f'p={p_unc:.3f}')
        if not pd.isna(d_val):
            bits.append(f'd={d_val:.2f}')
        if bits:
            ax.text(xpos, i, '\n'.join(bits), ha='right', va='center',
                    fontsize=fs, color=drug_color, fontweight=fw, alpha=alpha)


# ============================================================================
# FIGURE BUILDER
# ============================================================================

def make_figure(deltas, group_col, regions, comparisons, delta_col,
                title_main, xlabel_str, out_path, label_map,
                nboots=1000, min_units=1, min_sessions=1,
                stats_csv_path=None):
    n_comp = len(comparisons)
    n_reg  = len(regions)
    panel_h = max(4.0, n_reg * 0.5 + 1.5)
    fig, axes = plt.subplots(
        n_comp, 1,
        figsize=(8.5, panel_h * n_comp + 1.2),
        constrained_layout=True)
    if n_comp == 1:
        axes = np.array([axes])

    all_stats     = {}
    all_boots     = {}
    all_regions   = {}
    x_all_global  = []
    stats_records = []

    # --- first pass: stats + bootstraps + xlim gather ---
    for row_idx, (drug_cond, ctrl_cond) in enumerate(comparisons):
        conds = [drug_cond, ctrl_cond]
        valid = set(filter_regions(deltas, group_col, conds, min_units, min_sessions))
        plot_regions = [r for r in regions if r in valid]
        all_regions[row_idx] = plot_regions

        if not plot_regions:
            continue

        stats, boots = compute_region_stats(
            deltas, group_col, plot_regions, conds,
            delta_col, nboots=nboots)
        stats = apply_fdr(stats)  # BH across rows in this panel
        all_stats[row_idx] = stats
        all_boots[row_idx] = boots

        # CSV log
        slog = stats.copy()
        slog['comparison'] = f'{drug_cond}_vs_{ctrl_cond}'
        slog['delta_col']  = delta_col
        stats_records.append(slog)

        for cond in conds:
            for key in [f'{cond}_ci_lo', f'{cond}_ci_hi']:
                if key in stats.columns:
                    x_all_global.extend(stats[key].dropna().tolist())
        # Also include violin support so x-axis fits the distributions
        for region, cb in boots.items():
            for cond, b in cb.items():
                bv = b[~np.isnan(b)]
                if len(bv) > 0:
                    x_all_global.extend(np.percentile(bv, [1, 99]).tolist())

    xs = [x for x in x_all_global if np.isfinite(x)]
    shared_xlim = (min(xs), max(xs)) if xs else (-1, 1)

    # --- second pass: draw ---
    for row_idx, (drug_cond, ctrl_cond) in enumerate(comparisons):
        ax = axes[row_idx]
        plot_regions = all_regions.get(row_idx, [])

        if not plot_regions:
            ax.text(0.5, 0.5, 'No data\n(filter threshold not met)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=11, color='grey')
            ax.set_axis_off()
            continue

        stats = all_stats.get(row_idx, pd.DataFrame())
        boots = all_boots.get(row_idx, {})

        x_vals, sig_annots = forest_panel_with_violins(
            ax, stats, boots, group_col, plot_regions,
            drug_cond, ctrl_cond, label_map=label_map, sig_col='p_fdr')

        apply_xlim_and_annotations(
            ax, shared_xlim, sig_annots, CONDITION_COLORS[drug_cond])

        ax.set_title(
            f'{drug_cond} vs {ctrl_cond}  '
            f'[hierarchical bootstrap (session → unit), FDR corrected]',
            fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel(xlabel_str, fontsize=10)

    metric_key  = '_'.join(delta_col.split('_')[2:])
    metric_nice = METRIC_LABELS.get(metric_key, metric_key)
    fig.suptitle(
        f'{title_main}  |  {metric_nice}  |  {WINDOW_LABEL}',
        fontsize=12.5, fontweight='bold', y=1.005)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

    if stats_csv_path and stats_records:
        df = pd.concat(stats_records, ignore_index=True)
        os.makedirs(os.path.dirname(stats_csv_path) or '.', exist_ok=True)
        df.to_csv(stats_csv_path, index=False)
        print(f"  Saved: {stats_csv_path}")


# ============================================================================
# RUN ONE PLOT GROUP (cortical / thalamic / hpf) — all 4 metrics
# ============================================================================

def run_plot_group(deltas, *, group_col, row_order, label_map,
                   title_main, out_subdir, nboots,
                   min_units=10, min_sessions=3):
    metric = 'firing_rate'
    delta_col = f'{metric}_delta'

    if delta_col not in deltas.columns:
        print(f"  Skipping {delta_col} — column missing")
        return

    present = set(deltas[group_col].dropna().unique())
    rows = [r for r in row_order if r in present]
    print(f"\n[{title_main}] rows present: {rows}")

    out_path  = os.path.join(out_subdir, f'{metric}_delta.png')
    stats_csv = os.path.join(out_subdir, f'{metric}_delta_stats.csv')
    xlabel    = f'Δ {METRIC_LABELS[metric]} (absolute change from baseline)'
    print(f"\n=== {title_main} | {metric} ===")
    make_figure(
        deltas, group_col, rows, COMPARISONS, delta_col,
        title_main=title_main, xlabel_str=xlabel,
        out_path=out_path, label_map=label_map,
        nboots=nboots, min_units=min_units, min_sessions=min_sessions,
        stats_csv_path=stats_csv)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Firing-rate distribution plots (half-violin + point) — configurable window',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--data_dir', default='.',
                        help='Root data dir (psilocybin/, saline/, ketanserin/ subfolders)')
    parser.add_argument('--out_dir',  default='firing_rate_plots_25_50')
    parser.add_argument('--nboots',   type=int, default=1000)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--min_units',    type=int, default=10,
                        help='Min units PER CONDITION per region (default: 10, matches heatmap)')
    parser.add_argument('--min_sessions', type=int, default=3,
                        help='Min sessions PER CONDITION per region (default: 3, matches heatmap)')
    parser.add_argument('--window_start', type=int, default=25,
                        help='Effect window start in minutes post-injection (default: 25)')
    parser.add_argument('--window_end',   type=int, default=50,
                        help='Effect window end in minutes post-injection (default: 50)')
    args = parser.parse_args()

    # Build effect window from CLI args (in 5-min bins)
    global EFFECT_WINDOW, WINDOW_LABEL
    bin_edges = list(range(args.window_start, args.window_end, 5))
    EFFECT_WINDOW = {f'effect_{s}-{s+5}' for s in bin_edges}
    WINDOW_LABEL  = f'{args.window_start}-{args.window_end} min post-injection'
    print(f"Effect window: {WINDOW_LABEL}  →  bins: {sorted(EFFECT_WINDOW)}")

    np.random.seed(args.seed)

    metrics = ['firing_rate']

    raw     = load_all_data(args.data_dir)
    data    = annotate(raw)
    deltas  = compute_unit_deltas(data, metrics)

    win_tag = f'{args.window_start}_{args.window_end}'

    # ── Plot 1: Cortical layer rows ──────────────────────────────────────────
    run_plot_group(
        deltas,
        group_col='cortical_row',
        row_order=CORTICAL_ROW_ORDER,
        label_map=CORTICAL_ROW_LABELS,
        title_main='Cortical Layers (Prefrontal / Somatosensory / Visual / RSP)',
        out_subdir=os.path.join(args.out_dir, f'cortical_layers_{win_tag}'),
        nboots=args.nboots,
        min_units=args.min_units, min_sessions=args.min_sessions)

    # ── Plot 2: Thalamic functional groups ───────────────────────────────────
    run_plot_group(
        deltas,
        group_col='thalamic_group',
        row_order=THALAMIC_GROUP_ORDER,
        label_map={k: k for k in THALAMIC_GROUP_ORDER},
        title_main='Thalamic Functional Groups',
        out_subdir=os.path.join(args.out_dir, f'thalamic_groups_{win_tag}'),
        nboots=args.nboots,
        min_units=args.min_units, min_sessions=args.min_sessions)

    # ── Plot 3: Hippocampal nuclei ───────────────────────────────────────────
    run_plot_group(
        deltas,
        group_col='hpf_nucleus',
        row_order=HPF_ROW_ORDER,
        label_map={k: k for k in HPF_ROW_ORDER},
        title_main='Hippocampal Nuclei',
        out_subdir=os.path.join(args.out_dir, f'hpf_nuclei_{win_tag}'),
        nboots=args.nboots,
        min_units=args.min_units, min_sessions=args.min_sessions)

    print(f"\n{'='*70}")
    print(f"Done. Outputs in: {args.out_dir}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
