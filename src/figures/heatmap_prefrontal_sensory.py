"""
Heatmap — Prefrontal & Sensory Cortex — Layers × Metrics
=========================================================
Separate layer heatmaps for:
  Prefrontal: MO, ORB, ACA, ILA, PL, FRP
  Sensory:    SS, VIS

All 4 metrics: burst_fraction, burst_rate, mean_burst_size, prop_bursting
Uncorrected hierarchical bootstrap only.
Delta-of-delta coloring: (drug absolute Δ) − (control absolute Δ)

Usage
-----
  python heatmap_prefrontal_sensory.py --base_dir output_prestim_burst_new
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
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

METRICS = ['burst_fraction', 'burst_rate', 'mean_burst_size']
METRICS_RAW = ['prop_bursting']

METRIC_LABELS = {
    'burst_fraction':  'Burst Fraction',
    'burst_rate':      'Burst Rate',
    'mean_burst_size': 'Mean Burst Size',
    'prop_bursting':   'Prop. Bursting Units',
}

CONDITION_COLORS = {
    'Saline':     '#030aa7',
    'Psilocybin': '#a90308',
    'Ketanserin': '#F3CA4B',
}

COMPARISONS = [
    ('Psilocybin', 'Saline',    'Psilocybin_vs_Saline'),
    ('Ketanserin', 'Saline',    'Ketanserin_vs_Saline'),
]

CORTICAL_GROUPS = {
    'Prefrontal':      ['MO', 'ORB', 'ACA', 'ILA', 'PL', 'FRP'],
    'Somatosensory':   ['SS'],
    'Visual':          ['VIS'],
}

LAYER_ORDER = ['1/2/3', '4', '5', '6']
LAYER_LABEL_MAP = {l: f'Layer {l}' for l in LAYER_ORDER}
LAYER_NORMALIZE = {'1': '1/2/3', '2/3': '1/2/3', '6a': '6', '6b': '6'}

ISOCORTEX_ID = 315
TH_ID        = 549
OLF_ID       = 698
HPF_ID       = 1089
STR_ID       = 477
RT_ID        = 262

# ============================================================================
# ALLEN SDK SETUP (copied from heatmap_hier_bootstrap.py)
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


def _query_allen(acronym):
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


_LAYER_SUFFIXES = ['2/3', '6a', '6b', '1', '4', '5', '6']
_location_cache = {}


def _try_strip_layer(location):
    for suffix in _LAYER_SUFFIXES:
        if str(location).endswith(suffix):
            base = location[:-len(suffix)]
            if len(base) >= 2:
                return base, suffix
    return location, None


def _classify_location(location):
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
                layer_norm = LAYER_NORMALIZE.get(layer_raw, layer_raw)
                out = (base, layer_norm, 'Isocortex', iso_area)
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


# ============================================================================
# DATA LOADING & ANNOTATION
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
    if 'total_spikes' in data.columns:
        data.loc[data['total_spikes'] == 0, 'burst_fraction'] = np.nan
    return data


def annotate_locations(data):
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
    data['is_thalamus']    = data['major_region'] == 'Thalamus'
    mr_counts = data.groupby('major_region').apply(lambda g: g[['session_id','unit_id']].drop_duplicates().shape[0])
    print(f"  Unit counts per major_region:\n{mr_counts.to_string()}")
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


def compute_prop_bursting_stats(data, group_col):
    """
    Region-level prop_bursting: fraction of ALL units bursting per
    (session, group, timepoint), expressed as absolute change from session's baseline fraction.
    All units contribute (including those that start/stop bursting).
    Plotted value = mean across sessions ± SEM.

    Requires FULL data (baseline + effect periods).
    """
    data = data.copy()
    if 'rest_duration' in data.columns:
        data = data[data['rest_duration'] > 0].copy()
    data['prop_bursting'] = (data['burst_fraction'].fillna(0) > 0).astype(float)
    col = 'prop_bursting_abs_change'

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
    session_effect[col] = session_effect['effect_frac'] - session_effect['baseline_frac']

    rows = []
    for grp_val, grp_data in session_effect.groupby(group_col):
        for t, t_data in grp_data.groupby('relative_time'):
            row = {group_col: grp_val, 'relative_time': t}
            for cond in ['Psilocybin', 'Saline', 'Ketanserin']:
                cond_data = t_data[t_data['treatment'] == cond]
                label = cond.lower()
                vals  = cond_data[col].dropna()
                row[f'n_{label}']             = len(vals)
                row[f'{col}_{label}']         = vals.mean() if len(vals) > 0 else np.nan
                row[f'{col}_{label}_sem']     = vals.sem()  if len(vals) > 1 else np.nan
            rows.append(row)

    return pd.DataFrame(rows).sort_values('relative_time').reset_index(drop=True)




# ============================================================================
# BOOTSTRAP
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
    area, rt, drug_cond, ctrl_cond, subset, pct_col, nboots = args
    for cond in [drug_cond, ctrl_cond]:
        cd = (subset[(subset['relative_time'] == rt) &
                     (subset['treatment'] == cond)]
              .dropna(subset=[pct_col]))
        if cd['session_id'].nunique() < 1 or len(cd) < 2:
            return area, rt, None
    boots = {}
    for cond in [drug_cond, ctrl_cond]:
        cd = (subset[(subset['relative_time'] == rt) &
                     (subset['treatment'] == cond)]
              .dropna(subset=[pct_col]))
        um = (cd.groupby(['session_id', 'unit_id'])[pct_col]
                .mean()
                .reset_index())
        boots[cond] = _bootstrap_mean(
            um, pct_col, levels=['session_id', 'unit_id'], nboots=nboots)
    p = _p_from_boots(boots[drug_cond], boots[ctrl_cond])
    return area, rt, p


def run_bootstrap(effect_data, drug_cond, ctrl_cond, metric,
                  group_col='layer_norm', regions=None,
                  pct_col_override=None, nboots=1000, n_workers=1):
    pct_col   = pct_col_override or ('prop_bursting_abs_change' if metric == 'prop_bursting' else f'{metric}_delta')
    regions   = regions or LAYER_ORDER
    rel_times = sorted(effect_data['relative_time'].unique())

    jobs = [
        (region, rt, drug_cond, ctrl_cond,
         effect_data[effect_data[group_col] == region],
         pct_col, nboots)
        for region in regions
        for rt     in rel_times
        if len(effect_data[effect_data[group_col] == region]) > 0
    ]
    print(f"    [{metric}] {len(jobs)} jobs on {n_workers} workers...")

    records = []
    try:
        with mp.Pool(processes=n_workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_bootstrap_job, jobs, chunksize=4)
            ):
                area, rt, p = result
                if p is not None:
                    records.append({group_col: area, 'relative_time': rt, 'p_value': p})
                if i % max(1, len(jobs) // 5) == 0:
                    print(f"      {i}/{len(jobs)} done...", flush=True)
    except Exception as e:
        print(f"    Pool failed ({e}), single-process fallback...")
        for job in jobs:
            area, rt, p = _bootstrap_job(job)
            if p is not None:
                records.append({group_col: area, 'relative_time': rt, 'p_value': p})

    return pd.DataFrame(records)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_heatmap(stats, pvals_df, metric, drug_cond, ctrl_cond,
                 comp_label, sig_threshold, output_dir,
                 group_col='layer_norm', regions=None,
                 label_map=None, fname_prefix='heatmap', vmax=None):

    _base = 'prop_bursting_abs_change' if metric == 'prop_bursting' else f'{metric}_delta'
    pct_col_drug = f'{_base}_{drug_cond.lower()}'
    pct_col_ctrl = f'{_base}_{ctrl_cond.lower()}'
    rel_times = sorted(stats['relative_time'].unique())

    region_order = regions or LAYER_ORDER
    areas        = [a for a in region_order if a in stats[group_col].values]

    n_areas = len(areas)
    n_times = len(rel_times)
    area_idx = {a: i for i, a in enumerate(areas)}
    rt_idx   = {t: j for j, t in enumerate(rel_times)}

    colour_matrix = np.full((n_areas, n_times), np.nan)
    for _, row in stats.iterrows():
        a = row[group_col]
        t = row['relative_time']
        v_drug = row.get(pct_col_drug, np.nan)
        v_ctrl = row.get(pct_col_ctrl, np.nan)
        if (a in area_idx and t in rt_idx
                and not pd.isna(v_drug) and not pd.isna(v_ctrl)):
            colour_matrix[area_idx[a], rt_idx[t]] = v_drug - v_ctrl

    sig_matrix = np.zeros((n_areas, n_times), dtype=bool)
    if not pvals_df.empty and 'p_value' in pvals_df.columns and group_col in pvals_df.columns:
        for _, row in pvals_df.iterrows():
            a = row[group_col]
            t = row['relative_time']
            if a in area_idx and t in rt_idx:
                sig_matrix[area_idx[a], rt_idx[t]] = row['p_value'] < sig_threshold

    fig_w = max(8, n_times * 0.9 + 2.5)
    fig_h = max(2, n_areas * 0.9 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax_data = np.nanmax(np.abs(colour_matrix))
    if np.isnan(vmax_data) or vmax_data == 0:
        vmax_data = 1.0
    vmax_use = vmax if vmax is not None else vmax_data

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

    drug_color = CONDITION_COLORS.get(drug_cond, '#333333')
    ylabel_str = 'Absolute change' if metric == 'prop_bursting' else 'Absolute Δ'
    ax.set_title(
        f'{METRIC_LABELS[metric]}: {drug_cond} − {ctrl_cond} ({ylabel_str})\n'
        f'* = sig., uncorrected, p < {sig_threshold}',
        fontsize=12, color=drug_color, fontweight='bold', pad=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f'{drug_cond} − {ctrl_cond}\n({ylabel_str})', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    fname  = f'{fname_prefix}_{metric}_{comp_label}_baseline_uncorrected.png'
    fpath  = os.path.join(output_dir, fname)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {fname}")


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
# Share ONE colour bar per metric across all sub-panels of the same figure
# (Prefrontal / Somatosensory / Visual and both comparisons). Different metrics
# keep their own scale. Only the colour scaling changes — stats, bootstrap
# p-values and asterisks are untouched.
# ============================================================================

def _panel_absmax(stats, metric, drug_cond, ctrl_cond, group_col, regions=None):
    """Max |drug_delta - ctrl_delta| that plot_heatmap would colour for this
    panel. Uses the exact same columns plot_heatmap reads, so this equals
    np.nanmax(np.abs(colour_matrix)) for the panel. 0.0 if nothing plottable."""
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
        description='Heatmap — Prefrontal & Sensory cortex layers')
    parser.add_argument('--base_dir',      default='../data_extraction')
    parser.add_argument('--output_dir',    default='../plots/heatmaps/heatmap_pfc_sensory')
    parser.add_argument('--nboots',        type=int,   default=10000)
    parser.add_argument('--sig_threshold', type=float, default=0.05)
    parser.add_argument('--vmax',          type=float, default=None)
    parser.add_argument('--n_workers',     type=int,   default=None)
    args = parser.parse_args()
    args.base_dir   = os.path.abspath(args.base_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    n_workers = args.n_workers or min(os.cpu_count() - 1, 8)
    np.random.seed(42)

    all_pvals = []

    # Defer all plotting so a shared per-metric colour scale can be computed
    # across every sub-panel of the figure before anything is drawn.
    plot_jobs = []
    vmax_by_metric = {}

    # ── Load & annotate ──────────────────────────────────────────────────────
    raw    = load_all_data(args.base_dir)
    data   = annotate_locations(raw)
    effect = normalize_to_baseline(data)

    # ── Process each cortical group ──────────────────────────────────────────
    for group_name, area_list in CORTICAL_GROUPS.items():
        print(f"\n{'='*70}")
        print(f"  {group_name}: {area_list}")
        print(f"{'='*70}")

        out_dir = os.path.join(args.output_dir, group_name)

        # Filter to this cortical group + valid layers
        grp_effect = effect[
            effect['isocortex_area'].isin(area_list) &
            effect['layer_norm'].isin(LAYER_ORDER) &
            (effect['relative_time'] <= 60)
        ].copy()

        grp_full = data[
            data['isocortex_area'].isin(area_list) &
            data['layer_norm'].isin(LAYER_ORDER)
        ].copy()

        layers_present = filter_valid_groups(grp_effect, 'layer_norm')
        layers_present = [l for l in LAYER_ORDER if l in layers_present]

        for layer in layers_present:
            n = grp_effect[grp_effect['layer_norm'] == layer][['session_id','unit_id']].drop_duplicates().shape[0]
            print(f"  Layer {layer}: {n:,} units")

        if not layers_present:
            print(f"  No data — skipping {group_name}")
            continue

        # Group stats for normalized metrics
        stats = compute_group_stats(grp_effect, 'layer_norm')

        # Prop_bursting stats (needs full data)
        prop_stats = compute_prop_bursting_stats(grp_full, 'layer_norm')
        stats = stats.merge(
            prop_stats[['layer_norm', 'relative_time'] +
                       [c for c in prop_stats.columns
                        if c.startswith('prop_bursting')]],
            on=['layer_norm', 'relative_time'], how='left')

        # Prop_bursting bootstrap data — region-level normalization
        prop_src = grp_full[grp_full['major_region'] == 'Isocortex'].copy()
        if 'rest_duration' in prop_src.columns:
            prop_src = prop_src[prop_src['rest_duration'] > 0].copy()
        prop_src['prop_bursting'] = (prop_src['burst_fraction'].fillna(0) > 0).astype(float)
        _bsl_frac = (prop_src[prop_src['period'] == 'Baseline']
                     .groupby(['layer_norm', 'session_id', 'treatment'])['prop_bursting']
                     .mean().reset_index()
                     .rename(columns={'prop_bursting': 'baseline_frac'}))
        _eff = prop_src[
            (prop_src['period'] == 'Effect') &
            (prop_src['relative_time'] <= 60)
        ].copy()
        prop_effect = (_eff
                       .groupby(['layer_norm', 'session_id', 'treatment', 'relative_time'])['prop_bursting']
                       .mean().reset_index()
                       .rename(columns={'prop_bursting': 'effect_frac'}))
        prop_effect = prop_effect.merge(_bsl_frac, on=['layer_norm', 'session_id', 'treatment'], how='left')
        # Absolute change — no division, no 1e-4 filter needed
        prop_effect['prop_bursting_abs_change'] = (
            prop_effect['effect_frac'] - prop_effect['baseline_frac']
        )
        prop_effect['unit_id'] = prop_effect['session_id']

        # ── Bootstrap + plot per comparison × metric ─────────────────────────
        for drug_cond, ctrl_cond, comp_label in COMPARISONS:
            print(f"\n  {drug_cond} vs {ctrl_cond}")

            layers_comp = filter_valid_groups(grp_effect, 'layer_norm',
                                               drug_cond=drug_cond)
            layers_comp = [l for l in LAYER_ORDER if l in layers_comp]
            if not layers_comp:
                print(f"    No valid layers for {comp_label} — skipping")
                continue

            for metric in METRICS + METRICS_RAW:
                print(f"    Metric: {metric}")

                boot_data = prop_effect if metric == 'prop_bursting' else grp_effect

                pvals = run_bootstrap(
                    boot_data, drug_cond, ctrl_cond, metric,
                    group_col='layer_norm', regions=layers_comp,
                    nboots=args.nboots, n_workers=n_workers)

                pv = pvals.copy()
                pv['cortical_group'] = group_name
                pv['metric'] = metric
                pv['comparison'] = comp_label
                all_pvals.append(pv)

                plot_jobs.append(dict(
                    stats=stats, pvals=pvals, metric=metric,
                    drug_cond=drug_cond, ctrl_cond=ctrl_cond,
                    comp_label=comp_label, output_dir=out_dir,
                    group_col='layer_norm', regions=layers_comp,
                    label_map=LAYER_LABEL_MAP,
                    fname_prefix=f'heatmap_{group_name}'))
                if args.vmax is None:
                    _m = _panel_absmax(stats, metric, drug_cond, ctrl_cond,
                                       'layer_norm', layers_comp)
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
            sig_threshold=args.sig_threshold,
            output_dir=_j['output_dir'],
            group_col=_j['group_col'],
            regions=_j['regions'],
            label_map=_j['label_map'],
            fname_prefix=_j['fname_prefix'],
            vmax=_vm)

    if all_pvals:
        pvals_out = pd.concat(all_pvals, ignore_index=True)
        pvals_out.to_csv(os.path.join(args.output_dir, f'pvals_nboots{args.nboots}.csv'), index=False)

    print(f"\nDone. Outputs in: {args.output_dir}/")


if __name__ == '__main__':
    main()