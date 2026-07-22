import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import h5py
import argparse
import os
from collections import defaultdict


class ManualStructureTree:
    def __init__(self, structures):
        self.structures = structures
        self.structures_by_id = {s['id']: s for s in structures}
        self.structures_by_acronym = defaultdict(list)
        for s in structures:
            if 'acronym' in s:
                self.structures_by_acronym[s['acronym']].append(s)

    def get_structures_by_acronym(self, acronyms):
        if isinstance(acronyms, str):
            acronyms = [acronyms]
        results = []
        for acronym in acronyms:
            if acronym in self.structures_by_acronym:
                results.extend(self.structures_by_acronym[acronym])
        return results

    def get_structures_by_id(self, structure_ids):
        if isinstance(structure_ids, (int, np.integer)):
            structure_ids = [structure_ids]
        results = []
        for sid in structure_ids:
            if sid in self.structures_by_id:
                results.append(self.structures_by_id[sid])
        return results

    def get_ancestor_id_map(self, structure_ids):
        if isinstance(structure_ids, (int, np.integer)):
            structure_ids = [structure_ids]
        result = {}
        for sid in structure_ids:
            ancestors = self._get_ancestors(sid)
            result[sid] = ancestors
        return result

    def _get_ancestors(self, structure_id):
        ancestors = []
        current_id = structure_id
        visited = set()
        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            ancestors.append(current_id)
            if current_id in self.structures_by_id:
                structure = self.structures_by_id[current_id]
                parent_path = structure.get('structure_id_path')
                if parent_path and isinstance(parent_path, list) and len(parent_path) > 1:
                    current_id = parent_path[-2]
                else:
                    current_id = None
            else:
                current_id = None
        return ancestors


ALLEN_SDK_AVAILABLE = False
structure_tree = None

try:
    from allensdk.api.queries.ontologies_api import OntologiesApi
    oapi = OntologiesApi()
    structure_graph = oapi.get_structures_with_sets([1])
    try:
        from allensdk.core.structure_tree import StructureTree
        structure_tree = StructureTree(structure_graph)
        test = structure_tree.get_structures_by_acronym(['VPM'])
        if test:
            ALLEN_SDK_AVAILABLE = True
        else:
            raise ValueError("Structure tree is empty")
    except Exception:
        structure_tree = ManualStructureTree(structure_graph)
        test = structure_tree.get_structures_by_acronym(['VPM'])
        if test:
            ALLEN_SDK_AVAILABLE = True
        else:
            raise ValueError("Manual structure tree failed")
except Exception:
    ALLEN_SDK_AVAILABLE = False

RT_ID = 262
STR_ID = 477


def get_region_category_allen(region):
    if not ALLEN_SDK_AVAILABLE:
        return get_region_category_fallback(region)
    try:
        structures = structure_tree.get_structures_by_acronym([region])
        if not structures:
            return get_region_category_fallback(region)
        structure = structures[0]
        structure_id = structure['id']
        if structure_id == RT_ID:
            return 'RT'
        structure_id_path = structure.get('structure_id_path', [])
        if isinstance(structure_id_path, str):
            path_ids = [int(x) for x in structure_id_path.strip('/').split('/') if x]
        elif isinstance(structure_id_path, list):
            path_ids = structure_id_path
        else:
            path_ids = []
        if RT_ID in path_ids:
            return 'RT'
        if structure_id == STR_ID or STR_ID in path_ids:
            return 'STR'
        ancestor_ids = structure_tree.get_ancestor_id_map([structure_id])[structure_id]
        ancestors = structure_tree.get_structures_by_id(ancestor_ids)
        ancestor_names = [a['name'].lower() for a in ancestors]
        if any('motor' in name for name in ancestor_names) or \
           any(region.startswith(m) for m in ['MOp', 'MOs']):
            return 'Motor'
        if any('somatosensory' in name for name in ancestor_names) or \
           any(region.startswith(s) for s in ['SSp', 'SSs']):
            return 'Somatosensory'
        if any('thalamus' in name for name in ancestor_names):
            return 'Thalamus'
        return 'Other'
    except Exception:
        return get_region_category_fallback(region)


def get_region_category_fallback(region):
    region_upper = region.upper()
    if region_upper == 'RT':
        return 'RT'
    if region_upper in ['STR', 'CP', 'ACB', 'OT', 'LSX', 'LS', 'SF', 'SH',
                         'AAA', 'BA', 'CEA', 'MEA', 'IA'] or \
       region_upper.startswith('CP') or region_upper.startswith('ACB'):
        return 'STR'
    if region_upper.startswith('MOP') or region_upper.startswith('MOS') or \
       region_upper.startswith('M1') or region_upper.startswith('M2') or \
       region_upper.startswith('MO'):
        return 'Motor'
    elif region_upper.startswith('SSP') or region_upper.startswith('SSS') or \
         region_upper.startswith('S1') or region_upper.startswith('S2') or \
         region_upper.startswith('SS'):
        return 'Somatosensory'
    elif any(th in region_upper for th in [
        'VPM', 'VPL', 'VAL', 'VM', 'VL', 'PO', 'LP', 'LD',
        'CL', 'MD', 'MGM', 'MGD', 'MGV', 'LGD', 'LGV', 'TH',
        'IAD', 'IAM', 'IGL', 'INTG', 'POL', 'PT', 'SPF',
        'CM', 'PCN', 'IMD', 'PVT', 'ATN', 'AVD', 'AVV',
        'PF', 'PIL', 'SGN'
    ]):
        return 'Thalamus'
    else:
        return 'Other'


def find_bursts_THunit(spike_times):
    if len(spike_times) < 3:
        return np.array([]), np.array([])
    preISIs = np.diff(spike_times)[:-1]
    postISIs = np.diff(spike_times)[1:]
    bs_inds = np.nonzero((preISIs > 0.1) * (postISIs < 0.005))[0]
    if len(bs_inds) == 0:
        return np.array([]), np.array([])
    burst_starts = bs_inds + 1
    burst_counts = []
    for st_ind in bs_inds:
        spkind = st_ind + 1
        bcount = 1
        while (spkind < len(preISIs)) and (preISIs[spkind] < 0.004):
            spkind += 1
            bcount += 1
        burst_counts.append(bcount)
    return spike_times[burst_starts], np.array(burst_counts)


def find_bursts_otherunit(spike_times):
    ISI_threshold = 0.010  # 10 ms (changed from 12 ms per Stark et al.)
    spike_count_thresh = 3  # minimum 2 spikes (changed from 3)
    if len(spike_times) < spike_count_thresh:
        return np.array([]), np.array([])
    preISIs = np.insert(np.diff(spike_times), 0, 1.0)
    burst_starts = []
    burst_counts = []
    spkind = 0
    while spkind < len(spike_times):
        burst_start_idx = spkind
        tempevent = [spike_times[spkind]]
        spkind += 1
        while (spkind < len(spike_times)) and (preISIs[spkind] < ISI_threshold):
            tempevent.append(spike_times[spkind])
            spkind += 1
        if len(tempevent) >= spike_count_thresh and preISIs[burst_start_idx] >= 0.050:
            burst_starts.append(tempevent[0])
            burst_counts.append(len(tempevent))
        del tempevent
    return np.array(burst_starts), np.array(burst_counts)


def detect_bursts_for_unit(spike_times, brain_area):
    if brain_area in ('Thalamus', 'RT'):
        return find_bursts_THunit(spike_times)
    else:
        return find_bursts_otherunit(spike_times)


def compute_burst_metrics_for_periods(spike_times, spike_clusters, unit_ids,
                                     units_df, time_periods, bin_label=''):
    total_duration = sum([end - start for start, end in time_periods])

    burst_results = {
        'unit_id': [],
        'brain_area': [],
        'location': [],
        'n_bursts': [],
        'burst_rate': [],
        'total_spikes_in_bursts': [],
        'mean_burst_size': [],
        'total_spikes': [],
        'burst_fraction': [],
        'rest_duration': [],
        'bin': []
    }

    for unit_id in unit_ids:
        unit_info = units_df.loc[unit_id]
        brain_area = unit_info['brain_area']
        location = unit_info['location']

        if total_duration == 0:
            n_bursts = np.nan
            burst_rate = np.nan
            total_spikes_in_bursts = np.nan
            mean_burst_size = np.nan
            total_spikes = np.nan
            burst_fraction = np.nan
        else:
            unit_mask = spike_clusters == unit_id
            unit_spike_times = spike_times[unit_mask]

            period_spikes = []
            for start, end in time_periods:
                period_mask = (unit_spike_times >= start) & (unit_spike_times < end)
                ps = unit_spike_times[period_mask]
                if len(ps) > 0:
                    period_spikes.append(ps)

            total_spikes = sum(len(ps) for ps in period_spikes)

            all_burst_counts = []
            for ps in period_spikes:
                bs, bc = detect_bursts_for_unit(ps, brain_area)
                if len(bc) > 0:
                    all_burst_counts.append(bc)

            if len(all_burst_counts) > 0:
                burst_counts_combined = np.concatenate(all_burst_counts)
                n_bursts = len(burst_counts_combined)
                total_spikes_in_bursts = burst_counts_combined.sum()
                mean_burst_size = burst_counts_combined.mean()
                burst_fraction = total_spikes_in_bursts / total_spikes if total_spikes > 0 else 0
            else:
                n_bursts = 0
                total_spikes_in_bursts = 0
                mean_burst_size = 0
                burst_fraction = 0

            burst_rate = n_bursts / total_duration

        burst_results['unit_id'].append(unit_id)
        burst_results['brain_area'].append(brain_area)
        burst_results['location'].append(location)
        burst_results['n_bursts'].append(n_bursts)
        burst_results['burst_rate'].append(burst_rate)
        burst_results['total_spikes_in_bursts'].append(total_spikes_in_bursts)
        burst_results['mean_burst_size'].append(mean_burst_size)
        burst_results['total_spikes'].append(total_spikes)
        burst_results['burst_fraction'].append(burst_fraction)
        burst_results['rest_duration'].append(total_duration)
        burst_results['bin'].append(bin_label)

    return pd.DataFrame(burst_results)


def extract_units(file_path, apply_quality_filter=True,
                 isi_viol_thresh=0.5,
                 amp_cutoff_thresh=0.1,
                 spike_count_thresh=50,
                 rs_only=True):
    with h5py.File(file_path, 'r') as f:
        if 'units' in f:
            units_group = f['units']
            unit_data = {}
            lengths = []
            for key in units_group.keys():
                data = units_group[key][:]
                if data.ndim == 1:
                    if data.dtype.kind in ['S', 'O']:
                        unit_data[key] = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in data]
                    else:
                        unit_data[key] = data
                    lengths.append(len(data))
            if len(set(lengths)) > 1:
                min_len = min(lengths)
                for key in unit_data:
                    unit_data[key] = unit_data[key][:min_len]
            units_df = pd.DataFrame(unit_data)
            if 'spike_count' not in units_df.columns and 'num_spikes' not in units_df.columns:
                if 'spike_times_index' in units_df.columns:
                    spike_times_index = units_df['spike_times_index'].values
                    spike_counts = []
                    prev_idx = 0
                    for idx in spike_times_index:
                        spike_counts.append(idx - prev_idx)
                        prev_idx = idx
                    units_df['spike_count'] = spike_counts
            n_total = len(units_df)
            if apply_quality_filter:
                filter_mask = np.ones(len(units_df), dtype=bool)
                if 'isi_violations' in units_df.columns:
                    filter_mask &= units_df['isi_violations'] < isi_viol_thresh
                if 'amplitude_cutoff' in units_df.columns:
                    filter_mask &= units_df['amplitude_cutoff'] < amp_cutoff_thresh
                spike_col = 'spike_count' if 'spike_count' in units_df.columns else 'num_spikes'
                if spike_col in units_df.columns:
                    filter_mask &= units_df[spike_col] > spike_count_thresh
                if 'quality' in units_df.columns:
                    filter_mask &= units_df['quality'] == 'good'
                units_df = units_df[filter_mask].copy()
            if 'location' in units_df.columns:
                units_df['brain_area'] = units_df['location'].apply(get_region_category_allen)
            if rs_only and 'waveform_duration' in units_df.columns:
                exempt_mask = units_df['brain_area'].isin(['STR', 'RT'])
                rs_mask = exempt_mask | (units_df['waveform_duration'] > 0.4)
                units_df = units_df[rs_mask].copy()
            return units_df
        return None


def extract_spike_data(file_path):
    with h5py.File(file_path, 'r') as f:
        if 'units' in f:
            spike_times_data = f['units']['spike_times']
            spike_times_index = f['units']['spike_times_index'][:]
            all_spike_times = spike_times_data[:]
            all_spike_clusters = []
            start_idx = 0
            for unit_id, end_idx in enumerate(spike_times_index):
                num_spikes = end_idx - start_idx
                all_spike_clusters.extend([unit_id] * num_spikes)
                start_idx = end_idx
            return np.array(all_spike_times), np.array(all_spike_clusters)
    return None, None


def extract_speed(file_path):
    with h5py.File(file_path, 'r') as f:
        rs = f['processing/behavior/BehavioralTimeSeries/running_speed']
        speed_data = rs['data'][:]
        if 'timestamps' in rs:
            speed_timestamps = rs['timestamps'][:]
        elif 'starting_time' in rs.attrs:
            starting_time = rs.attrs['starting_time']
            sr = 100.0
            speed_timestamps = starting_time + np.arange(len(speed_data)) / sr
        else:
            sr = 100.0
            speed_timestamps = np.arange(len(speed_data)) / sr
        return speed_data, speed_timestamps


def get_injection_times(file_path):
    with h5py.File(file_path, 'r') as f:
        if 'intervals/epochs' not in f:
            return []
        epochs_group = f['intervals/epochs']
        start_times = epochs_group['start_time'][:]
        tags = epochs_group['tags'][:]
        tags_index = epochs_group['tags_index'][:]
        tags_decoded = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in tags]
        injections = []
        for i in range(len(start_times)):
            start_idx = tags_index[i-1] if i > 0 else 0
            end_idx = tags_index[i]
            injection_tags = tags_decoded[start_idx:end_idx]
            injections.append({'time': start_times[i], 'tags': injection_tags})
        return injections


def get_stimulus_times(file_path):
    with h5py.File(file_path, 'r') as f:
        # If no trial structure exists — purely spontaneous recording
        if 'intervals/trials/start_time' not in f:
            return np.array([])

        trial_times = f['intervals/trials/start_time'][:]

        if 'intervals/trials/stimulus_type' in f:
            # Evoked recording: filter to electrical + visual trials only
            stim_types = f['intervals/trials/stimulus_type'][:]
            stim_types = np.array([
                v.decode('utf-8') if isinstance(v, bytes) else str(v)
                for v in stim_types])
            stim_mask = np.isin(stim_types, ['electrical', 'visual'])
            return trial_times[stim_mask]
        else:
            # Trial structure exists but no stimulus_type — treat as spontaneous
            return np.array([])


def segment_by_state(speed_data, speed_timestamps, seg_start, seg_end, threshold=1.0):
    mask = (speed_timestamps >= seg_start) & (speed_timestamps <= seg_end)
    seg_speeds = speed_data[mask]
    seg_times = speed_timestamps[mask]
    if len(seg_speeds) == 0:
        return {'rest': [], 'movement': []}
    is_rest = seg_speeds < threshold
    rest_bouts = []
    movement_bouts = []
    in_rest = is_rest[0]
    bout_start = seg_times[0]
    for i in range(1, len(is_rest)):
        if is_rest[i] != in_rest:
            bout_end = seg_times[i]
            if bout_end - bout_start >= 1.0:
                if in_rest:
                    rest_bouts.append((bout_start, bout_end))
                else:
                    movement_bouts.append((bout_start, bout_end))
            in_rest = is_rest[i]
            bout_start = seg_times[i]
    bout_end = seg_times[-1]
    if bout_end - bout_start >= 1.0:
        if in_rest:
            rest_bouts.append((bout_start, bout_end))
        else:
            movement_bouts.append((bout_start, bout_end))
    return {'rest': rest_bouts, 'movement': movement_bouts}


def extract_pre_stimulus_periods(stim_times, analysis_start, analysis_end,
                                 pre_stim_exclude=0.1, post_stim_exclude=0.6):
    stim_in_window = stim_times[(stim_times >= analysis_start) & (stim_times <= analysis_end)]
    if len(stim_in_window) == 0:
        return [(analysis_start, analysis_end)]
    pre_stim_periods = []
    first_period_start = analysis_start
    first_period_end = stim_in_window[0] - pre_stim_exclude
    if first_period_end > first_period_start:
        pre_stim_periods.append((first_period_start, first_period_end))
    for i in range(len(stim_in_window) - 1):
        period_start = stim_in_window[i] + post_stim_exclude
        period_end = stim_in_window[i+1] - pre_stim_exclude
        if period_end > period_start:
            pre_stim_periods.append((period_start, period_end))
    last_period_start = stim_in_window[-1] + post_stim_exclude
    last_period_end = analysis_end
    if last_period_end > last_period_start:
        pre_stim_periods.append((last_period_start, last_period_end))
    return pre_stim_periods


def filter_periods_by_rest(pre_stim_periods, rest_bouts):
    rest_pre_stim_periods = []
    for ps_start, ps_end in pre_stim_periods:
        for rest_start, rest_end in rest_bouts:
            overlap_start = max(ps_start, rest_start)
            overlap_end = min(ps_end, rest_end)
            if overlap_end > overlap_start:
                rest_pre_stim_periods.append((overlap_start, overlap_end))
    return rest_pre_stim_periods


def analyze_session(file_path, output_dir):
    filename = os.path.basename(file_path).replace('.nwb', '')

    units_df = extract_units(file_path, rs_only=True)
    if units_df is None or len(units_df) == 0:
        raise ValueError("No units found or all units filtered out")

    spike_times, spike_clusters = extract_spike_data(file_path)

    try:
        speed_data, speed_timestamps = extract_speed(file_path)
    except:
        speed_data = None
        speed_timestamps = None

    injections = get_injection_times(file_path)
    if len(injections) < 2:
        raise ValueError(f"Expected 2 injections, found {len(injections)}")

    inj1_time = injections[0]['time']
    inj2_time = injections[1]['time']

    all_stim_times = get_stimulus_times(file_path)

    rec_start = spike_times[0]
    rec_end = spike_times[-1]

    # Use only the last 10 minutes before first injection as baseline
    baseline_duration_sec = 600  # 10 minutes
    baseline_end = inj1_time
    baseline_start = baseline_end - baseline_duration_sec

    if baseline_start < rec_start:
        raise ValueError(
            f"Session has < 10 min before first injection "
            f"(available: {(inj1_time - rec_start)/60:.1f} min). Skipping."
        )

    baseline_duration_min = (baseline_end - baseline_start) / 60

    baseline_stim = all_stim_times[(all_stim_times >= baseline_start) &
                                    (all_stim_times < baseline_end)]

    baseline_pre_stim = extract_pre_stimulus_periods(
        all_stim_times, baseline_start, baseline_end,
        pre_stim_exclude=0.1, post_stim_exclude=0.6
    )

    if speed_data is not None:
        states_baseline = segment_by_state(speed_data, speed_timestamps,
                                          baseline_start, baseline_end)
        rest_bouts_baseline = states_baseline['rest']
        baseline_rest_pre_stim = filter_periods_by_rest(baseline_pre_stim, rest_bouts_baseline)
    else:
        baseline_rest_pre_stim = baseline_pre_stim

    bin_size = 5
    n_baseline_bins = int(np.ceil(baseline_duration_min / bin_size))

    baseline_results = []
    for bin_idx in range(n_baseline_bins):
        bin_start_min = bin_idx * bin_size
        bin_end_min = min(bin_start_min + bin_size, baseline_duration_min)
        bin_start = baseline_start + bin_start_min * 60
        bin_end = baseline_start + bin_end_min * 60
        bin_label = f"baseline_{int(bin_start_min)}-{int(bin_end_min)}"
        bin_rest = [(max(s, bin_start), min(e, bin_end))
                   for s, e in baseline_rest_pre_stim
                   if e > bin_start and s < bin_end]
        bin_rest = [(s, e) for s, e in bin_rest if e > s]
        burst_df = compute_burst_metrics_for_periods(
            spike_times, spike_clusters, units_df.index.values,
            units_df, bin_rest if bin_rest else [], bin_label=bin_label
        )
        baseline_results.append(burst_df)

    effect_start = inj2_time
    effect_end = rec_end  # no hard cap — keep all available data
    effect_duration_min = (effect_end - effect_start) / 60

    effect_stim = all_stim_times[(all_stim_times >= effect_start) &
                                  (all_stim_times < effect_end)]

    effect_pre_stim = extract_pre_stimulus_periods(
        all_stim_times, effect_start, effect_end,
        pre_stim_exclude=0.1, post_stim_exclude=0.6
    )

    if speed_data is not None:
        states_effect = segment_by_state(speed_data, speed_timestamps,
                                        effect_start, effect_end)
        rest_bouts_effect = states_effect['rest']
        effect_rest_pre_stim = filter_periods_by_rest(effect_pre_stim, rest_bouts_effect)
    else:
        effect_rest_pre_stim = effect_pre_stim

    n_effect_bins = int(np.ceil(effect_duration_min / bin_size))

    effect_results = []
    for bin_idx in range(n_effect_bins):
        bin_start_min = bin_idx * bin_size
        bin_end_min = min(bin_start_min + bin_size, effect_duration_min)
        bin_start = effect_start + bin_start_min * 60
        bin_end = effect_start + bin_end_min * 60
        bin_label = f"effect_{int(bin_start_min)}-{int(bin_end_min)}"
        bin_rest = [(max(s, bin_start), min(e, bin_end))
                   for s, e in effect_rest_pre_stim
                   if e > bin_start and s < bin_end]
        bin_rest = [(s, e) for s, e in bin_rest if e > s]
        burst_df = compute_burst_metrics_for_periods(
            spike_times, spike_clusters, units_df.index.values,
            units_df, bin_rest if bin_rest else [], bin_label=bin_label
        )
        effect_results.append(burst_df)

    all_bins_df = pd.concat(baseline_results + effect_results, ignore_index=True)

    all_bins_df.to_csv(os.path.join(output_dir, f'{filename}_prestim_burst_timecourse.csv'))
    units_df.to_csv(os.path.join(output_dir, f'{filename}_prestim_units_info.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output_prestim_burst')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    analyze_session(args.file, args.output_dir)