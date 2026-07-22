# Brain-wide reconfiguration of burst firing by psilocybin reveals 5-HT2A-dependent circuit dynamics

This repository contains the code and processed data to **reproduce the results** of the paper:

> **Brain-wide reconfiguration of burst firing by psilocybin reveals 5-HT2A-dependent circuit dynamics.**
> D. Momi\*, Y. Nahas\*, L. Marks, D. Wyrick, L. Claar, R. De Filippo, P. Seyfourian, D. Pizzagalli, M. Buice, T. Ott, C. Koch, I. Rembado.
> (\*equal contribution; corresponding author: irene.rembado@alleninstitute.org)

![Experimental design and burst metrics](assets/figure1.png)

*Head-fixed Neuropixels + EEG recordings during acute psilocybin. (a–b) Recording configuration: up to four Neuropixels probes across cortical and subcortical regions with a 30-channel scalp EEG array. (c) Unit yield by region. (d) Three session types: saline→saline, saline→psilocybin, and ketanserin→psilocybin. (e) Example session: locomotion, pupil, EEG/LFP spectra, and unit raster across the two injections. (f) Region-specific burst definitions (cortex/hippocampus vs. thalamus). (g) The four burst metrics: proportion of bursting units, burst rate, burst fraction, mean burst size. (h) Baseline normalization, drug–saline contrast, and hierarchical bootstrap.*

---

## Overview

Using simultaneous multi-region Neuropixels recordings (46,360 single units from 35 mice), together with scalp EEG, pupillometry, and locomotion, the study characterizes the acute, brain-wide effects of psilocybin and dissects their receptor dependence with the 5-HT2A antagonist ketanserin. The central finding is that psilocybin reconfigures **burst coding** rather than mean firing rate across the cortico-striato-thalamo-cortical circuit, and that these burst effects are abolished by ketanserin.

This repository reproduces the two single-unit analysis families in the paper:

1. **Burst analysis** — region-specific burst detection and four burst metrics.
2. **Firing-rate analysis** — mean firing rate per unit across the same regions.

Both are quantified as an absolute change from a pre-injection baseline, contrasted against saline, and tested with a hierarchical bootstrap.

## Experimental protocol (brief)

Mice were head-fixed on a running wheel for ~2 hours while locomotion, pupil, EEG, LFP, and single-unit activity were recorded. Each session used one of three drug conditions, delivered as two intraperitoneal injections ~10–12 min apart:

- **Saline**: saline, then saline.
- **Psilocybin**: saline, then psilocybin (1 mg/kg).
- **Ketanserin+Psilocybin (Ket+Psi)**: ketanserin (1 mg/kg), then psilocybin (1 mg/kg).

For the unit analyses, the **baseline** is the last 10 min before the first injection, and the **effect window** is the first 60 min after the second injection. Analyses are restricted to resting epochs (speed < 1 cm/s).

## Analyses (brief)

- **Burst detection** is region-specific. Cortex/hippocampus: ≥3 spikes with ISI < 10 ms, preceded by ≥50 ms silence. Thalamus: low-threshold calcium-spike criteria (≥2 spikes, ≥100 ms preceding silence, first intra-burst ISI < 5 ms). Metrics: proportion of bursting units, burst rate, burst fraction, mean burst size.
- **Firing rate** is spikes divided by analyzed rest time per 5-min bin, expressed as absolute change from baseline.
- **Statistics**: hierarchical bootstrap resampling sessions then units (10,000 iterations); drug–saline contrast; Benjamini–Hochberg FDR correction for the collapsed distribution plots. The proportion-of-bursting-units metric is a session-level aggregate and is bootstrapped at the session level only.

---

## Repository structure

```
psilocybin-burst-firing/
├── README.md
├── environment.yml
├── assets/
│   └── figure1.png
├── src/
│   ├── extraction/
│   │   ├── burst_analysis_10min_baseline_slience50ms.py   # burst extraction (per session)
│   │   └── firing_rate_10min_baseline.py                  # firing-rate extraction (per session)
│   └── figures/
│       ├── heatmap_hier_bootstrap.py                          # burst — major regions
│       ├── heatmap_hpf_thalamic_nuclei.py                     # burst — HPF & thalamic nuclei
│       ├── heatmap_prefrontal_sensory.py                      # burst — cortical layers
│       ├── distribution_layers_thalamus_hpf_40_60_propfix.py  # burst — layer/thalamic/HPF distributions
│       ├── distribution_striatum_IR_40_60_propfix.py          # burst — striatum distribution
│       ├── heatmap_hier_bootstrap_firing_rate.py              # firing rate — major regions
│       ├── heatmap_hpf_thalamic_nuclei_firing_rate.py         # firing rate — HPF & thalamic nuclei
│       ├── heatmap_prefrontal_sensory_firing_rate.py          # firing rate — cortical layers
│       ├── distribution_layers_thalamus_hpf_firing_rate_40_60_IR.py  # firing rate — 40–60 min window (paper)
│       └── distribution_layers_thalamus_hpf_firing_rate.py           # firing rate — 25–50 min window (alt.)
└── data/
    └── processed/
        ├── burst/
        │   ├── psilocybin/     # *_prestim_burst_timecourse.csv, *_prestim_units_info.csv
        │   ├── saline/
        │   └── ketanserin/
        └── firing_rate/
            ├── psilocybin/     # *_prestim_firing_rate_timecourse.csv, *_prestim_units_info.csv
            ├── saline/
            └── ketanserin/
```

The CSVs under `data/processed/` are the output of the extraction scripts and are committed so the figures can be reproduced without re-running extraction. Burst and firing-rate CSVs are kept in separate folders because the heatmap loaders glob `*_timecourse.csv`.

---

## 1. Get the repository

```bash
git clone git@github.com:Davi1990/psilocybin-burst-firing.git
cd psilocybin-burst-firing
```

## 2. Set up the environment

Requires [conda](https://docs.conda.io/en/latest/miniconda.html).

```bash
conda env create -f environment.yml
conda activate allen_nwb
```

## 3. Reproduce the figures (fast path — from committed CSVs)

No download required; the processed CSVs are in the repo.

**Burst figures:**
```bash
python src/figures/heatmap_hier_bootstrap.py                         --base_dir data/processed/burst
python src/figures/heatmap_hpf_thalamic_nuclei.py                    --base_dir data/processed/burst
python src/figures/heatmap_prefrontal_sensory.py                     --base_dir data/processed/burst
python src/figures/distribution_layers_thalamus_hpf_40_60_propfix.py --data_dir  data/processed/burst
python src/figures/distribution_striatum_IR_40_60_propfix.py         --data_dir  data/processed/burst
```

**Firing-rate figures:**
```bash
python src/figures/heatmap_hier_bootstrap_firing_rate.py                        --base_dir data/processed/firing_rate
python src/figures/heatmap_hpf_thalamic_nuclei_firing_rate.py                   --base_dir data/processed/firing_rate
python src/figures/heatmap_prefrontal_sensory_firing_rate.py                    --base_dir data/processed/firing_rate
python src/figures/distribution_layers_thalamus_hpf_firing_rate_40_60_IR.py     --data_dir  data/processed/firing_rate
```

## 4. Reproduce from raw data (full path — from NWB)

The raw single-unit recordings will be released in NWB format on the DANDI Archive (link added on publication). Download the sessions into `data/raw/`, then run extraction per session, writing each session's CSVs into the matching condition folder. For example, for a psilocybin session:

```bash
python src/extraction/burst_analysis_10min_baseline_slience50ms.py \
    --file data/raw/<session>.nwb --output_dir data/processed/burst/psilocybin

python src/extraction/firing_rate_10min_baseline.py \
    --file data/raw/<session>.nwb --output_dir data/processed/firing_rate/psilocybin
```

Repeat for every session, sending saline sessions to `saline/` and ketanserin+psilocybin sessions to `ketanserin/`. Then run the figure commands in step 3.

---

## Data availability

Raw data will be made publicly available in Neurodata Without Borders (NWB) format on the DANDI Archive upon publication.

## Funding

Templeton World Charity Foundation (TWCF-2022-30262), the Tiny Blue Dot Foundation, and the Allen Institute.
