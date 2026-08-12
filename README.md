# Brain-wide reconfiguration of burst firing by psilocybin reveals 5-HT2A-dependent circuit dynamics

This repository contains the source code and a demo dataset to **reproduce the quantitative results** of the paper.

**Authors**

D. Momi<sup>1,\*</sup>, Y. Nahas<sup>2,\*</sup>, D. Wyrick<sup>3,+</sup>, L. C. Marks<sup>4,+</sup>, L. D. Claar<sup>4</sup>, R. De Filippo<sup>6</sup>, P. Seyfourian<sup>7</sup>, D. A. Pizzagalli<sup>1</sup>, M. Buice<sup>3</sup>, T. Ott<sup>6</sup>, C. Koch<sup>4,8</sup>, I. Rembado<sup>4</sup>

<sup>\*</sup> First authors, equal contribution &nbsp;&nbsp; <sup>+</sup> Second authors, equal contribution
Corresponding author: irene.rembado@alleninstitute.org

<sup>1</sup> Noel Drury, M.D. Institute for Translational Depression Discoveries, University of California, Irvine, USA
<sup>2</sup> University of Chicago, Chicago, USA
<sup>3</sup> Allen Institute, Studio D3, Seattle, WA, USA
<sup>4</sup> Allen Institute, Brain and Consciousness, Seattle, WA, USA
<sup>6</sup> Humboldt-Universität zu Berlin, Bernstein Center for Computational Neuroscience Berlin and Institute of Biology, Berlin, Germany
<sup>7</sup> University of British Columbia, Vancouver, Canada
<sup>8</sup> Tiny Blue Dot Foundation, Santa Monica, CA, USA

![Experimental design and burst metrics](assets/figure1.png)

*Head-fixed Neuropixels + EEG recordings during acute psilocybin. (a-b) Recording configuration: up to four Neuropixels probes across cortical and subcortical regions with a 30-channel scalp EEG array. (c) Unit yield by region. (d) Three session types: saline->saline, saline->psilocybin, and ketanserin->psilocybin. (e) Example session: locomotion, pupil, EEG/LFP spectra, and unit raster across the two injections. (f) Region-specific burst definitions (cortex/hippocampus vs. thalamus). (g) The four burst metrics: proportion of bursting units, burst rate, burst fraction, mean burst size. (h) Baseline normalization, drug-saline contrast, and hierarchical bootstrap.*

---

## Overview

Using simultaneous multi-region Neuropixels recordings (46,360 single units from 35 mice), together with scalp EEG, pupillometry, and locomotion, the study characterizes the acute, brain-wide effects of psilocybin and dissects their receptor dependence with the 5-HT2A antagonist ketanserin. The central finding is that psilocybin reconfigures **burst coding** rather than mean firing rate across the cortico-striato-thalamo-cortical circuit, and that these burst effects are abolished by ketanserin.

The repository reproduces the two single-unit analysis families in the paper: a **burst analysis** (region-specific burst detection and four burst metrics) and a **firing-rate analysis** (mean firing rate per unit). Both are computed as an absolute change from a pre-injection baseline, contrasted against saline, and tested with a hierarchical bootstrap.

### Experimental protocol (brief)

Mice were head-fixed on a running wheel for ~2 hours while locomotion, pupil, EEG, LFP, and single-unit activity were recorded. Each session used one of three drug conditions, delivered as two intraperitoneal injections ~10-12 min apart: **Saline** (saline, then saline), **Psilocybin** (saline, then psilocybin 1 mg/kg), or **Ketanserin+Psilocybin** (ketanserin 1 mg/kg, then psilocybin 1 mg/kg). For the unit analyses the baseline is the last 10 min before the first injection, the effect window is the first 60 min after the second injection, and analyses are restricted to resting epochs (speed < 1 cm/s).

### Analyses (brief)

- **Burst detection** is region-specific. Cortex/hippocampus: >=3 spikes with ISI < 10 ms, preceded by >=50 ms silence. Thalamus: low-threshold calcium-spike criteria (>=2 spikes, >=100 ms preceding silence, first intra-burst ISI < 5 ms). Metrics: proportion of bursting units, burst rate, burst fraction, mean burst size.
- **Firing rate** is spikes divided by analyzed rest time per 5-min bin, expressed as absolute change from baseline.
- **Statistics**: hierarchical bootstrap resampling sessions then units (10,000 iterations); drug-saline contrast; Benjamini-Hochberg FDR correction for the collapsed distribution plots. The proportion-of-bursting-units metric is a session-level aggregate and is bootstrapped at the session level only.

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
│       ├── heatmap_hier_bootstrap.py                          # burst - major regions
│       ├── heatmap_hpf_thalamic_nuclei.py                     # burst - HPF & thalamic nuclei
│       ├── heatmap_prefrontal_sensory.py                      # burst - cortical layers
│       ├── distribution_layers_thalamus_hpf_40_60_propfix.py  # burst - layer/thalamic/HPF distributions
│       ├── distribution_striatum_IR_40_60_propfix.py          # burst - striatum distribution
│       ├── heatmap_hier_bootstrap_firing_rate.py              # firing rate - major regions
│       ├── heatmap_hpf_thalamic_nuclei_firing_rate.py         # firing rate - HPF & thalamic nuclei
│       ├── heatmap_prefrontal_sensory_firing_rate.py          # firing rate - cortical layers
│       ├── distribution_layers_thalamus_hpf_firing_rate_40_60_IR.py  # firing rate - 40-60 min window (paper)
│       └── distribution_layers_thalamus_hpf_firing_rate.py           # firing rate - 25-50 min window (alt.)
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

The CSVs under `data/processed/` are the output of the extraction scripts and serve as the demo dataset, so the figures can be reproduced without re-running extraction. Burst and firing-rate CSVs are kept in separate folders because the heatmap loaders glob `*_timecourse.csv`.

---

## 1. System requirements

**Software dependencies and versions.** All Python dependencies and their exact versions are pinned in [`environment.yml`](environment.yml), which is a direct `conda env export` of the environment used to produce the results. Core libraries: `python`, `numpy`, `pandas`, `h5py`, `scipy`, `matplotlib`. The exact Python version is recorded in `environment.yml` (line beginning `- python=`).

**Operating systems tested.** `<Tested on macOS 15 (Sequoia), Apple Silicon (M-series)>`.

**Non-standard hardware.** None required to run the analysis code; it runs on a normal desktop or laptop. (Neuropixels probes and the EEG array were used for data *acquisition* only and are not needed to run this software.)

## 2. Installation guide

Requires [conda](https://docs.conda.io/en/latest/miniconda.html).

```bash
git clone git@github.com:Davi1990/psilocybin-burst-firing.git
cd psilocybin-burst-firing
conda env create -f environment.yml
conda activate allen_nwb
```

**Typical install time** on a normal desktop: `<~10 minutes on a normal desktop with a broadband connection>` (dominated by conda solving and downloading dependencies).

## 3. Demo

The processed CSVs in `data/processed/` are the demo dataset. To generate the figures from them:

**Burst figures**
```bash
python src/figures/heatmap_hier_bootstrap.py                         --base_dir data/processed/burst
python src/figures/heatmap_hpf_thalamic_nuclei.py                    --base_dir data/processed/burst
python src/figures/heatmap_prefrontal_sensory.py                     --base_dir data/processed/burst
python src/figures/distribution_layers_thalamus_hpf_40_60_propfix.py --data_dir  data/processed/burst
python src/figures/distribution_striatum_IR_40_60_propfix.py         --data_dir  data/processed/burst
```

**Firing-rate figures**
```bash
python src/figures/heatmap_hier_bootstrap_firing_rate.py                    --base_dir data/processed/firing_rate
python src/figures/heatmap_hpf_thalamic_nuclei_firing_rate.py               --base_dir data/processed/firing_rate
python src/figures/heatmap_prefrontal_sensory_firing_rate.py                --base_dir data/processed/firing_rate
python src/figures/distribution_layers_thalamus_hpf_firing_rate_40_60_IR.py --data_dir  data/processed/firing_rate
```

**Expected output.** Each script writes figure image files (`.png`, 300 dpi) plus, for the distribution scripts, the per-region `*_delta_stats.csv` tables of effect sizes and FDR-corrected p-values. The heatmaps show the drug-saline contrast across 5-min bins per region/nucleus/layer; the distribution plots show the bootstrap group-mean half-violins with Cohen's *d* and significance markers for the collapsed effect window.

**Expected run time** for the demo on a normal desktop: `<~2–5 minutes per script with the default --nboots 10000; under ~1 minute with --nboots 1000>`. Run time scales with the bootstrap count; pass `--nboots 1000` for a faster check.

## 4. Instructions for use (running on your own data)

The raw single-unit recordings will be released in NWB format on the DANDI Archive (link added on publication). Download the sessions into `data/raw/`, then run each extraction script per session, writing each session's CSVs into the matching condition folder. For a psilocybin session:

```bash
python src/extraction/burst_analysis_10min_baseline_slience50ms.py \
    --file data/raw/<session>.nwb --output_dir data/processed/burst/psilocybin

python src/extraction/firing_rate_10min_baseline.py \
    --file data/raw/<session>.nwb --output_dir data/processed/firing_rate/psilocybin
```

Send saline sessions to `saline/` and ketanserin+psilocybin sessions to `ketanserin/`.

## 5. Reproduction instructions

To reproduce all quantitative results in the manuscript from scratch: complete the installation (section 2), extract every session from the DANDI NWB files into the two `data/processed/` trees (section 4), then run every figure command (section 3). The figure scripts regenerate the statistics (effect sizes and FDR-corrected p-values in the `*_delta_stats.csv` tables) reported in the paper. To reproduce only the figures and statistics without re-extracting, run section 3 directly against the committed CSVs.

---

## Data availability

Raw data will be made publicly available in Neurodata Without Borders (NWB) format on the DANDI Archive upon publication.

## Funding

Templeton World Charity Foundation (TWCF-2022-30262), the Tiny Blue Dot Foundation, and the Allen Institute.
