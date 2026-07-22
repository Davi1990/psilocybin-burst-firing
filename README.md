# Brain-wide reconfiguration of burst firing by psilocybin

Code and processed data to reproduce the figures.

## Environment
    conda env create -f environment.yml
    conda activate allen_nwb

## Two ways to reproduce

**Fast (from processed data, no download):**
The CSVs in `data/processed/` are the output of the extraction step.
    python src/figures/heatmap_hier_bootstrap.py --base_dir data/processed
    python src/figures/heatmap_hpf_thalamic_nuclei.py --base_dir data/processed
    python src/figures/heatmap_prefrontal_sensory__1_.py --base_dir data/processed
    python src/figures/distribution_layers_thalamus_hpf_40_60_propfix.py --data_dir data/processed
    python src/figures/distribution_striatum_IR_40_60_propfix.py --data_dir data/processed

**Full (from raw NWB):**
Download the NWB sessions from DANDI (link added on publication) into `data/raw/`, then
run `src/extraction/burst_analysis_10min_baseline_slience50ms.py` per session
(`--file <nwb> --output_dir <condition folder>`) to regenerate the CSVs.
