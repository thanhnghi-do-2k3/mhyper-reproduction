# M-Hyper reproduction study

> Scope: this repository is a research reproduction scaffold, not evidence of a complete paper reproduction. Full clean-environment installation, dataset preparation, and GPU reruns remain unverified.

## Reported hardware and environment limits

The group's rental record for the M-Hyper attempt lists Windows 10, an Intel Core i5 (generation not recorded), and an NVIDIA RTX 5070. This describes the rented machine; it does not establish the exact hardware or software used by every historical run. The W&B export did not preserve the historical Python, PyTorch, or CUDA versions. The commands below use Python 3.12 as a setup choice, not as a verified historical version.

The [M-Hyper W&B project](https://forge.coreweave.com/wandb/nghidt-hcmus/mhyper-link-completion/workspace?nw=c9t57at1lic) contains 37 historical runs and was checked in an incognito session on 14 November 2026; runs and charts were visible. Moving runs did not move their associated artifacts. This confirms access to the logged histories, not benchmark equivalence or complete artifacts.

The upstream M-Hyper source is not redistributed in this repository because its public repository does not specify a license. Obtain the source directly from the authors at https://github.com/zjukg/M-Hyper and place its checkout at `third_party/m_hyper/source/` before running the adapter. This project does not include the group's local patch to upstream files. The historical run provenance and exact upstream changes remain unverified.

Install from the source checkout in editable mode as shown below. A regular non-editable wheel install was checked and resolves the upstream-source directory inside the Python environment, where it does not exist; that installation mode is not supported by this package layout.

This repository contains the group's experiment registry, adapter, configurations, and scripts for M-Hyper. Dataset files and the dataset workspace are not included and must be prepared separately. Results must be labeled according to the actual split and artifact provenance.

## Windows GPU setup

```powershell
cd C:\Users\<USER>\mhyper-reproduction
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install numpy scikit-learn tqdm fastapi pydantic uvicorn wandb
```

Verify CUDA and authenticate W&B:

```powershell
python -c "import torch; print('CUDA=',torch.cuda.is_available()); print('GPU=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
wandb login
```

## Register datasets

```powershell
$env:MHYPER_WORKSPACE="$PWD\workspace_vm"
$env:PYTHONPATH="$PWD\src"
python -m mhyper_lab.cli data register --name DB15K --path "$PWD\workspace_vm\upstream_data\DB15K"
python -m mhyper_lab.cli data register --name MKG-W --path "$PWD\workspace_vm\upstream_data\MKG-W"
python -m mhyper_lab.cli data register --name MKG-Y --path "$PWD\workspace_vm\upstream_data\MKG-Y"
```

## Main-suite command and split warning

Do not run the suite below unchanged on the locally inspected DB15K files: `run_paper1_full_suite.py` selects `configs/reproduction/db15k_paper.toml` with `validation_source = "provided"`, while those files have overlapping validation data and the current split checker rejects them. `configs/reproduction/db15k.toml` specifies the train-holdout variant used by one archived seed-42 run. The exact configuration of the historical W&B multi-seed runs remains unverified. A full rerun requires an independently checked dataset and split policy.

The following is the original Windows suite invocation for reference, not a verified end-to-end rerun from this repository. New main, ablation and robustness runs belong in the [M-Hyper W&B project](https://forge.coreweave.com/wandb/nghidt-hcmus/mhyper-link-completion/workspace?nw=c9t57at1lic):

```powershell
$env:MHYPER_WORKSPACE="$PWD\workspace_vm"
$env:WANDB_MODE="online"
$env:WANDB_PROJECT="mhyper-link-completion"
$env:WANDB_RUN_GROUP="paper1-main-seed42-ep200"
python scripts\run_paper1_full_suite.py --continue-on-error 2>&1 |
  Tee-Object -FilePath "$PWD\paper1_full_suite.log"
```

The example configurations specify M-Hyper-B, rank 128, Adagrad, learning rate 0.1, batch size 1000, N3 weight 0.005, validation every five epochs, and 200 configured epochs. `db15k.toml` specifies a train-holdout policy; `db15k_paper.toml`, which the displayed suite selects, specifies the provided split. The selected checkpoint is the best validation-MRR checkpoint, not necessarily epoch 200.

## Inspect results

```powershell
Get-Content .\paper1_full_suite.log -Tail 50
Get-ChildItem .\workspace_vm -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 20
```

Use the W&B project and run links in the report. Do not call CPU smoke tests or failed split runs official results.

The `WANDB_PROJECT` setting above selects the destination for new runs. It does not move existing runs. This repository does not include the datasets, cached features or experiment logs.

## Minimal local checks

Run from this folder after installing the project and pytest:

```powershell
python -m pip install pytest
$env:WANDB_MODE="disabled"
python -m pytest -q -p no:cacheprovider
python -m mhyper_lab.cli --help
```

These checks do not reproduce the paper's metrics. Each registered dataset directory requires `train`, `valid`, `test`, `entity2id.txt`, `relation2id.txt`, `img_features.pth` and `text_features.pth`. Obtain them from sources for which the group has usage rights; the official upstream reference is https://github.com/zjukg/M-Hyper .

For a bounded CPU check with the locally inspected DB15K data, register the dataset in a separate workspace and run the explicitly corrected pilot:

```powershell
$env:MHYPER_WORKSPACE="$PWD\workspace_cpu_smoke"
$env:WANDB_MODE="disabled"
python -m mhyper_lab.cli data register --name DB15K --path <DB15K_DIRECTORY>
$runId = python -m mhyper_lab.cli run create --config configs/reproduction/db15k_cpu_holdout_smoke.toml
python -m mhyper_lab.cli run start --id $runId
python -m mhyper_lab.cli run show --id $runId
```

The worker starts asynchronously; inspect the run again until it finishes. The pilot uses a 10% train holdout, one epoch, at most 512 training examples and eight evaluation queries. It checks execution only and does not establish the provenance of prior runs. The older `db15k_cpu_pilot.toml` uses the provided validation split by default and is rejected with the overlapping local DB15K files.
