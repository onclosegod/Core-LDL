# Core-LDL

Core-LDL is a PyTorch implementation for learning label distributions from incomplete label degrees.

## Installation

Use Python 3.8 and install the dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

Training uses a CUDA GPU when available and otherwise runs on CPU.

## Datasets

The benchmark datasets are obtained from [SpriteMisaka/PyLDL](https://github.com/SpriteMisaka/PyLDL). Download the required `.mat` files from its [datasets directory](https://github.com/SpriteMisaka/PyLDL/tree/main/datasets) using **Download raw file** on the GitHub file page.

The datasets used by this project are `fbp5500`, `Movie`, `RAF_ML`, `M2B`, `Yeast_heat`, `Natural_Scene`, `Painting`, and `Music`.

Create a `dataset/` folder in the project root and place the downloaded files there. Each file must contain `features` and `labels` arrays. For example:

```text
Core-LDL/
├── dataset/
│   └── fbp5500.mat
├── train_torch.py
├── myloss_psd_projection.py
├── impute_loss.py
├── train_torch.sh
├── requirements.txt
└── README.md
```

We thank the PyLDL maintainers and the original dataset authors for providing the data. Please acknowledge their work and follow the original dataset terms when using these resources.

## Run

From the project root, run:

```bash
python train_torch.py --dataset fbp5500 --dataset-dir ./dataset --output-dir ./outputs/main
```

To use another dataset, replace `fbp5500` with its filename without the `.mat` extension. If the data is stored elsewhere, change `--dataset-dir` to that directory.

Training results, logs, and model checkpoints are saved under `outputs/main/`.

For the supplied four-GPU example on Linux, download `RAF_ML.mat` into `dataset/` and run:

```bash
bash train_torch.sh
```

To view all available options:

```bash
python train_torch.py --help
```
