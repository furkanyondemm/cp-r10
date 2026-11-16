# CP-R10: coverage-targeted temperature transfer

Code for **Learning Coverage-Targeted Temperatures for Label-Free Prediction Sets**.
CP-R10 learns inverse temperatures from labelled synthetic source shifts and
unlabelled batch statistics. This repository provides the paper's CIFAR and
ImageNet experiments, including the QTC wrapper.

## Install

Run from the repository root using Python 3.10 in your activated environment:

```bash
python -m pip install -r requirements.txt
python fetch_qtc.py
./run.sh check
python -m unittest discover -s tests -v
```

Use a PyTorch build compatible with your GPU driver. For CPU execution, add
`--device cpu` to the run commands. `fetch_qtc.py` downloads and verifies the
pinned QTC author code used by our wrapper in `cpr10/qtc.py`.

## Data

Provide the extracted datasets in this layout. `--download` downloads the
original CIFAR data; corruption datasets must be supplied separately.

```text
data/
  CIFAR-10-C/gaussian_noise.npy, ..., labels.npy
  CIFAR-100-C/gaussian_noise.npy, ..., labels.npy
  imagenet-val/n01440764/ILSVRC2012_val_00000293.JPEG, ...
  ImageNet-C/gaussian_noise/1/n01440764/ILSVRC2012_val_00000293.JPEG, ...
  imagenetv2-matched-frequency/0/*.jpeg, ..., 999/*.jpeg
```

ImageNet uses the ILSVRC2012 **validation** set, ImageNet-C and ImageNetV2
matched-frequency. For validation images without class folders, see the
[devkit instructions](PROTOCOL.md#data-and-run-options).

## Reproduce

```bash
./run.sh run --system c10_resnet18 --work runs/c10 \
  --data data --corruptions data/CIFAR-10-C --download

./run.sh run --system c100_vit_b_16 --work runs/c100 \
  --data data --corruptions data/CIFAR-100-C --download

./run.sh run --system imagenet_resnet50 --work runs/in-r50 \
  --data data/imagenet-val --corruptions data/ImageNet-C \
  --v2 data/imagenetv2-matched-frequency

./run.sh run --system imagenet_vit_b_16 --work runs/in-vit \
  --data data/imagenet-val --corruptions data/ImageNet-C \
  --v2 data/imagenetv2-matched-frequency
```

Each command prepares predictions, fits the temperature predictors and source
corrections, and evaluates the paper experiments. CIFAR first trains its
classifier; ImageNet uses frozen torchvision `IMAGENET1K_V1` weights.
To reuse a CIFAR classifier, add `--checkpoint /path/to/best.pt`
(a checkpoint containing `model_state`).

## Results and protocol

Results are saved under `<work>/results/`. [PROTOCOL.md](PROTOCOL.md) describes
the experimental settings, output files and options for individual stages
and reusing completed work.

## License

Copyright 2026 Sakir Furkan Yöndem. The original CP-R10 code, including the
QTC wrapper, is licensed under the [Apache License 2.0](LICENSE).
Third-party code retains its own attribution and license notices; see
[THIRD_PARTY.md](THIRD_PARTY.md).
