# Experimental protocol

All systems use 15 corruption types, five severities and alpha in {0.05, 0.10}.
Regressors and scalar corrections use source labels only. Classifiers stay
fixed during adaptation. The posterior-weighted set rule has no
distribution-free coverage guarantee under shift.

## Methods and source roles

Every system uses 320 synthetic conditions (eight transformation families × 40).
Conditions reuse 1,000 design and 1,500 feature identities; another
2,500 clean source images calibrates QTC. All three roles are disjoint, and
feature-role labels are unused. Design images supply both the source cutoff and
labelled loss, so this loss is not held-out coverage.

| Method | Source target / predictor |
|---|---|
| CP | Minimum absolute coverage error; ties prefer mean set size, then beta; one ridge predictor per alpha |
| NLL | Golden-section NLL search with endpoint checks; one predictor for both alphas |
| CP-grid / NLL-grid | Same 93 beta candidates; loss ties prefer smallest beta |
| Shared-CP | Minimum maximum error across alphas, then mean error, then beta; one predictor |
| NLL-beta / NLL-alpha | Source-selected scalar correction to the original NLL beta or threshold level |
| QTC / QTC-cap | Author routines and wrapper boundary handling; see [THIRD_PARTY.md](THIRD_PARTY.md) |

Ten batch features are standardised across source conditions. Ridge penalty is
0.1 with an unpenalised intercept; predicted betas are clipped to [0,20].
CP uses a finite search with local refinements. Scalar corrections minimise
mean absolute source coverage error, then mean set size, then absolute offset.

## Classifiers and splits

| System | Classifier training | Source / target |
|---|---|---|
| CIFAR-10 / ResNet-18 | Scratch, seed 42, 100 epochs, batch 128, SGD lr .1 / momentum .9 / decay 5e-4, cosine schedule | Original 40k train / 5k development indices; 256 pilot test IDs excluded |
| CIFAR-100 / ViT-B/16 | V1 fine-tuning, seed 20260908, 20 epochs, batch 64, AdamW backbone 3e-5 / head 3e-4, decay .05, two-epoch warmup then cosine, smoothing .1, CUDA AMP | Original 45k train / 5k development indices |
| ImageNet / ResNet-50 and ViT-B/16 | Frozen torchvision IMAGENET1K_V1 | Five validation images/class reserved for sources; 45k target identities |

CIFAR checkpoints minimise development cross-entropy (CIFAR-100 breaks ties by
accuracy). Exact split indices are in `cpr10/assets/`. CIFAR-10 uses a
small-image ResNet stem; CIFAR-100 uses bicubic tensor resizing to 224×224.
ImageNet clean/source/V2 images use bilinear short-side resize 256 and centre
crop 224; ImageNet-C uses centre crop 224 without resizing. Synthetic
transformations precede normalisation.

## Paper-to-command map

Bf is the feature pool, Bq the threshold pool and E the evaluation pool.
Slashed sizes follow CIFAR-10 / CIFAR-100 / ImageNet.

| Flag | Paper evidence | Pools and repeats |
|---|---|---|
| `main` | Table 1, scalar corrections, ImageNet QTC cap | Bf=Bq=E=1500; 50 paired draws |
| `grid` | Table 2, predictor-sharing control | Bf=1500; Bq=7744/8000/43000; E=2000 disjoint; 20 draws |
| `equal_coverage` | Mean-size comparison at matched realised coverage | Bf=Bq=1500; E=2000 disjoint; 20 draws; target labels retarget NLL only for this diagnostic |
| `pool` | ImageNet feature-pool sensitivity | Bf=1500 or all43000; same Bq=43000/E=2000 as grid |
| `source_repeats` | Three additional ImageNet source selections | Bf=Bq=1500; E=2000 disjoint; 20 draws; exclude union of all four source selections |
| `clean` | ImageNet clean and ImageNetV2 scalar comparisons | Main 50×1500 protocol; descriptive |

A/U/O denote absolute coverage error, undercoverage and overcoverage (percentage
points); S is mean set size (classes). Metrics are computed per draw and averaged
over draws, severities and equally weighted types. The 95% intervals use 2,000
paired corruption-type bootstrap resamples, preserving severities and draws.
They condition on the classifier/source fit, without multiplicity correction or
classifier-seed replication. Clean/natural and CIFAR-grid results are descriptive.

## Data and run options

For flat validation JPEGs, pass
`--ground-truth /path/to/ILSVRC2012_validation_ground_truth.txt` and
`--meta-mat /path/to/meta.mat` from the official devkit. The loader converts
devkit labels to torchvision class indices.

`--workers N` parallelises CPU source fitting and evaluation (default **1**).
Full-pool ImageNet evaluation requires substantial RAM per worker.

Completed caches and evaluations are reused with integrity checks. Changed
inputs/code require a new work directory. Interrupted classifier training is
not resumed; retrain in a fresh directory.
Run a single stage with `--stage train`, `prepare`, `fit`, or `evaluate`:

```bash
./run.sh run --system c10_resnet18 --work runs/c10 --stage evaluate
```

Use `--experiments` to select entries from the table above; `pool`,
`source_repeats` and `clean` are ImageNet-only. Choose before preparation because
the selection determines required data and source predictions. The default
ImageNet run includes `clean` and requires `--v2` for ImageNetV2.

Outputs are `metrics.csv` (per draw), `macro.csv` (averages),
`paired_intervals.csv` and, when selected, `equal_coverage.csv`.
