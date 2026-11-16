These two small NPZ files contain the original classifier training/development
indices and source design/calibration/feature positions within development.
They contain no images, predictions, or labels. Do not sort the source positions:
CIFAR-10's batch-keyed synthetic noise depends on their recorded order.

CIFAR-10: 40,000 training / 5,000 development images (split seed 42).
CIFAR-100: 45,000 / 5,000 (split seed 20260908; role seed 20260909).
Both source partitions contain 1,000 design, 2,500 calibration, and 1,500 feature images.
