"""Dataset identity, source randomness and training smoke checks without downloads."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from cpr10 import datasets as data, train, transforms


class DatasetTests(unittest.TestCase):
    def test_packaged_splits_keep_classifier_and_source_roles_disjoint(self):
        for system, train_n in [("c10_resnet18", 40000), ("c100_vit_b_16", 45000)]:
            with self.subTest(system=system):
                splits = data.cifar_splits(system)
                self.assertEqual(len(splits["train"]), train_n)
                self.assertEqual(len(splits["dev"]), 5000)
                self.assertEqual(len(np.intersect1d(splits["train"], splits["dev"])), 0)
                for name in ("train", "dev"):
                    values = splits[name]
                    self.assertEqual(len(np.unique(values)), len(values))
                    self.assertTrue(((0 <= values) & (values < 50000)).all())
                roles = [
                    splits[f"{role}_positions"]
                    for role in ("design", "calibration", "feature")
                ]
                self.assertEqual([len(role) for role in roles], [1000, 2500, 1500])
                np.testing.assert_array_equal(
                    np.sort(np.concatenate(roles)), np.arange(5000)
                )
        # The original CIFAR-10 role order is part of the noise-generation protocol.
        np.testing.assert_array_equal(
            data.cifar_splits("c10_resnet18")["design_positions"][:4],
            [4140, 1710, 1211, 1058],
        )

    def test_flat_imagenet_uses_devkit_mapping_not_ground_truth_minus_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in (1, 2):
                Image.new("RGB", (4, 4)).save(root / f"ILSVRC2012_val_{index:08d}.JPEG")
            truth = root / "truth.txt"
            truth.write_text("17\n4\n")
            synsets = np.array(
                [[17, "n00000010", "", "", 0], [4, "n00000002", "", "", 0]],
                dtype=object,
            )
            with mock.patch("scipy.io.loadmat", return_value={"synsets": synsets}):
                records = data.index_imagenet_validation(
                    root, truth, root / "meta.mat", expected_classes=2
                )
            self.assertEqual([record.label for record in records], [1, 0])
            self.assertEqual(records[0].image_id, "ILSVRC2012_val_00000001")
            self.assertEqual(records[0].wnid, "n00000010")
            self.assertEqual(
                data.exclude_source_ids(records, [records[0].image_id]), [records[1]]
            )

    def test_imagenetv2_labels_follow_numeric_folder_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in range(12):
                folder = root / str(label)
                folder.mkdir()
                Image.new("RGB", (4, 4)).save(folder / "image.jpeg")
            records = data.index_imagenet_v2(root, expected_classes=12)
            self.assertEqual(len(records), 12)
            for record in records:
                self.assertEqual(record.label, int(record.path.parent.name))
                self.assertEqual(
                    record.image_id, f"imagenetv2/{record.label}/image.jpeg"
                )

    def test_imagenet_c_checks_base_identity_and_source_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for label, wnid in enumerate(["n00000002", "n00000010"]):
                folder = root / "fog" / "1" / wnid
                folder.mkdir(parents=True)
                path = folder / f"ILSVRC2012_val_{label + 1:08d}.JPEG"
                Image.new("RGB", (4, 4)).save(path)
                records.append(
                    data.ImageRecord(path, label, data.canonical_val_id(path), wnid)
                )
            retained = data.index_imagenet_c(
                root,
                "fog",
                1,
                records,
                exclude_ids=[records[0].image_id],
                expected_classes=2,
            )
            self.assertEqual(retained, [records[1]])
            wrong = [
                data.ImageRecord(
                    records[0].path, 1, records[0].image_id, records[0].wnid
                ),
                records[1],
            ]
            with self.assertRaisesRegex(ValueError, "label disagrees"):
                data.index_imagenet_c(root, "fog", 1, wrong, expected_classes=2)

    def test_imagenet_source_roles_are_fixed_and_balanced(self):
        records = [
            data.ImageRecord(Path(f"{i}.JPEG"), i // 10, f"ILSVRC2012_val_{i + 1:08d}")
            for i in range(40)
        ]
        first = data.split_source(records, seed=20260911, classes=4, per_class=10)
        second = data.split_source(records, seed=20260911, classes=4, per_class=10)
        for role in first:
            np.testing.assert_array_equal(first[role], second[role])
        self.assertEqual(
            [len(first[key]) for key in ("design", "feature", "calibration", "test")],
            [4, 6, 10, 20],
        )
        np.testing.assert_array_equal(
            np.sort(np.concatenate(list(first.values()))), np.arange(40)
        )
        source = np.concatenate(
            [first[role] for role in ("design", "feature", "calibration")]
        )
        np.testing.assert_array_equal(
            np.bincount([records[i].label for i in source]), [5] * 4
        )

    def test_synthetic_randomness_respects_declared_keys(self):
        images = torch.full((4, 3, 32, 32), 0.5)
        spec = transforms.make_specs()[0]
        self.assertEqual(len(transforms.make_specs()), 320)
        self.assertEqual(len({s["family"] for s in transforms.make_specs()}), 8)
        original = transforms.apply_spec(images, spec, 1000000)
        torch.testing.assert_close(
            original, transforms.apply_spec(images, spec, 1000000), atol=0, rtol=0
        )
        self.assertFalse(
            torch.equal(original, transforms.apply_spec(images, spec, 2000000))
        )
        # CIFAR noise is keyed by batch offset; arbitrary rebatching is not equivalent.
        rebatched = torch.cat(
            [
                transforms.apply_spec(images[:2], spec, 1000000),
                transforms.apply_spec(images[2:], spec, 1000002),
            ]
        )
        self.assertFalse(torch.equal(original, rebatched))
        # ImageNet uses image identities and is independent of batching.
        ids = [f"ILSVRC2012_val_{i:08d}" for i in range(1, 5)]
        whole = transforms.apply_synthetic(images, spec, ids, 1000000)
        separate = torch.stack(
            [
                transforms.apply_synthetic(x, spec, image_id, 1000000)
                for x, image_id in zip(images, ids)
            ]
        )
        torch.testing.assert_close(whole, separate, atol=0, rtol=0)

    def test_cache_rejects_changed_bytes_labels_and_id_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probabilities.npz"
            data._write_npz(
                path,
                p=np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32),
                y=np.array([1, 0]),
                ids=np.array(["a", "b"]),
            )
            self.assertTrue(
                data._cached(
                    path, ("p", "y", "ids"), expected_y=[1, 0], expected_ids=["a", "b"]
                )
            )
            with self.assertRaisesRegex(ValueError, "identities differ"):
                data._cached(path, ("p", "y", "ids"), expected_ids=["b", "a"])
            with self.assertRaisesRegex(ValueError, "labels differ"):
                data._cached(path, ("p", "y", "ids"), expected_y=[0, 1])
            with path.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                data._cached(path, ("p", "y", "ids"))
            data._write_npz(
                path, p=np.array([[0.2, 0.8]]), y=np.array([-1]), ids=np.array(["a"])
            )
            with self.assertRaisesRegex(ValueError, "Invalid labels"):
                data._cached(path, ("p", "y", "ids"))

    def test_resume_refuses_mixing_cache_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = {
                "system": "c10_resnet18",
                "checkpoint_sha256": "one",
                "source_specifications": [],
            }
            data._begin_cache(temporary, manifest)
            data._begin_cache(temporary, manifest)
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                data._begin_cache(temporary, dict(manifest, checkpoint_sha256="two"))


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 50000

    def __getitem__(self, index):
        return torch.full((3, 4, 4), (index % 5) / 5.0), index % 2


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3)
        self.heads = torch.nn.Sequential(torch.nn.Linear(3, 2))

    def forward(self, x):
        return self.heads(self.backbone(x.mean((2, 3))))


class TrainingSmokeTests(unittest.TestCase):
    def test_both_training_paths_save_load_and_refuse_overwrite(self):
        # Mock images/model/epoch iteration only: optimizers, schedulers, losses,
        # development selection and checkpoint serialization remain real.
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                for system in ("c10_resnet18", "c100_vit_b_16"):
                    with self.subTest(system=system):
                        output = Path(temporary) / system
                        with mock.patch.object(
                            train.datasets, "CIFAR10", _TinyDataset
                        ), mock.patch.object(
                            train.datasets, "CIFAR100", _TinyDataset
                        ), mock.patch.object(
                            train, "classifier", return_value=_TinyModel()
                        ), mock.patch.object(
                            train,
                            "cifar_splits",
                            return_value={
                                "train": np.arange(4),
                                "dev": np.arange(4, 8),
                            },
                        ), mock.patch.object(
                            train, "range", return_value=range(1, 3), create=True
                        ), redirect_stdout(
                            io.StringIO()
                        ):
                            checkpoint = train.train_classifier(
                                system, temporary, output, device="cpu", workers=0
                            )
                        saved = torch.load(
                            checkpoint, map_location="cpu", weights_only=True
                        )
                        model = _TinyModel()
                        model.load_state_dict(saved["model_state"])
                        self.assertTrue(
                            torch.isfinite(model(torch.ones(2, 3, 4, 4))).all()
                        )
                        history = json.loads((output / "history.json").read_text())
                        self.assertEqual(len(history), 2)
                        self.assertEqual(saved["system"], system)
                        complete = json.loads((output / "complete.json").read_text())
                        self.assertEqual(
                            complete["checkpoint_sha256"], data.sha256_file(checkpoint)
                        )
                        with self.assertRaises(FileExistsError):
                            train.train_classifier(
                                system, temporary, output, device="cpu", workers=0
                            )
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
