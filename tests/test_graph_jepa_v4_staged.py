from types import SimpleNamespace
import unittest

from graph_jepa_v4.config import Config
from graph_jepa_v4.finetune import _apply_finetune_args
from graph_jepa_v4.training import FINAL_CHECKPOINT_NAME, PRETRAIN_CHECKPOINT_NAME


class GraphJEPAv4StagedTrainingTests(unittest.TestCase):
    def test_pretrain_and_finetune_checkpoints_are_separate(self):
        self.assertEqual(PRETRAIN_CHECKPOINT_NAME, "graph_jepa_v4_pretrain.pt")
        self.assertEqual(FINAL_CHECKPOINT_NAME, "graph_jepa_v4.pt")

    def test_finetune_args_preserve_pretrain_epoch_count(self):
        cfg = Config()
        cfg.train.pretrain_epochs = 5
        args = SimpleNamespace(
            epochs=7,
            lr=1e-3,
            batch_size=4,
            num_workers=0,
            revision_weight=0.8,
            revision_mask_ratio=0.3,
            revision_neg_per_pos=2,
            synthetic_graphs=11,
            synthetic_min_nodes=4,
            synthetic_max_nodes=9,
            context_patches=None,
            target_patches=None,
        )

        _apply_finetune_args(args, cfg)

        self.assertEqual(cfg.train.pretrain_epochs, 5)
        self.assertEqual(cfg.train.finetune_epochs, 7)
        self.assertEqual(cfg.train.epochs, 12)
        self.assertEqual(cfg.train.revision_weight, 0.8)

    def test_legacy_joint_config_loads_as_zero_pretrain(self):
        cfg = Config.from_dict({
            "train": {
                "epochs": 9,
                "edge_head_weight": 0.7,
            },
        })

        self.assertEqual(cfg.train.pretrain_epochs, 0)
        self.assertEqual(cfg.train.finetune_epochs, 9)
        self.assertEqual(cfg.train.epochs, 9)


if __name__ == "__main__":
    unittest.main()
