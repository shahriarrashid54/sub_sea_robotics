import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(1)


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """Render a tiny synthetic dataset once per test session."""
    from seabed_sentinel.sim.dataset_gen import generate_dataset
    from seabed_sentinel.sim.scene import SceneConfig
    root = str(tmp_path_factory.mktemp("sas_tiny"))
    cfg = SceneConfig(n_along=160, n_range=160, p_empty=0.0, range_start_range=(20.0, 60.0))
    generate_dataset(root, n_train=10, n_val=4, n_test=4, cfg=cfg, workers=1, seed=123)
    return root
