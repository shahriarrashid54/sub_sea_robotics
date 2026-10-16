import numpy as np

from seabed_sentinel.autonomy.aspect_fusion import AspectConfusionModel, bayesian_fuse, entropy
from seabed_sentinel.autonomy.reacquire import Contact2D, plan_reacquisition
from seabed_sentinel.sim.scene import CLASSES


def test_bayesian_fusion_sharpens_with_consistent_looks():
    model = AspectConfusionModel.synthetic(CLASSES, base_acc=0.75)
    prior = np.ones(len(CLASSES)) / len(CLASSES)
    obs = [{"pred": 0, "aspect_deg": a} for a in (30, 90, 150)]
    post = bayesian_fuse(prior, obs, model)
    assert np.argmax(post) == 0
    assert entropy(post) < entropy(prior)
    assert abs(post.sum() - 1) < 1e-9


def test_information_gain_is_nonnegative_and_zero_for_certain_posterior():
    model = AspectConfusionModel.synthetic(CLASSES)
    certain = np.zeros(len(CLASSES)); certain[2] = 1.0
    assert abs(model.expected_information_gain(certain, 90)) < 1e-6
    uniform = np.ones(len(CLASSES)) / len(CLASSES)
    assert model.expected_information_gain(uniform, 90) > 0


def test_planner_respects_budget_and_prefers_uncertain_contacts():
    rng = np.random.default_rng(0)
    model = AspectConfusionModel.synthetic(CLASSES)
    certain = np.zeros(len(CLASSES)); certain[0] = 1.0
    contacts = [Contact2D(0, 0, 100, certain, first_aspect_deg=90, est_heading_deg=0),
                Contact2D(1, 50, 150, np.ones(len(CLASSES)) / len(CLASSES), first_aspect_deg=90, est_heading_deg=0),
                Contact2D(2, -80, 200, rng.dirichlet(np.ones(len(CLASSES))), first_aspect_deg=0, est_heading_deg=45)]
    plan = plan_reacquisition(contacts, model, budget_m=800.0, standoff_m=30, leg_len_m=40)
    assert plan.total_length_m <= 800.0 * 1.5  # tour length includes turn penalties, selection respects the straight-line budget
    assert 0 in plan.skipped          # the already-certain contact is not revisited
    assert any(leg.contact_id == 1 for leg in plan.legs)
    assert plan.path.shape[1] == 2 and len(plan.path) == 1 + 2 * len(plan.legs)
