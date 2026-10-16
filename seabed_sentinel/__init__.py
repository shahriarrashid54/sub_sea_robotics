"""Seabed Sentinel: physics-grounded synthetic aperture sonar (SAS) simulation,
image formation, and AI-based automatic target recognition (ATR) for mine
countermeasures and seabed survey.

Sub-packages
------------
sim       : seabed/target scene generation, acoustic echo simulation, SAS image
            formation (back-projection), DPCA micronavigation, fast renderer.
data      : COCO-style dataset I/O, PyTorch datasets, sonar-aware augmentation.
models    : compact anchor-free detector (CenterNet-style) with shadow-aware
            auxiliary segmentation, MC-dropout uncertainty, MAE pre-training.
train     : training loop and MCM-standard evaluation (PD vs FA/km^2, mAP).
deploy    : ONNX export, real-time streaming inference, geo-referencing.
autonomy  : multi-aspect Bayesian fusion and information-driven re-acquisition
            planning for AUV re-look missions.
viz       : plotting helpers and Grad-CAM explanations.
"""

__version__ = "0.1.0"
