# Using real sonar data

Everything downstream of the simulator (dataset class, training, evaluation,
ONNX export, streaming) works on any **COCO-format** directory:

```
my_dataset/
  train/annotations.json   images/*.png
  val/annotations.json     images/*.png
  test/annotations.json    images/*.png
```

Each image record needs `id`, `file_name`, `width`, `height` and ideally
`pixel_m` (ground pixel size, used for the FA/km² metric; defaults to 0.03).
Optional fields used by the breakdown metrics: `range_m`, `grazing_deg`,
`aspect_deg`, `burial`, `is_milco`, `visible` on each annotation and
`mask_file` on each image (3-class PNG: 0 seabed, 1 highlight, 2 shadow; if
absent the segmentation loss is simply skipped).

## Converting YOLO-format datasets

Several public side-scan / SAS datasets ship YOLO labels. Convert them with:

```python
from seabed_sentinel.data.coco import yolo_dir_to_coco
yolo_dir_to_coco("sctd/images/train", "sctd/labels/train", class_names=CLASSES,
                 out_json="data/sctd/train/annotations.json",
                 class_map={0: 0, 1: 4, 2: 5},   # source id -> Seabed Sentinel class id
                 pixel_m=0.05)
```

## Public datasets worth trying

* **SCTD 1.0 / 2.0 / 3.0** – Sonar Common Target Detection (side-scan and SAS
  images of wrecks, aircraft, humans and other targets).
  https://github.com/MingqiangNing/SCTD · https://github.com/zszhang724/SCTD2 · https://github.com/automlresearch/SCTD-3.0
* **Side-scan sonar imaging data of underwater vehicles for mine detection**
  (Gavia AUV, 1 170 images with MILCO / NOMBO boxes), Data in Brief 2024.
* **Seabed Objects KLSG** (side-scan: wrecks, aircraft, mines, seafloor).
* The 2025 survey *Sonar Image Datasets: A Comprehensive Survey* (arXiv
  2510.03353) lists further SSS / FLS / SAS / MBES sets.

## Recommended sim-to-real recipe

1. Pre-train the encoder with `sentinel pretrain` on **all** available
   unlabelled real tiles (plus synthetic tiles).
2. Train the detector on the synthetic set with the real tiles mixed in
   (e.g. 1:1 sampling) – or fine-tune the synthetic checkpoint on the real
   training split with a 5–10× lower learning rate.
3. Evaluate on *real* held-out data only; report PD vs FA/km² and the
   MILCO / NOMBO confusion, not just mAP.
4. Re-estimate the aspect-binned confusion model from real results before
   using the re-acquisition planner operationally.

The class taxonomy (`sim/scene.py: CLASSES`) can be changed; the detector,
metrics and planner all take the class list from the dataset / checkpoint.
