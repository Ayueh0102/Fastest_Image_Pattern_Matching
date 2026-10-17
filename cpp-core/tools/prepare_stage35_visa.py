#!/usr/bin/env python3
"""Prepare a small, pinned VisA RGB pilot without downloading the 1.9 GB archive.

Requires Pillow. Original images are obtained from an immutable unmodified mirror;
all 100 capsule masks are checked byte-for-byte against a small range of the
official archive. This is data preparation, not an algorithm benchmark.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

from PIL import Image


MIRROR = "imaadd05/visa-anomaly-detection"
REVISION = "912ae1aa35ac0791e64524a42342850af1605363"
AUTHOR_REVISION = "2a692ab575001cbde74d402d897a7286086c6199"
ARCHIVE = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
ARCHIVE_BYTES = 1929840640
MASK_START, MASK_LENGTH = 301562368, 314880
ANNOTATION_START, ANNOTATION_LENGTH = 129239040, 36870
SELECTION = (
    ("pilot", "normal", "000"), ("pilot", "normal", "200"),
    ("pilot", "normal", "400"), ("pilot", "normal", "050"),
    ("pilot", "anomaly", "030"),
    ("pilot", "anomaly", "036"), ("pilot", "anomaly", "042"),
    ("heldout", "normal", "100"), ("heldout", "normal", "300"),
    ("heldout", "normal", "500"), ("heldout", "anomaly", "033"),
    ("heldout", "anomaly", "039"), ("heldout", "anomaly", "044"),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def acquire(root: Path, relative: str, url: str, *, offline: bool,
            expected_sha: str | None = None, expected_size: int | None = None,
            byte_range: tuple[int, int] | None = None) -> bytes:
    path = root / relative
    if path.exists():
        data = path.read_bytes()
    else:
        if offline:
            raise RuntimeError(f"Missing cached input in offline mode: {path}")
        headers = {"User-Agent": "FPM-Stage35-Research/1.0"}
        if byte_range:
            start, length = byte_range
            headers["Range"] = f"bytes={start}-{start + length - 1}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            if byte_range:
                start, length = byte_range
                content_range = f"bytes {start}-{start + length - 1}/{ARCHIVE_BYTES}"
                if response.status != 206 or response.headers.get("Content-Range") != content_range:
                    raise RuntimeError("Official archive did not honor the exact bounded range")
            # A fixed upper bound also prevents silently downloading the full archive.
            limit = expected_size or (8 * 1024 * 1024)
            data = response.read(limit + 1)
            if len(data) > limit:
                raise RuntimeError(f"Response exceeded expected limit for {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(path.suffix + ".tmp").write_bytes(data)
        path.with_suffix(path.suffix + ".tmp").replace(path)
    if expected_size is not None and len(data) != expected_size:
        raise RuntimeError(f"Size mismatch: {relative}")
    if expected_sha and sha256(data) != expected_sha:
        raise RuntimeError(f"SHA256 mismatch: {relative}")
    return data


def prepare(root: Path, offline: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    prior_manifest = root / "manifest.json"
    if prior_manifest.exists():
        prior_bytes = prior_manifest.read_bytes()
        if len(json.loads(prior_bytes).get("samples", [])) == 12:
            backup = root / "manifest-initial-12.json"
            if backup.exists() and backup.read_bytes() != prior_bytes:
                raise RuntimeError("Existing 12-image historical manifest differs; refusing to overwrite it")
            backup.write_bytes(prior_bytes)
    mirror_base = f"https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}/"
    tree_url = f"https://huggingface.co/api/datasets/{MIRROR}/tree/{REVISION}/capsules?recursive=true&limit=1000"
    tree_data = acquire(root, "provenance/capsules_tree.json", tree_url, offline=offline)
    tree = {entry["path"]: entry for entry in json.loads(tree_data) if entry["type"] == "file"}
    if len(tree) != 803:
        raise RuntimeError("Unexpected capsule inventory; do not use an incomplete paginated listing")

    records = []
    def mirror_file(path: str) -> bytes:
        item = tree[path]
        data = acquire(root, path, mirror_base + path, offline=offline,
                       expected_sha=item.get("lfs", {}).get("oid"), expected_size=item["size"])
        records.append({"path": path, "url": mirror_base + path,
                        "bytes": len(data), "sha256": sha256(data)})
        return data

    annotation = mirror_file("capsules/image_anno.csv")
    original_annotation = acquire(root, "provenance/capsules_image_anno_original.csv", ARCHIVE,
                                  offline=offline, expected_size=ANNOTATION_LENGTH,
                                  byte_range=(ANNOTATION_START, ANNOTATION_LENGTH))
    if annotation != original_annotation:
        raise RuntimeError("Mirror image annotation differs from official archive")
    rows = {row["image"]: row for row in csv.DictReader(io.StringIO(annotation.decode()))}

    author_base = f"https://raw.githubusercontent.com/amazon-science/spot-diff/{AUTHOR_REVISION}/"
    labels_source = acquire(root, "provenance/author_id2class.py", author_base + "utils/id2class.py", offline=offline)
    label_assignment = next(node for node in ast.parse(labels_source).body
                            if isinstance(node, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "id2class_map" for t in node.targets))
    label_map = ast.literal_eval(label_assignment.value)["capsules"]
    if label_map["2"] != "Discolor":
        raise RuntimeError("Unexpected author-defined capsule class map")
    acquire(root, "provenance/author_README.md", author_base + "README.md", offline=offline)
    acquire(root, "provenance/LICENSE-DATASET", mirror_base + "LICENSE-DATASET", offline=offline)
    archive_masks = acquire(root, "provenance/capsules_masks_original.tar", ARCHIVE,
                            offline=offline, expected_size=MASK_LENGTH,
                            byte_range=(MASK_START, MASK_LENGTH))
    original_masks = {}
    # Read members explicitly, rather than extracting untrusted archive paths.
    with tarfile.open(fileobj=io.BytesIO(archive_masks), mode="r:") as archive:
        for member in archive:
            if member.isfile():
                if member.name not in tree or "/Masks/Anomaly/" not in member.name:
                    raise RuntimeError(f"Unexpected official archive member: {member.name}")
                original_masks[member.name] = archive.extractfile(member).read()
    if len(original_masks) != 100:
        raise RuntimeError("Expected all 100 original capsule anomaly masks")

    mask_inventory = []
    for path, data in sorted(original_masks.items()):
        item = tree[path]
        if len(data) != item["size"] or sha256(data) != item["lfs"]["oid"]:
            raise RuntimeError(f"Official mask differs from immutable mirror: {path}")
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != data:
            raise RuntimeError(f"Existing mask differs from official bytes: {path}")
        destination.write_bytes(data)
        with Image.open(io.BytesIO(data)) as image:
            values = {str(value): count for count, value in image.getcolors(maxcolors=256)}
        mask_inventory.append({"path": path, "bytes": len(data), "sha256": sha256(data),
                               "label_pixel_counts": values})
    inventory = {entry["path"]: entry for entry in mask_inventory}

    samples = []
    for split, status, ident in SELECTION:
        path = f"capsules/Data/Images/{status.title()}/{ident}.JPG"
        row = rows[path]
        if (status == "normal" and row["label"] != "normal") or (status == "anomaly" and "discolor" not in row["label"].split(",")):
            raise RuntimeError(f"Unexpected author image status: {path}")
        image_data = mirror_file(path)
        with Image.open(io.BytesIO(image_data)) as image:
            if image.mode != "RGB":
                raise RuntimeError(f"Expected original RGB input: {path}")
            width, height = image.size
        mask_path = row["mask"] or None
        counts = inventory[mask_path]["label_pixel_counts"] if mask_path else {"0": width * height}
        if mask_path:
            with Image.open(root / mask_path) as mask:
                if mask.size != (width, height):
                    raise RuntimeError(f"Image/mask size mismatch: {path}")
            # This selected range has label 2 as the majority non-background label.
            # Small lower IDs also occur around higher-ID defects elsewhere;
            # presence of a pixel equal to 2 alone is NOT a valid selection rule.
            if max((key for key in counts if key != "0"), key=lambda key: counts[key]) != "2":
                raise RuntimeError(f"Selected sample is not dominated by Discolor: {path}")
        evaluation_mask_path = f"derived/evaluation_discolor_masks/capsules_{status}_{ident}.png"
        destination = root / evaluation_mask_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mask_path:
            with Image.open(root / mask_path) as mask:
                evaluation_mask = mask.point(lambda value: 255 if value == 2 else 0)
        else:
            evaluation_mask = Image.new("L", (width, height), 0)
        evaluation_mask.save(destination)
        is_template_source = status == "normal" and ident == "000"
        samples.append({
            "sample_id": f"capsules_{status}_{ident}", "split": split,
            "role": "template_self_check" if is_template_source else ("calibration" if split == "pilot" else "heldout"),
            "threshold_fit_eligible": split == "pilot" and not is_template_source,
            "self_match": is_template_source,
            "image": path, "image_sha256": sha256(image_data),
            "width": width, "height": height, "mode": "RGB", "label": status,
            "author_defect_labels": row["label"].split(","),
            "mask": mask_path, "mask_sha256": inventory[mask_path]["sha256"] if mask_path else None,
            "label_pixel_counts": counts, "target_label": 2 if mask_path else None,
            "target_name": "Discolor" if mask_path else None,
            "evaluation_binary_mask": {
                "path": evaluation_mask_path, "sha256": sha256(destination.read_bytes()),
                "width": width, "height": height, "mode": "L", "foreground_value": 255,
                "foreground_pixels": counts.get("2", 0),
                "rule": "original multiclass mask == 2; normal images are all-zero",
                "use": "Evaluation only; do not use to generate or filter search candidates",
            },
            "object_boxes": None, "object_poses": None,
        })

    manifest = {
        "schema_version": 2, "dataset": "VisA", "subset": "capsules_stage35_pilot",
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, Onkar Dabeer; Amazon Science; VisA (ECCV 2022)",
        "paper": "https://arxiv.org/abs/2207.14315",
        "author_repository": "https://github.com/amazon-science/spot-diff",
        "author_revision": AUTHOR_REVISION, "official_archive": ARCHIVE,
        "mirror_repository": MIRROR, "mirror_revision": REVISION,
        "mirror_image_verification": "Pinned mirror LFS SHA256 and size; images not individually compared to official tar",
        "annotation_verification": "image_anno.csv and all 100 capsule masks byte-identical to official archive ranges",
        "official_archive_ranges": [
            {"start": ANNOTATION_START, "length": ANNOTATION_LENGTH, "sha256": sha256(original_annotation)},
            {"start": MASK_START, "length": MASK_LENGTH, "sha256": sha256(archive_masks)},
        ],
        "class_map": label_map,
        "selection": "Fixed 13 images before formal benchmark: template/self-check normal000 excluded from threshold fitting, 3 distinct normal images + 3 mixed bubble/discolor images for calibration, and 3 normal + 3 mixed bubble/discolor heldout images",
        "protocol_revision": "13 images; normal050 added before formal benchmark so normal000 does not calibrate thresholds",
        "split_caveat": "Local pilot/heldout split, not the standard VisA split; same collection protocol and possible correlated views; no physical-object identity split",
        "annotation_caveat": "Pixel-level anomaly masks only, no full-object boxes or poses. Selected images have author CSV labels bubble,discolor and masks dominated by ID 2; they are not pure-discolor images. Lower nonzero IDs coexist in masks; their presence must be preserved, not silently relabelled.",
        "evaluation_constraint": "Ground-truth masks may score detections or define explicitly oracle-conditioned analysis; never feed masks into search and describe the result as end-to-end detection.",
        "provenance_files": [
            {"path": path, "sha256": sha256((root / path).read_bytes()),
             "bytes": (root / path).stat().st_size, "url": url}
            for path, url in (
                ("provenance/capsules_tree.json", tree_url),
                ("provenance/author_id2class.py", author_base + "utils/id2class.py"),
                ("provenance/author_README.md", author_base + "README.md"),
                ("provenance/LICENSE-DATASET", mirror_base + "LICENSE-DATASET"),
            )
        ],
        "template_reference": {
            "sample_id": "capsules_normal_000", "image": "capsules/Data/Images/Normal/000.JPG",
            "bbox_xywh": [878, 25, 112, 245], "coordinate_convention": "zero-based half-open pixel rectangle",
            "selection": "Manual visual selection of a complete upright normal capsule; no anomaly mask used",
            "source_split": "pilot", "self_match_sample": "capsules_normal_000",
            "threshold_fit_eligible": False,
            "source_sha256": next(sample["image_sha256"] for sample in samples if sample["sample_id"] == "capsules_normal_000"),
            "crop_width": 112, "crop_height": 245,
            "foreground_mask": {
                "status": "Not generated by data preparation; experiment runner must record its own mask and hash",
                "allowed_source": "Only normal000 template crop RGB pixels; never query image anomaly ground truth",
            },
        },
        "raw_images_modified": False, "samples": samples, "downloaded_images_and_csv": records,
        "mask_inventory": mask_inventory,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def render_preview(root: Path, manifest: dict) -> None:
    """Render a review figure; never use its resized pixels as benchmark inputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np

    rows = (len(manifest["samples"]) + 2) // 3
    figure, axes = plt.subplots(rows, 3, figsize=(15, 3.5 * rows))
    for axis in axes.flat:
        axis.axis("off")
    for axis, sample in zip(axes.flat, manifest["samples"]):
        with Image.open(root / sample["image"]) as image:
            axis.imshow(image)
        axis.set_title(f"{sample['sample_id']} | {sample['role']}")
        axis.axis("off")
    figure.suptitle("VisA RGB pilot: original scenes, no prediction overlays\nAmazon Science / Zou et al. (ECCV 2022), CC BY 4.0")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(root / "contact_sheet.png", dpi=120)
    plt.close(figure)

    reference = manifest["template_reference"]
    figure, axis = plt.subplots(figsize=(12, 8))
    with Image.open(root / reference["image"]) as image:
        axis.imshow(image)
    x, y, width, height = reference["bbox_xywh"]
    axis.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="red", linewidth=2))
    axis.set_title(f"Pilot template reference: xywh={reference['bbox_xywh']}; source image is a self-match control")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(root / "template_reference.png", dpi=120)
    plt.close(figure)

    anomalies = [sample for sample in manifest["samples"] if sample["mask"]]
    figure, axes = plt.subplots(len(anomalies), 2, figsize=(12, 4 * len(anomalies)))
    for row, sample in zip(axes, anomalies):
        with Image.open(root / sample["image"]) as image:
            pixels = np.array(image)
        with Image.open(root / sample["mask"]) as image:
            mask = np.array(image)
        row[0].imshow(pixels)
        row[0].set_title(f"{sample['sample_id']}: source")
        # A GT-conditioned zoom is labelled explicitly and is review-only.
        yy, xx = np.where(mask > 0)
        margin = 70
        x0, x1 = max(0, int(xx.min()) - margin), min(pixels.shape[1], int(xx.max()) + margin + 1)
        y0, y1 = max(0, int(yy.min()) - margin), min(pixels.shape[0], int(yy.max()) + margin + 1)
        row[1].imshow(pixels[y0:y1, x0:x1])
        row[1].contour(mask[y0:y1, x0:x1] == 2, levels=[0.5], colors=["red"], linewidths=1)
        row[1].set_title(f"GT-conditioned review zoom: ID 2 = Discolor\n{sample['label_pixel_counts'].get('2', 0)} pixels; CSV also labels Bubble")
        for axis in row:
            axis.axis("off")
    figure.suptitle("Ground-truth review only: masks must not generate search candidates\nVisA / Amazon Science / Zou et al. (ECCV 2022), CC BY 4.0")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(root / "anomaly_review.png", dpi=110)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/stage35-real-rgb/data/visa"))
    parser.add_argument("--offline", action="store_true", help="Revalidate cached files without network access")
    parser.add_argument("--preview", action="store_true", help="Render review figures (requires matplotlib and numpy)")
    args = parser.parse_args()
    manifest = prepare(args.output, args.offline)
    if args.preview:
        render_preview(args.output, manifest)
    print(json.dumps({"manifest": str(args.output / "manifest.json"), "samples": len(manifest["samples"]),
                      "rgb_pixels_per_image": [manifest["samples"][0]["width"], manifest["samples"][0]["height"]],
                      "official_masks_verified": len(manifest["mask_inventory"])}))


if __name__ == "__main__":
    main()
