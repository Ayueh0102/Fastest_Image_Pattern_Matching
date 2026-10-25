#!/usr/bin/env python3
"""Index research evidence without moving it or extracting any archives.

The classification is a navigation aid, not an experiment pass/fail verdict.
Only explicitly named Boolean gates in standalone JSON are reported as false;
archive members are indexed by path and are not recursively evaluated.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile


OUTPUTS = {"log-catalog.json", "log-archive-members.json.gz"}
STAGES = {
    "real-image-experiments": "2", "performance-accuracy": "3",
    "gpu-refinement": "4", "pose-routing": "5", "metal-cost-profile": "5",
    "resident-refinement": "6", "ncc-rescue": "7", "pipeline-stages": "8",
    "shared-fine": "9", "coarse-next": "10", "many-objects": "11",
    "coarse-overlap": "12", "coarse-tiling": "13", "gui-integration": "14",
    "stability-30min": "14", "fixed-scale": "15", "sparse-orientation": "16",
    "sparse-compact": "17", "sparse-backends": "18", "sparse-metal": "19",
    "sparse-cpu-lut": "20", "sparse-metal-merge": "21", "pose-fused": "22",
    "pose-simd": "23", "pose-shared": "24", "pose-u8": "25",
    "adaptive-ncc": "26", "fine-reuse": "27", "fine-strip": "28",
    "fine-angle-reuse": "29", "fine-fused": "30", "gray-loader": "31",
    "run-diagnostics": "32", "resident-multiscale": "33", "stage34-closure": "34",
    "stage35-real-rgb": "35", "stage36-coarse-resident": "36", "stage37-full-gpu": "37", "stage38-gpu-peaks": "38", "stage39-fine-tiles": "39",
    "stage40-ncc-closure": "40", "stage41-new-frame-cost": "41", "stage42-parallel-peaks": "42",
    "stage43-candidate-dedup": "43",
}
ROLE_DESCRIPTIONS = {
    "protocol": "Settings, workload definition, budgets and acceptance gates",
    "formal": "Formal workers, paired measurements and formal summaries",
    "pilot": "Development-only exploration; not a formal performance claim",
    "calibration": "Threshold fitting using normal calibration images",
    "raw": "Raw worker output, trace or process log",
    "summary": "Aggregated results or analysis",
    "failure_review": "Named failed, clipped, interrupted or superseded attempt; inspect context",
    "false_gate": "Standalone JSON contains at least one explicitly named false Boolean gate",
    "validation": "Tests, sanitizer, numerical checks or archive verification",
    "source": "Source code or a source snapshot/archive",
    "provenance": "Hashes, manifests, environment or build provenance",
    "input": "Input image, fixture or ground truth",
    "visual": "Figure or image; may be either input or output",
    "report": "Human-readable report or README",
    "binary": "Archived native executable or library",
    "archive": "Archive or compressed evidence",
    "metadata": "Filesystem sidecar metadata",
    "other": "No recognized role; inspect manually",
}


def roles_for(name):
    path = PurePosixPath(name)
    low = name.lower()
    base = path.name.lower()
    roles = set()
    if base.startswith("._") or base == ".ds_store":
        return ["metadata"]
    if "protocol" in base:
        roles.add("protocol")
    if "pilot" in low:
        roles.add("pilot")
    if "calibration" in low:
        roles.add("calibration")
    if re.search(r"(^|/)(formal[^/]*|workers?|jobs)(/|[-_.])", low):
        roles.add("formal")
    if any(word in low for word in ("raw-results", "stderr", "stdout", "trace")) or base.endswith((".log", ".log.gz")):
        roles.add("raw")
    if any(word in base for word in ("summary", "results", "report", "analysis", "diagnosis", "recheck")):
        roles.add("summary")
    if any(word in low for word in ("fail", "clipped", "interrupted", "attempt1", "attempt2", "before-fix", "pre-fix")):
        roles.add("failure_review")
    if any(word in low for word in ("validation", "sanitizer", "ctest", "test-results", "release-tests", "archive-check", "replay")):
        roles.add("validation")
    if base.endswith((".cpp", ".hpp", ".h", ".c", ".mm", ".metal", ".py", ".cmake", ".patch")) or base == "cmakelists.txt" or any(word in low for word in ("/source/", "sources/", "baseline-source", "source-and-fixtures", "source-snapshot")):
        roles.add("source")
    if any(word in base for word in ("sha256", "manifest", "provenance", "environment")) or base == "archive.json":
        roles.add("provenance")
    if any(word in low for word in ("/inputs/", "/fixtures/", "/truth/", "ground-truth")) or base.endswith((".blend", ".npz")):
        roles.add("input")
    if base.endswith((".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".svg")):
        roles.add("visual")
    if base.endswith(".md") or base.startswith("readme"):
        roles.add("report")
    if "/binaries/" in f"/{low}" or base.endswith((".dylib", ".so", ".dll", ".exe")):
        roles.add("binary")
    if base.endswith((".gz", ".tar", ".zip")):
        roles.add("archive")
    return sorted(roles or {"other"})


def false_gates(value, pointer=""):
    """Return JSON pointers, never interpret ordinary options such as enabled=false."""
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            here = f"{pointer}/{escaped}"
            named_gate = key in {"pass", "passed", "success", "succeeded", "correctness_passed"} or key.endswith(("_pass", "_passed", "_succeeded"))
            if child is False and named_gate:
                found.append(here)
            found.extend(false_gates(child, here))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(false_gates(child, f"{pointer}/{index}"))
    return found


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_catalog(docs):
    artifacts, members = [], []
    role_counts, member_role_counts = Counter(), Counter()
    for path in sorted(docs.rglob("*")):
        if not path.is_file() or path.name in OUTPUTS or path.name.startswith("."):
            continue
        relative = path.relative_to(docs).as_posix()
        group = relative.split("/")[0] if "/" in relative else "root"
        entry = {"path": relative, "group": group, "stage": STAGES.get(group),
                 "bytes": path.stat().st_size, "sha256": sha256(path), "roles": roles_for(relative)}
        if path.suffix == ".json":
            try:
                gates = false_gates(json.loads(path.read_text(encoding="utf-8")))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid evidence JSON: {relative}") from error
            if gates:
                entry["explicit_false_gates"] = gates
                entry["roles"] = sorted(set(entry["roles"]) | {"false_gate"})
        if relative.endswith(".tar.gz"):
            counts = Counter()
            with tarfile.open(path, "r:gz") as archive:
                for item in archive:
                    if not item.isfile():
                        continue
                    roles = roles_for(item.name)
                    # Index headers only. Do not extract paths or open nested archives.
                    members.append({"archive": relative, "path": item.name,
                                    "bytes": item.size, "roles": roles})
                    counts.update(roles)
                    member_role_counts.update(roles)
            entry["archive_file_count"] = sum(1 for m in members if m["archive"] == relative)
            entry["archive_role_counts"] = dict(sorted(counts.items()))
        artifacts.append(entry)
        role_counts.update(entry["roles"])
    return {
        "schema_version": 1,
        "scope": "cpp-core/docs, excluding generated catalog files and dotfiles",
        "path_base": "cpp-core/docs",
        "classification": "Multiple navigation roles inferred from names; not a pass/fail verdict",
        "false_gate_scope": "Standalone JSON only; archive members and compressed JSON are not evaluated",
        "archive_scope": "Regular files in tar.gz headers, without extraction or nested archive traversal",
        "archive_member_index": "log-archive-members.json.gz",
        "role_descriptions": ROLE_DESCRIPTIONS,
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(a["bytes"] for a in artifacts),
        "artifact_role_counts": dict(sorted(role_counts.items())),
        "archive_member_count": len(members),
        "archive_member_role_counts": dict(sorted(member_role_counts.items())),
        "artifacts": artifacts,
    }, {"schema_version": 1, "path_base": "cpp-core/docs", "members": members}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=Path(__file__).resolve().parents[1] / "docs")
    parser.add_argument("--check", action="store_true", help="Verify indexes without changing them")
    args = parser.parse_args()
    catalog, members = create_catalog(args.docs)
    member_bytes = gzip.compress(json.dumps(members, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), mtime=0)
    catalog["archive_member_index_sha256"] = hashlib.sha256(member_bytes).hexdigest()
    outputs = {"log-catalog.json": (json.dumps(catalog, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
               "log-archive-members.json.gz": member_bytes}
    stale = []
    for name, content in outputs.items():
        target = args.docs / name
        if args.check:
            if not target.exists() or target.read_bytes() != content:
                stale.append(name)
        else:
            target.write_bytes(content)
    if stale:
        parser.exit(1, "Catalog is stale: " + ", ".join(stale) + "\n")
    print(json.dumps({"mode": "check" if args.check else "write", "artifacts": catalog["artifact_count"],
                      "archive_members": catalog["archive_member_count"], "unchanged": not stale}))


if __name__ == "__main__":
    main()
