"""Diagnostic phase/sampled macOS memory for the native template matcher.

Produces instrumented copies of the existing runner and Session implementation,
links the supplied build's remaining libraries, and keeps every source/diff/hash.
No core or normal benchmark source/binary is changed. Measurements are NOT speed
benchmark data: sampling and optional end-of-run malloc pressure relief perturb
execution. Run serially and separately from performance measurements.
"""
import argparse
import datetime
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Instrumentation anchor changed or is ambiguous: " + old[:100])
    return text.replace(old, new, 1)


def instrument_runner(text):
    # Keep frozen stage29 runners usable alongside the explicit gray loader.
    # For the new runner, sample only after its own decode timing and memory
    # snapshot, so instrumentation cannot inflate either native measurement.
    load_anchors = (
        ('tmp=read(j["template"].toString());Settings settings;',
         'tmp=read(j["template"].toString()); fpm_memory::stage("images_decoded"); Settings settings;'),
        ('const double decodeMs=ms(loadStart);const auto afterLoadMemory=memory();',
         'const double decodeMs=ms(loadStart);const auto afterLoadMemory=memory(); fpm_memory::stage("images_decoded");'),
    )
    recognized = [(old, new) for old, new in load_anchors if old in text]
    if len(recognized) != 1:
        raise ValueError("Unsupported or ambiguous runner image-load layout")
    text = replace_once(text, *recognized[0])
    # Select an entire first/repeat layout, not the two anchors independently:
    # a partially migrated or duplicated runner must never be instrumented.
    # Resource snapshots finish before our phase sample; beginCall runs before
    # their probe starts. Pipeline instrumentation still perturbs runtime, so
    # resource records from this diagnostic are not speed benchmark evidence.
    release = ('\n        fpm_memory::stage("before_session_release"); fpm_memory::pressureRelief("session_alive_relief");\n'
               '        session.reset(); fpm_memory::stage("after_session_release"); fpm_memory::pressureRelief("session_released_relief");')
    call_layouts = {
        "legacy": (
            ('Result first=session->run(o);QJsonObject firstRecord=record(first,true);',
             'fpm_memory::beginCall("first"); Result first=session->run(o); fpm_memory::stage("after_run"); QJsonObject firstRecord=record(first,true); fpm_memory::stage("after_serialization");'),
            ('for(int i=0;i<repeats;++i) {auto r=session->run(o);exact&=equal(r,first);runs.append(record(r,true));}',
             'for(int i=0;i<repeats;++i) {fpm_memory::beginCall("repeat"); auto r=session->run(o); fpm_memory::stage("after_run"); exact&=equal(r,first);runs.append(record(r,true)); fpm_memory::stage("after_serialization");}' + release),
        ),
        "run_resource_probe": (
            ('fpm::diagnostics::RunResourceProbe firstProbe(resourceDiagnostics);\n'
             '        Result first=session->run(o);const auto firstResources=firstProbe.finish();QJsonObject firstRecord=record(first,true);',
             'fpm_memory::beginCall("first"); fpm::diagnostics::RunResourceProbe firstProbe(resourceDiagnostics);\n'
             '        Result first=session->run(o);const auto firstResources=firstProbe.finish(); fpm_memory::stage("after_run"); QJsonObject firstRecord=record(first,true); fpm_memory::stage("after_serialization");'),
            ('for(int i=0;i<repeats;++i) {\n'
             '            fpm::diagnostics::RunResourceProbe probe(resourceDiagnostics);\n'
             '            auto r=session->run(o);const auto runResources=probe.finish();exact&=equal(r,first);\n'
             '            auto row=record(r,true);if(resourceDiagnostics) row["resources"]=resources(runResources);runs.append(row);\n'
             '        }',
             'for(int i=0;i<repeats;++i) {\n'
             '            fpm_memory::beginCall("repeat"); fpm::diagnostics::RunResourceProbe probe(resourceDiagnostics);\n'
             '            auto r=session->run(o);const auto runResources=probe.finish(); fpm_memory::stage("after_run"); exact&=equal(r,first);\n'
             '            auto row=record(r,true);if(resourceDiagnostics) row["resources"]=resources(runResources);runs.append(row); fpm_memory::stage("after_serialization");\n'
             '        }' + release),
        ),
    }
    present = [name for name, anchors in call_layouts.items()
               if any(old in text for old, _ in anchors)]
    if (len(present) != 1 or
            any(text.count(old) != 1 for old, _ in call_layouts[present[0]])):
        raise ValueError("Unsupported or ambiguous runner first/repeat layout")
    pairs = [
        ('#include "research_pipeline.hpp"', '#include "research_pipeline.hpp"\n#include "fine_memory_probe.hpp"'),
        ('QCoreApplication app(argc,argv);', 'QCoreApplication app(argc,argv); fpm_memory::Probe memoryProbe;'),
        ('const double construct=ms(start);', 'const double construct=ms(start); fpm_memory::stage("session_constructed");'),
        ('while(ms(start)<warm*1000) {auto r=session->run(o);exact&=equal(r,first);warmups.append(r.totalMs);}',
         'while(ms(start)<warm*1000) {fpm_memory::beginCall("warmup"); auto r=session->run(o); fpm_memory::stage("after_run"); exact&=equal(r,first);warmups.append(r.totalMs); fpm_memory::stage("after_serialization");}'),
    ]
    pairs.extend(call_layouts[present[0]])
    for old, new in pairs:
        text = replace_once(text, old, new)
    return text


def copy_runner_headers(source_root, source, runner_text):
    """Copy runner-local dependencies from the selected, possibly frozen tree."""
    originals = {}
    for name in ("qt_gray_loader.hpp", "run_resource_probe.hpp"):
        if '#include "' + name + '"' in runner_text:
            path = source_root / "cpp-core/tools" / name
            (source / name).write_bytes(path.read_bytes())
            originals[str(path)] = sha(path)
    return originals


def instrument_pipeline(text):
    pairs = [
        ('#include "research_pipeline.hpp"', '#include "research_pipeline.hpp"\n#include "fine_memory_probe.hpp"'),
        ('const auto prepare=Clock::now();', 'fpm_memory::stage("before_frame_prepare"); const auto prepare=Clock::now();'),
        ('r.prepareMs=ms(prepare);', 'r.prepareMs=ms(prepare); fpm_memory::stage("after_frame_prepare");'),
        ('for(size_t i=0;i<models.size();++i) {', 'for(size_t i=0;i<models.size();++i) { fpm_memory::stage("before_model_match",int(i));'),
        ('check(fpm_frame_coarse_overlap_metrics(s.frame.get(),&overlapMetrics,sizeof(overlapMetrics),error,sizeof(error)),error);',
         'fpm_memory::stage("after_model_match",int(i));\n            check(fpm_frame_coarse_overlap_metrics(s.frame.get(),&overlapMetrics,sizeof(overlapMetrics),error,sizeof(error)),error);'),
        ('const auto t=Clock::now();r.poseCreateMs=s.ensurePose(o.match.invertSource,r.poseConversionTemporaryBytes);',
         'fpm_memory::stage("before_pose_create"); const auto t=Clock::now();r.poseCreateMs=s.ensurePose(o.match.invertSource,r.poseConversionTemporaryBytes); fpm_memory::stage("after_pose_create");'),
        ('if(fpm_pose_last_resident_metrics(s.pose.get(),&r.resident))',
         'fpm_memory::stage("after_pose_refine"); if(fpm_pose_last_resident_metrics(s.pose.get(),&r.resident))'),
        ('r.totalMs=ms(start);return r;', 'r.totalMs=ms(start); fpm_memory::stage("search_finished"); return r;'),
        ('if(!pose) throw std::runtime_error(error);', 'fpm_memory::stage("pose_created_before_temporary_release"); if(!pose) throw std::runtime_error(error);'),
    ]
    for old, new in pairs:
        text = replace_once(text, old, new)
    return text


def target_compile_flags(build, target, source_root=ROOT):
    path = build / "experiments/CMakeFiles" / (target + ".dir") / "flags.make"
    flags = {}
    for line in path.read_text().splitlines():
        if " = " in line:
            name, value = line.split(" = ", 1)
            flags[name] = shlex.split(value)
    result = sum((flags[k] for k in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS")), [])
    # Different targets can have different FP contraction, macros and include
    # paths. Never compile the Session replacement using the runner's flags.
    return [v.replace(str(ROOT / "cpp-core"), str(source_root / "cpp-core")) for v in result], path


def prepare(build, dest, source_root=ROOT):
    source = dest / "source"
    source.mkdir()
    transformations = ((source_root / "cpp-core/tools/native_pipeline_runner.cpp", instrument_runner),
                       (source_root / "cpp-core/experiments/research_pipeline.cpp", instrument_pipeline))
    originals = {}
    for path, transform in transformations:
        original = path.read_text()
        changed = transform(original)
        (source / (path.stem + ".original.cpp")).write_text(original)
        (source / path.name).write_text(changed)
        (source / (path.name + ".diff")).write_text("".join(difflib.unified_diff(
            original.splitlines(True), changed.splitlines(True), fromfile=str(path), tofile=path.name)))
        originals[str(path)] = sha(path)
        if path.name == "native_pipeline_runner.cpp":
            originals.update(copy_runner_headers(source_root, source, original))
    for folder in ("include", "src", "experiments"):
        for path in (source_root / "cpp-core" / folder).rglob("*"):
            if path.is_file() and path.suffix in (".h", ".hpp"):
                originals[str(path)] = sha(path)
    header = ROOT / "cpp-core/tools/fine_memory_probe.hpp"
    (source / header.name).write_bytes(header.read_bytes())
    originals[str(header)] = sha(header)
    runner_flags, flags_path = target_compile_flags(build, "fpm_native_pipeline_runner", source_root)
    pipeline_flags, pipeline_flags_path = target_compile_flags(build, "fpm_research_pipeline", source_root)
    link_path = build / "experiments/CMakeFiles/fpm_native_pipeline_runner.dir/link.txt"
    link = shlex.split(link_path.read_text())
    objects = [i for i, v in enumerate(link) if v.endswith("native_pipeline_runner.cpp.o")]
    if len(objects) != 1 or link.count("-o") != 1:
        raise ValueError("Unsupported native runner link command")
    binary = dest / "fpm_memory_runner"
    runner_obj, pipeline_obj = dest / "runner.o", dest / "pipeline.o"
    link[objects[0]:objects[0] + 1] = [str(runner_obj), str(pipeline_obj)]
    link[link.index("-o") + 1] = str(binary)
    commands = [[link[0], *flags, "-c", str(source / filename), "-o", str(obj)]
                for filename, obj, flags in (("native_pipeline_runner.cpp", runner_obj, runner_flags),
                                             ("research_pipeline.cpp", pipeline_obj, pipeline_flags))]
    commands.append(link)
    # Retain the exact configuration and linked project library identities.
    (source / "flags.make").write_bytes(flags_path.read_bytes())
    (source / "pipeline-flags.make").write_bytes(pipeline_flags_path.read_bytes())
    (source / "link.txt").write_bytes(link_path.read_bytes())
    libraries = [Path(t) if Path(t).is_absolute() else build / "experiments" / t
                 for t in link if t.endswith((".a", ".dylib"))]
    metadata = dict(original_sha256=originals,
                    linked_library_sha256={str(p.resolve()): sha(p) for p in libraries},
                    commands=commands, build_cwd=str(build / "experiments"))
    save(dest / "build.json", metadata)
    env = dict(os.environ, DEVELOPER_DIR=os.environ.get("DEVELOPER_DIR", "/Library/Developer/CommandLineTools"))
    with (dest / "build.log").open("w") as log:
        for command in commands:
            subprocess.run(command, cwd=build / "experiments", env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata["binary_sha256"] = sha(binary)
    metadata["generated_source_sha256"] = {p.name: sha(p) for p in sorted(source.iterdir())}
    save(dest / "build.json", metadata)
    return binary, metadata


def summarize(trace, result, reference=None):
    rows = [json.loads(line) for line in Path(trace).read_text().splitlines()]
    if not rows or rows[0]["phase"] != "probe_started" or rows[-1]["phase"] != "probe_finished":
        raise ValueError("Incomplete memory trace")
    if any(not isinstance(r["elapsed_ms"], (int, float)) or not math.isfinite(r["elapsed_ms"]) for r in rows):
        raise ValueError("Invalid sample clock")
    for a, b in zip(rows, rows[1:]):
        if b["elapsed_ms"] < a["elapsed_ms"] or b["pid"] != a["pid"]:
            raise ValueError("Invalid sample order/process")
    if any(r["task_info_status"] != 0 or r["resident_bytes"] < 0 or r["physical_footprint_bytes"] < 0 for r in rows):
        raise ValueError("Required macOS memory counters unavailable")
    raw = json.loads(Path(result).read_text())
    exact = raw["exact_repeat"]
    calls = [raw["first"], *raw["runs"]]
    if not calls or not isinstance(raw["exact_repeat"], bool):
        raise ValueError("Malformed native results")
    exact = exact and all(c[k] == calls[0][k] for c in calls for k in ("matches", "proposals"))
    before = [r for r in rows if r["event"] == "stage" and r["phase"] == "before_run"]
    after = [r for r in rows if r["event"] == "stage" and r["phase"] == "after_run"]
    expected_calls = len(calls) + len(raw["warmups_ms"])
    if ([r["call"] for r in before] != list(range(expected_calls)) or
            [r["call"] for r in after] != list(range(expected_calls))):
        raise ValueError("Missing/duplicated diagnostic call boundaries")
    if reference:
        expected = json.loads(Path(reference).read_text())["first"]
        exact = exact and all(c[k] == expected[k] for c in calls for k in ("matches", "proposals"))
    fields = [k for k in rows[0] if k.endswith("_bytes") and k != "relieved_bytes"]
    maxima = {k: max(r[k] for r in rows) for k in fields}
    result = dict(sample_count=len(rows), exact_outputs=bool(exact),
                  reference_checked=reference is not None, maxima=maxima,
                  phases=[r for r in rows if r["event"] != "sample"],
                  resident_max_sample=max(rows, key=lambda r: r["resident_bytes"]),
                  footprint_max_sample=max(rows, key=lambda r: r["physical_footprint_bytes"]),
                  note="Diagnostic only; phase observations perturb timing. RSS, compressed, footprint and malloc counters overlap; do not sum them. Pressure-relief effects apply only to this diagnostic process, not the original benchmark.")
    if not exact:
        raise ValueError("Diagnostic output differs from repeat/reference")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/fine-search")
    parser.add_argument("--source-root", type=Path, default=ROOT, help="Source tree matching --build; use archived source for a frozen binary")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--interval-ms", type=int, default=100, help="0 disables periodic samples; stage snapshots remain")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmup-seconds", type=float)
    parser.add_argument("--pressure-relief", action="store_true", help="Attempt malloc release after all calls, then after Session destruction")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        raise ValueError("This diagnostic requires macOS")
    if not 0 <= args.interval_ms <= 60000:
        raise ValueError("Sampling interval must be 0..60000 ms")
    job = json.loads(args.job.read_text())
    if "truth" in job or job.get("soak_seconds", 0):
        raise ValueError("Use a non-soak native job without ground truth")
    if args.repeats is not None:
        job["repeats"] = args.repeats
    if args.warmup_seconds is not None:
        job["warmup_seconds"] = args.warmup_seconds
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    save(dest / "job.json", job)
    protocol = dict(version=1, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    diagnostic_not_speed_benchmark=True, platform=platform.platform(),
                    original_job=str(args.job.resolve()), original_job_sha256=sha(args.job),
                    job_sha256=sha(dest / "job.json"), script_sha256=sha(__file__),
                    build=str(args.build.resolve()), source_root=str(args.source_root.resolve()), interval_ms=args.interval_ms,
                    pressure_relief=args.pressure_relief, reference=str(args.reference.resolve()) if args.reference else None,
                    reference_sha256=sha(args.reference) if args.reference else None,
                    input_sha256={k: sha(job[k]) for k in ("source", "template")})
    save(dest / "protocol.json", protocol)
    binary, metadata = prepare(args.build.resolve(), dest, args.source_root.resolve())
    if args.build_only:
        print(binary)
        return
    env = dict(os.environ, FPM_MEMORY_TRACE=str(dest / "trace.jsonl"),
               FPM_MEMORY_INTERVAL_MS=str(args.interval_ms), FPM_MEMORY_RELIEF=str(int(args.pressure_relief)))
    (dest / "vm-before.txt").write_text(subprocess.check_output(["/usr/bin/vm_stat"], text=True))
    with (dest / "result.json").open("w") as out, (dest / "stderr.txt").open("w") as err:
        process = subprocess.run([str(binary), str(dest / "job.json")], stdout=out, stderr=err, env=env)
    (dest / "vm-after.txt").write_text(subprocess.check_output(["/usr/bin/vm_stat"], text=True))
    if process.returncode:
        raise RuntimeError("Diagnostic worker failed; raw trace/stderr retained at " + str(dest))
    summary = summarize(dest / "trace.jsonl", dest / "result.json", args.reference)
    summary["unchanged"] = (all(sha(p) == h for group in ("original_sha256", "linked_library_sha256") for p, h in metadata[group].items())
        and all(sha(job[k]) == h for k, h in protocol["input_sha256"].items())
        and sha(binary) == metadata["binary_sha256"] and sha(__file__) == protocol["script_sha256"])
    summary["hashes"] = {p.name: sha(p) for p in dest.iterdir() if p.is_file() and p.name != "summary.json"}
    save(dest / "summary.json", summary)
    if not summary["unchanged"]:
        raise RuntimeError("Source or linked library changed during diagnostic; do not compare this run")
    print(json.dumps(dict(output=str(dest), exact=summary["exact_outputs"], maxima=summary["maxima"]), indent=2))


if __name__ == "__main__":
    main()
