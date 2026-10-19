"""Memory diagnostics must fail closed on partial or misleading evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from profile_fine_memory import (copy_runner_headers, instrument_runner, replace_once,
                                 sha, summarize, target_compile_flags, ROOT)


def runner_fixture(load):
    return '\n'.join((
        '#include "research_pipeline.hpp"',
        'QCoreApplication app(argc,argv);',
        load,
        'const double construct=ms(start);',
        'Result first=session->run(o);QJsonObject firstRecord=record(first,true);',
        'while(ms(start)<warm*1000) {auto r=session->run(o);exact&=equal(r,first);warmups.append(r.totalMs);}',
        'for(int i=0;i<repeats;++i) {auto r=session->run(o);exact&=equal(r,first);runs.append(record(r,true));}',
    ))


LEGACY_LOAD = 'tmp=read(j["template"].toString());Settings settings;'
GRAY_LOAD = 'const double decodeMs=ms(loadStart);const auto afterLoadMemory=memory();'
LEGACY_FIRST = 'Result first=session->run(o);QJsonObject firstRecord=record(first,true);'
LEGACY_REPEAT = 'for(int i=0;i<repeats;++i) {auto r=session->run(o);exact&=equal(r,first);runs.append(record(r,true));}'
RESOURCE_FIRST = ('fpm::diagnostics::RunResourceProbe firstProbe(resourceDiagnostics);\n'
                  '        Result first=session->run(o);const auto firstResources=firstProbe.finish();QJsonObject firstRecord=record(first,true);')
RESOURCE_REPEAT = ('for(int i=0;i<repeats;++i) {\n'
                   '            fpm::diagnostics::RunResourceProbe probe(resourceDiagnostics);\n'
                   '            auto r=session->run(o);const auto runResources=probe.finish();exact&=equal(r,first);\n'
                   '            auto row=record(r,true);if(resourceDiagnostics) row["resources"]=resources(runResources);runs.append(row);\n'
                   '        }')


def resource_runner_fixture():
    return runner_fixture(GRAY_LOAD).replace(LEGACY_FIRST, RESOURCE_FIRST).replace(LEGACY_REPEAT, RESOURCE_REPEAT)


class FineMemoryProfileTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = {"exact_repeat": True, "first": {"matches": [{"score": .9}], "proposals": [1]},
                    "runs": [{"matches": [{"score": .9}], "proposals": [1]}], "warmups_ms": []}
        seed = dict(elapsed_ms=0, pid=1, event="stage", phase="probe_started", call=-1,
                    task_info_status=0, resident_bytes=100, physical_footprint_bytes=140,
                    resident_peak_bytes=160, rusage_peak_rss_bytes=155,
                    compressed_bytes=40, relieved_bytes=0)
        self.rows = [seed]
        for call in range(2):
            for phase in ("before_run", "after_run"):
                self.rows.append(dict(seed, elapsed_ms=len(self.rows), phase=phase, call=call))
        self.rows.append(dict(seed, elapsed_ms=10, phase="probe_finished"))

    def tearDown(self):
        self.temp.cleanup()

    def run_summary(self):
        (self.root / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        (self.root / "result.json").write_text(json.dumps(self.raw))
        return summarize(self.root / "trace.jsonl", self.root / "result.json")

    def test_distinct_peak_accounting_is_preserved(self):
        result = self.run_summary()
        self.assertEqual(result["maxima"]["resident_peak_bytes"], 160)
        self.assertEqual(result["maxima"]["rusage_peak_rss_bytes"], 155)
        self.assertEqual(result["maxima"]["physical_footprint_bytes"], 140)
        self.assertNotIn("relieved_bytes", result["maxima"])

    def test_anchor_must_be_unambiguous(self):
        for text in ("none", "twice twice"):
            with self.assertRaises(ValueError):
                replace_once(text, "twice", "once")

    def test_frozen_runner_load_anchor_remains_supported(self):
        instrumented = instrument_runner(runner_fixture(LEGACY_LOAD))
        self.assertIn('tmp=read(j["template"].toString()); fpm_memory::stage("images_decoded"); Settings settings;', instrumented)
        self.assertEqual(instrumented.count('stage("images_decoded")'), 1)

    def test_gray_loader_probe_preserves_native_measurement_boundaries(self):
        instrumented = instrument_runner(runner_fixture(GRAY_LOAD))
        self.assertIn(GRAY_LOAD + ' fpm_memory::stage("images_decoded");', instrumented)
        self.assertEqual(instrumented.count('stage("images_decoded")'), 1)

    def test_unknown_or_ambiguous_load_layout_fails_closed(self):
        for load in ("unknown load", LEGACY_LOAD + GRAY_LOAD, LEGACY_LOAD * 2, GRAY_LOAD * 2):
            with self.subTest(load=load), self.assertRaises(ValueError):
                instrument_runner(runner_fixture(load))

    def test_resource_runner_preserves_native_probe_boundaries(self):
        instrumented = instrument_runner(resource_runner_fixture())
        self.assertIn('fpm_memory::beginCall("first"); fpm::diagnostics::RunResourceProbe firstProbe(', instrumented)
        self.assertIn('firstProbe.finish(); fpm_memory::stage("after_run");', instrumented)
        self.assertIn('fpm_memory::beginCall("repeat"); fpm::diagnostics::RunResourceProbe probe(', instrumented)
        self.assertIn('probe.finish(); fpm_memory::stage("after_run");', instrumented)
        self.assertIn('runs.append(row); fpm_memory::stage("after_serialization");', instrumented)
        self.assertEqual(instrumented.count('stage("before_session_release")'), 1)
        self.assertEqual(instrumented.count('stage("after_session_release")'), 1)

    def test_mixed_call_layouts_fail_closed(self):
        old = runner_fixture(GRAY_LOAD)
        for text in (old.replace(LEGACY_FIRST, RESOURCE_FIRST),
                     old.replace(LEGACY_REPEAT, RESOURCE_REPEAT),
                     old + RESOURCE_FIRST + RESOURCE_REPEAT):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "first/repeat layout"):
                instrument_runner(text)

    def test_missing_or_duplicate_call_anchors_fail_closed(self):
        for text, anchors in ((runner_fixture(LEGACY_LOAD), (LEGACY_FIRST, LEGACY_REPEAT)),
                              (resource_runner_fixture(), (RESOURCE_FIRST, RESOURCE_REPEAT))):
            for anchor in anchors:
                for replacement in ("unknown call", anchor * 2):
                    with self.subTest(anchor=anchor, replacement=replacement), self.assertRaisesRegex(ValueError, "first/repeat layout"):
                        instrument_runner(text.replace(anchor, replacement))

    def test_current_runner_layout_is_supported(self):
        runner = (ROOT / "cpp-core/tools/native_pipeline_runner.cpp").read_text()
        instrumented = instrument_runner(runner)
        self.assertEqual(instrumented.count('beginCall("first")'), 1)
        self.assertEqual(instrumented.count('beginCall("repeat")'), 1)
        self.assertEqual(instrumented.count('beginCall("warmup")'), 1)

    def test_runner_header_comes_from_selected_source_and_is_hashed(self):
        frozen = self.root / "frozen"
        header = frozen / "cpp-core/tools/qt_gray_loader.hpp"
        header.parent.mkdir(parents=True)
        header.write_text("// frozen loader contents\n")
        source = self.root / "source"
        source.mkdir()
        originals = copy_runner_headers(frozen, source, '#include "qt_gray_loader.hpp"')
        self.assertEqual((source / header.name).read_bytes(), header.read_bytes())
        self.assertEqual(originals, {str(header): sha(header)})

    def test_legacy_runner_does_not_need_new_loader_header(self):
        self.assertEqual(copy_runner_headers(self.root, self.root, runner_fixture(LEGACY_LOAD)), {})

    def test_missing_selected_loader_header_fails_closed(self):
        with self.assertRaises(FileNotFoundError):
            copy_runner_headers(self.root, self.root, '#include "qt_gray_loader.hpp"')

    def test_resource_header_comes_from_selected_source_and_is_hashed(self):
        frozen = self.root / "frozen"
        header = frozen / "cpp-core/tools/run_resource_probe.hpp"
        header.parent.mkdir(parents=True)
        header.write_text("// selected source resource probe\n")
        source = self.root / "source"
        source.mkdir()
        originals = copy_runner_headers(frozen, source, '#include "run_resource_probe.hpp"')
        self.assertEqual((source / header.name).read_bytes(), header.read_bytes())
        self.assertEqual(originals, {str(header): sha(header)})

    def test_missing_selected_resource_header_fails_closed(self):
        # The live tree has this header; a missing selected/frozen header must
        # not silently fall back to ROOT or an already-existing output copy.
        (self.root / "run_resource_probe.hpp").write_text("// stale output\n")
        with self.assertRaises(FileNotFoundError):
            copy_runner_headers(self.root, self.root, '#include "run_resource_probe.hpp"')

    def test_both_runner_dependencies_are_copied_and_hashed(self):
        frozen = self.root / "frozen"
        folder = frozen / "cpp-core/tools"
        folder.mkdir(parents=True)
        source = self.root / "source"
        source.mkdir()
        names = ("qt_gray_loader.hpp", "run_resource_probe.hpp")
        for name in names:
            (folder / name).write_text("// selected " + name + "\n")
        originals = copy_runner_headers(frozen, source, '\n'.join('#include "' + name + '"' for name in names))
        self.assertEqual(originals, {str(folder / name): sha(folder / name) for name in names})
        for name in names:
            self.assertEqual((source / name).read_bytes(), (folder / name).read_bytes())

    def test_second_required_dependency_cannot_be_omitted(self):
        folder = self.root / "cpp-core/tools"
        folder.mkdir(parents=True)
        (folder / "qt_gray_loader.hpp").write_text("// loader\n")
        with self.assertRaises(FileNotFoundError):
            copy_runner_headers(self.root, self.root, '#include "qt_gray_loader.hpp"\n#include "run_resource_probe.hpp"')

    def test_each_target_keeps_its_floating_point_contract(self):
        for target, options in (("runner", "-O3 -std=c++17"),
                                ("pipeline", "-O3 -std=c++17 -ffp-contract=off")):
            directory = self.root / "experiments/CMakeFiles" / (target + ".dir")
            directory.mkdir(parents=True)
            (directory / "flags.make").write_text(
                "CXX_DEFINES = -DTARGET_" + target + "\nCXX_INCLUDES = -I" + str(ROOT / "cpp-core/include")
                + "\nCXX_FLAGS = " + options + "\n")
        frozen = self.root / "frozen"
        runner, _ = target_compile_flags(self.root, "runner", frozen)
        pipeline, _ = target_compile_flags(self.root, "pipeline", frozen)
        self.assertNotIn("-ffp-contract=off", runner)
        self.assertIn("-ffp-contract=off", pipeline)
        self.assertIn("-DTARGET_pipeline", pipeline)
        self.assertNotIn("-DTARGET_runner", pipeline)
        self.assertIn("-I" + str(frozen / "cpp-core/include"), pipeline)

    def test_missing_target_flags_fail_closed(self):
        with self.assertRaises(FileNotFoundError):
            target_compile_flags(self.root, "missing")

    def test_missing_end_rejected(self):
        self.rows.pop()
        with self.assertRaises(ValueError): self.run_summary()

    def test_unavailable_counter_rejected(self):
        self.rows[2]["physical_footprint_bytes"] = -1
        with self.assertRaises(ValueError): self.run_summary()

    def test_out_of_order_rejected(self):
        self.rows[2]["elapsed_ms"] = -1
        with self.assertRaises(ValueError): self.run_summary()

    def test_missing_call_rejected(self):
        del self.rows[2]
        with self.assertRaises(ValueError): self.run_summary()

    def test_repeat_flag_cannot_hide_changed_outputs(self):
        self.raw["runs"][0]["matches"] = []
        with self.assertRaises(ValueError): self.run_summary()

    def test_nonfinite_clock_rejected(self):
        self.rows[2]["elapsed_ms"] = float("nan")
        with self.assertRaises(ValueError): self.run_summary()

    def test_reference_mismatch_rejected(self):
        self.run_summary()
        other = copy.deepcopy(self.raw)
        other["first"]["matches"][0]["score"] = .8
        (self.root / "reference.json").write_text(json.dumps(other))
        with self.assertRaises(ValueError):
            summarize(self.root / "trace.jsonl", self.root / "result.json", self.root / "reference.json")


if __name__ == "__main__":
    unittest.main()
