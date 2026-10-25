"""Pilot kernel variants (scratch). Builds fpm_cli per variant; not formal evidence."""
import shutil
import subprocess
import sys
from pathlib import Path

worktree, out = Path(sys.argv[1]), Path(sys.argv[2])
original = (worktree / "cpp-core/src/ncc.cpp").read_text()
MAIN_V1 = """			uint32x4_t sum = vdupq_n_u32(0), odd = vdupq_n_u32(0);
			for (; x + 32 <= end; x += 32) {
				sum = vdotq_u32(sum, vld1q_u8(a + x), vld1q_u8(b + x));
				odd = vdotq_u32(odd, vld1q_u8(a + x + 16), vld1q_u8(b + x + 16));
			}
			if (x < end) { sum = vdotq_u32(sum, vld1q_u8(a + x), vld1q_u8(b + x)); x += 16; }
			total += vaddvq_u32(vaddq_u32(sum, odd));
"""
MAIN_ONE = """			uint32x4_t sum = vdupq_n_u32(0);
			for (; x < end; x += 16) sum = vdotq_u32(sum, vld1q_u8(a + x), vld1q_u8(b + x));
			total += vaddvq_u32(sum);
"""
TAIL_COND = "if (x >= 16 && x < size) {"
assert original.count(MAIN_V1) == 1 and original.count(TAIL_COND) == 1
VARIANTS = {
    "v1_two_mask1": original,
    "v2_one_scalar": original.replace(MAIN_V1, MAIN_ONE).replace(TAIL_COND, "if (false) {"),
    "v3_two_scalar": original.replace(TAIL_COND, "if (false) {"),
    "v4_one_mask1": original.replace(MAIN_V1, MAIN_ONE),
    "v5_two_mask4": original.replace(TAIL_COND, "if (x >= 16 && size - x >= 4) {"),
    "v6_two_mask8": original.replace(TAIL_COND, "if (x >= 16 && size - x >= 8) {"),
}
for name, text in VARIANTS.items():
    src = out / name / "src"
    if src.exists():
        shutil.rmtree(src)
    for part in ("cpp-core/src", "cpp-core/include", "cpp-core/tests", "cpp-core/tools", "cpp-core/examples"):
        shutil.copytree(worktree / part, src / part)
    shutil.copy(worktree / "cpp-core/CMakeLists.txt", src / "cpp-core/CMakeLists.txt")
    shutil.copytree(worktree / "macos-cli", src / "macos-cli")
    shutil.copy(worktree / "LICENSE", src / "LICENSE")
    (src / "cpp-core/src/ncc.cpp").write_text(text)
    build = out / name / "build"
    with (out / name / "build.log").open("w") as log:
        subprocess.run(["cmake", "-S", str(src / "cpp-core"), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                        "-DFPM_WITH_OPENCV=ON"], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "fpm_cli", "fpm_core_test", "-j", "8"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    test = subprocess.run([str(build / "fpm_core_test")], capture_output=True, text=True)
    print(name, "core_test exit", test.returncode, flush=True)
