// SPDX-License-Identifier: BSD-2-Clause
#pragma once
// A Metal test may return 77 (skipped) only when this process has no Metal
// device. The probe does not use the backends under test, and messages are
// never matched: CPU default implementations also report "... is unavailable",
// so a routing regression must fail rather than be reported as skipped.
namespace fpm_test {
#ifdef __APPLE__
bool metalDeviceAvailable();
#else
inline bool metalDeviceAvailable() { return false; }
#endif
}
