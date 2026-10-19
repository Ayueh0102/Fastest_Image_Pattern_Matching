// SPDX-License-Identifier: BSD-2-Clause
// Opt-in, two-snapshot diagnostics. These counters are not uninstrumented
// benchmark timings; process CPU time can exceed wall time with multiple threads.
#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <sys/resource.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace fpm::diagnostics {

struct RunResourceRecord {
    // available validates the process counter interval. Optional host counters
    // have their own flag; wall_ms can remain valid after getrusage fails.
    bool available = false;
    double wall_ms = -1;
    double process_user_ms = -1;
    double process_system_ms = -1;
    std::int64_t minor_faults = -1;
    std::int64_t major_faults = -1;
    std::int64_t voluntary_switches = -1;
    std::int64_t involuntary_switches = -1;
    // Apple host CPU counters cover the WHOLE MACHINE, including other
    // processes. They do not measure this process's CPU occupancy or GPU load.
    bool host_load_available = false;
    std::int64_t host_busy_ticks = -1;
    std::int64_t host_idle_ticks = -1;
};

namespace detail {
struct RunResourceSnapshot {
    std::chrono::steady_clock::time_point wall{};
    bool process_available = false;
    std::int64_t user_us = -1;
    std::int64_t system_us = -1;
    std::int64_t minor_faults = -1;
    std::int64_t major_faults = -1;
    std::int64_t voluntary_switches = -1;
    std::int64_t involuntary_switches = -1;
    bool host_load_available = false;
    std::uint32_t host_user_ticks = 0;
    std::uint32_t host_system_ticks = 0;
    std::uint32_t host_nice_ticks = 0;
    std::uint32_t host_idle_ticks = 0;
};

inline bool timevalMicroseconds(const timeval& value, std::int64_t& result)
{
    if (value.tv_sec < 0 || value.tv_usec < 0 || value.tv_usec >= 1000000)
        return false;
    constexpr auto limit = std::numeric_limits<std::int64_t>::max();
    if (value.tv_sec > (limit - value.tv_usec) / 1000000) return false;
    result = static_cast<std::int64_t>(value.tv_sec) * 1000000 + value.tv_usec;
    return true;
}

inline RunResourceSnapshot captureRunResourceSnapshot()
{
    RunResourceSnapshot result;
    result.wall = std::chrono::steady_clock::now();
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0 &&
        timevalMicroseconds(usage.ru_utime, result.user_us) &&
        timevalMicroseconds(usage.ru_stime, result.system_us)) {
        result.process_available = true;
        result.minor_faults = usage.ru_minflt;
        result.major_faults = usage.ru_majflt;
        result.voluntary_switches = usage.ru_nvcsw;
        result.involuntary_switches = usage.ru_nivcsw;
    }
#if defined(__APPLE__)
    host_cpu_load_info_data_t cpu{};
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    const host_t host = mach_host_self();
    if (host != MACH_PORT_NULL) {
        const kern_return_t status = host_statistics(host, HOST_CPU_LOAD_INFO,
            reinterpret_cast<host_info_t>(&cpu), &count);
        // mach_host_self gives a send right: do not accumulate rights when
        // diagnostics wrap many Session.run calls in a single process.
        mach_port_deallocate(mach_task_self(), host);
        if (status == KERN_SUCCESS && count >= HOST_CPU_LOAD_INFO_COUNT) {
            result.host_load_available = true;
            result.host_user_ticks = cpu.cpu_ticks[CPU_STATE_USER];
            result.host_system_ticks = cpu.cpu_ticks[CPU_STATE_SYSTEM];
            result.host_nice_ticks = cpu.cpu_ticks[CPU_STATE_NICE];
            result.host_idle_ticks = cpu.cpu_ticks[CPU_STATE_IDLE];
        }
    }
#endif
    return result;
}

inline bool validCounterPair(std::int64_t first, std::int64_t last)
{
    return first >= 0 && last >= first;
}

// Host tick counters are 32-bit. Permit a single natural wrap with an unsigned
// delta, rejecting backwards/reset-like changes larger than half the range.
// This diagnostic is intended for individual runs, not intervals long enough
// to accumulate 2^31 ticks per counter or to span multiple wraps.
inline bool hostTickDelta(std::uint32_t first, std::uint32_t last,
    std::uint32_t& result)
{
    result = last - first;
    return result <= std::numeric_limits<std::int32_t>::max();
}

inline RunResourceRecord runResourceDifference(const RunResourceSnapshot& first,
    const RunResourceSnapshot& last)
{
    RunResourceRecord result;
    if (last.wall < first.wall) return result;
    result.wall_ms = std::chrono::duration<double, std::milli>(last.wall - first.wall).count();
    if (first.process_available && last.process_available &&
        validCounterPair(first.user_us, last.user_us) &&
        validCounterPair(first.system_us, last.system_us) &&
        validCounterPair(first.minor_faults, last.minor_faults) &&
        validCounterPair(first.major_faults, last.major_faults) &&
        validCounterPair(first.voluntary_switches, last.voluntary_switches) &&
        validCounterPair(first.involuntary_switches, last.involuntary_switches)) {
        result.available = true;
        result.process_user_ms = static_cast<double>(last.user_us - first.user_us) / 1000;
        result.process_system_ms = static_cast<double>(last.system_us - first.system_us) / 1000;
        result.minor_faults = last.minor_faults - first.minor_faults;
        result.major_faults = last.major_faults - first.major_faults;
        result.voluntary_switches = last.voluntary_switches - first.voluntary_switches;
        result.involuntary_switches = last.involuntary_switches - first.involuntary_switches;
    }
    std::uint32_t user = 0, system = 0, nice = 0, idle = 0;
    if (first.host_load_available && last.host_load_available &&
        hostTickDelta(first.host_user_ticks, last.host_user_ticks, user) &&
        hostTickDelta(first.host_system_ticks, last.host_system_ticks, system) &&
        hostTickDelta(first.host_nice_ticks, last.host_nice_ticks, nice) &&
        hostTickDelta(first.host_idle_ticks, last.host_idle_ticks, idle)) {
        result.host_load_available = true;
        result.host_busy_ticks = std::int64_t(user) + system + nice;
        result.host_idle_ticks = idle;
    }
    return result;
}
} // namespace detail

class RunResourceProbe {
public:
    explicit RunResourceProbe(bool enabled) : enabled_(enabled)
    {
        // The disabled path does not read clocks or system counters.
        if (enabled_) first_ = detail::captureRunResourceSnapshot();
    }
    RunResourceRecord finish()
    {
        if (enabled_ && !finished_) {
            result_ = detail::runResourceDifference(first_, detail::captureRunResourceSnapshot());
            finished_ = true;
        }
        return result_;
    }
    RunResourceProbe(const RunResourceProbe&) = delete;
    RunResourceProbe& operator=(const RunResourceProbe&) = delete;
private:
    bool enabled_ = false;
    bool finished_ = false;
    detail::RunResourceSnapshot first_{};
    RunResourceRecord result_{};
};

} // namespace fpm::diagnostics
