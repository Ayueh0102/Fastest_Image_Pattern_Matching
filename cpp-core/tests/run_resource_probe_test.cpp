// SPDX-License-Identifier: BSD-2-Clause
#include "run_resource_probe.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using fpm::diagnostics::RunResourceProbe;
using fpm::diagnostics::RunResourceRecord;
using fpm::diagnostics::detail::RunResourceSnapshot;
using fpm::diagnostics::detail::runResourceDifference;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void unavailableProcess(const RunResourceRecord& value)
{
    require(!value.available && value.process_user_ms == -1 &&
        value.process_system_ms == -1 && value.minor_faults == -1 &&
        value.major_faults == -1 && value.voluntary_switches == -1 &&
        value.involuntary_switches == -1, "invalid process counters became fabricated values");
}

RunResourceSnapshot snapshot()
{
    RunResourceSnapshot value;
    value.wall = std::chrono::steady_clock::time_point(std::chrono::seconds(20));
    value.process_available = true;
    value.user_us = 1000000;
    value.system_us = 2000000;
    value.minor_faults = 30;
    value.major_faults = 40;
    value.voluntary_switches = 50;
    value.involuntary_switches = 60;
    value.host_load_available = true;
    value.host_user_ticks = 100;
    value.host_system_ticks = 200;
    value.host_nice_ticks = 300;
    value.host_idle_ticks = 400;
    return value;
}

void knownDifference()
{
    const auto first = snapshot();
    auto last = first;
    last.wall += std::chrono::microseconds(12345);
    last.user_us += 23456;
    last.system_us += 34567;
    last.minor_faults += 7;
    last.major_faults += 8;
    last.voluntary_switches += 9;
    last.involuntary_switches += 10;
    last.host_user_ticks += 11;
    last.host_system_ticks += 12;
    last.host_nice_ticks += 13;
    last.host_idle_ticks += 14;
    const auto value = runResourceDifference(first, last);
    require(value.available && std::abs(value.wall_ms - 12.345) < 1e-9 &&
        std::abs(value.process_user_ms - 23.456) < 1e-9 &&
        std::abs(value.process_system_ms - 34.567) < 1e-9 &&
        value.minor_faults == 7 && value.major_faults == 8 &&
        value.voluntary_switches == 9 && value.involuntary_switches == 10,
        "known process counter differences are wrong");
    require(value.host_load_available && value.host_busy_ticks == 36 &&
        value.host_idle_ticks == 14, "host counters did not aggregate all busy CPU states");
    const auto zero = runResourceDifference(first, first);
    require(zero.available && zero.wall_ms == 0 && zero.process_user_ms == 0 &&
        zero.major_faults == 0, "equal valid snapshots should produce valid zero deltas");
}

void invalidSnapshots()
{
    const auto valid = snapshot();
    auto invalid = valid;
    invalid.process_available = false;
    unavailableProcess(runResourceDifference(invalid, valid));
    unavailableProcess(runResourceDifference(valid, invalid));
    for (auto member : {&RunResourceSnapshot::user_us, &RunResourceSnapshot::system_us,
        &RunResourceSnapshot::minor_faults, &RunResourceSnapshot::major_faults,
        &RunResourceSnapshot::voluntary_switches, &RunResourceSnapshot::involuntary_switches}) {
        invalid = valid;
        invalid.*member = -1;
        unavailableProcess(runResourceDifference(invalid, valid));
        unavailableProcess(runResourceDifference(valid, invalid));
        invalid = valid;
        --(invalid.*member);
        unavailableProcess(runResourceDifference(valid, invalid));
    }
    invalid = valid;
    invalid.wall -= std::chrono::nanoseconds(1);
    const auto backwards = runResourceDifference(valid, invalid);
    unavailableProcess(backwards);
    require(backwards.wall_ms == -1 && !backwards.host_load_available &&
        backwards.host_busy_ticks == -1, "backwards wall clock was accepted");
    invalid = valid;
    invalid.host_load_available = false;
    for (const auto& value : {runResourceDifference(invalid, valid),
        runResourceDifference(valid, invalid)})
        require(value.available && !value.host_load_available &&
            value.host_busy_ticks == -1 && value.host_idle_ticks == -1,
            "missing optional host snapshot damaged valid process counters");
    invalid = valid;
    --invalid.host_user_ticks;
    const auto reset = runResourceDifference(valid, invalid);
    require(reset.available && !reset.host_load_available && reset.host_busy_ticks == -1,
        "backwards host counter produced a huge apparent busy interval");
}

void counterLimits()
{
    auto first = snapshot(), last = first;
    first.user_us = std::numeric_limits<std::int64_t>::max() - 1000;
    last.user_us = std::numeric_limits<std::int64_t>::max();
    first.host_user_ticks = std::numeric_limits<std::uint32_t>::max() - 2;
    last.host_user_ticks = 4;
    const auto value = runResourceDifference(first, last);
    require(value.available && value.process_user_ms == 1,
        "large valid process counters overflowed during subtraction");
    require(value.host_load_available && value.host_busy_ticks == 7,
        "single 32-bit host counter wrap was not handled");
    first = snapshot(); last = first;
    first.host_user_ticks = first.host_system_ticks = first.host_nice_ticks = 0;
    last.host_user_ticks = last.host_system_ticks = last.host_nice_ticks =
        std::numeric_limits<std::int32_t>::max();
    require(runResourceDifference(first, last).host_busy_ticks ==
        std::int64_t(std::numeric_limits<std::int32_t>::max()) * 3,
        "host busy total overflowed the 32-bit per-state counter type");
    std::int64_t result = -1;
    timeval time{};
    time.tv_sec = 123; time.tv_usec = 456789;
    require(fpm::diagnostics::detail::timevalMicroseconds(time, result) && result == 123456789,
        "timeval conversion lost microseconds");
    for (auto microseconds : {-1, 1000000}) {
        time.tv_usec = microseconds;
        require(!fpm::diagnostics::detail::timevalMicroseconds(time, result),
            "invalid timeval fraction was accepted");
    }
    time.tv_sec = -1; time.tv_usec = 0;
    require(!fpm::diagnostics::detail::timevalMicroseconds(time, result),
        "negative timeval seconds were accepted");
    if constexpr (std::numeric_limits<decltype(time.tv_sec)>::digits >= 63) {
        time.tv_sec = static_cast<decltype(time.tv_sec)>(
            std::numeric_limits<std::int64_t>::max() / 1000000 + 1);
        require(!fpm::diagnostics::detail::timevalMicroseconds(time, result),
            "timeval microsecond conversion overflow was accepted");
    }
}

void liveProbe()
{
    RunResourceProbe disabled(false);
    const auto absent = disabled.finish();
    unavailableProcess(absent);
    require(absent.wall_ms == -1 && !absent.host_load_available &&
        absent.host_busy_ticks == -1 && absent.host_idle_ticks == -1,
        "disabled probe did not preserve unavailable sentinels");
    RunResourceProbe probe(true);
    volatile std::uint64_t work = 3;
    for (int i = 0; i < 2000000; ++i) work = work * 6364136223846793005ULL + 1;
    const auto value = probe.finish();
    require(value.available && value.wall_ms >= 0 && value.process_user_ms >= 0 &&
        value.process_system_ms >= 0 && value.minor_faults >= 0 && value.major_faults >= 0 &&
        value.voluntary_switches >= 0 && value.involuntary_switches >= 0,
        "live POSIX probe failed or returned negative resource deltas");
    if (value.host_load_available)
        require(value.host_busy_ticks >= 0 && value.host_idle_ticks >= 0,
            "live host counters returned negative deltas");
    // Finishing twice must not turn one run into two different intervals.
    const auto repeated = probe.finish();
    require(value.wall_ms == repeated.wall_ms &&
        value.process_user_ms == repeated.process_user_ms &&
        value.involuntary_switches == repeated.involuntary_switches &&
        value.host_busy_ticks == repeated.host_busy_ticks,
        "second finish captured a different interval");
}
} // namespace

int main()
{
    try {
        knownDifference(); invalidSnapshots(); counterLimits(); liveProbe();
        std::cout << "Run resource probe validates process deltas, unavailable counters, host wraps and opt-in capture\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
