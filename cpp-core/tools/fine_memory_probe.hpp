// SPDX-License-Identifier: BSD-2-Clause
// Diagnostic-only instrumentation. Never link this into a speed benchmark.
#pragma once
#include <mach/mach.h>
#include <malloc/malloc.h>
#include <sys/resource.h>
#include <unistd.h>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace fpm_memory {
struct State {
    FILE* file = nullptr;
    std::mutex mutex;
    std::condition_variable wake;
    std::thread sampler;
    bool stopping = false;
    bool relief = false;
    const char* phase = "starting";
    const char* callKind = "none";
    int call = -1, model = -1;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
};
inline State* state = nullptr;

// Caller holds state->mutex. No retained per-sample collection or QJson object.
inline void sampleLocked(const char* event, size_t relieved = 0) {
    auto& s = *state;
    task_vm_info_data_t vm{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    const int status = task_info(mach_task_self(), TASK_VM_INFO,
        reinterpret_cast<task_info_t>(&vm), &count);
    const bool vmValid = status == KERN_SUCCESS && count >= TASK_VM_INFO_REV0_COUNT;
    const bool footprintValid = status == KERN_SUCCESS && count >= TASK_VM_INFO_REV1_COUNT;
    const bool peakFootprintValid = status == KERN_SUCCESS && count >= TASK_VM_INFO_REV3_COUNT;
    rusage ru{};
    const bool rssValid = getrusage(RUSAGE_SELF, &ru) == 0;
    malloc_statistics_t allocation{};
    malloc_zone_statistics(nullptr, &allocation); // All malloc zones; not Metal allocations.
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - s.start).count();
    // -1 means unavailable, never a fabricated zero. VM counters overlap and
    // malloc size_allocated is reserved capacity, not measured resident bytes.
    auto v = [vmValid](uint64_t n) { return vmValid ? static_cast<long long>(n) : -1LL; };
    std::fprintf(s.file,
        "{\"elapsed_ms\":%.6f,\"event\":\"%s\",\"phase\":\"%s\",\"call_kind\":\"%s\","
        "\"call\":%d,\"model\":%d,\"pid\":%d,\"task_info_status\":%d,\"task_info_count\":%u,"
        "\"resident_bytes\":%lld,\"resident_peak_bytes\":%lld,\"rusage_peak_rss_bytes\":%lld,"
        "\"physical_footprint_bytes\":%lld,\"physical_footprint_peak_bytes\":%lld,"
        "\"compressed_bytes\":%lld,\"compressed_peak_bytes\":%lld,\"compressed_lifetime_bytes\":%lld,"
        "\"internal_bytes\":%lld,\"external_bytes\":%lld,\"device_bytes\":%lld,\"reusable_bytes\":%lld,"
        "\"malloc_in_use_bytes\":%llu,\"malloc_reserved_bytes\":%llu,"
        "\"malloc_max_touched_bytes\":%llu,\"malloc_blocks_in_use\":%u,\"relieved_bytes\":%llu}\n",
        elapsed, event, s.phase, s.callKind, s.call, s.model, int(getpid()), status, count,
        v(vm.resident_size), v(vm.resident_size_peak), rssValid ? static_cast<long long>(ru.ru_maxrss) : -1LL,
        footprintValid ? static_cast<long long>(vm.phys_footprint) : -1LL,
        peakFootprintValid ? static_cast<long long>(vm.ledger_phys_footprint_peak) : -1LL,
        v(vm.compressed), v(vm.compressed_peak), v(vm.compressed_lifetime),
        v(vm.internal), v(vm.external), v(vm.device), v(vm.reusable),
        static_cast<unsigned long long>(allocation.size_in_use),
        static_cast<unsigned long long>(allocation.size_allocated),
        static_cast<unsigned long long>(allocation.max_size_in_use), allocation.blocks_in_use,
        static_cast<unsigned long long>(relieved));
    std::fflush(s.file);
}
inline void stage(const char* phase, int model = -1) {
    if (!state) return;
    std::lock_guard<std::mutex> lock(state->mutex);
    state->phase = phase; state->model = model; sampleLocked("stage");
}
inline void beginCall(const char* kind) {
    if (!state) return;
    std::lock_guard<std::mutex> lock(state->mutex);
    ++state->call; state->callKind = kind; state->model = -1;
    state->phase = "before_run"; sampleLocked("stage");
}
inline void pressureRelief(const char* phase) {
    if (!state || !state->relief) return;
    std::lock_guard<std::mutex> lock(state->mutex);
    state->phase = phase;
    sampleLocked("before_pressure_relief");
    const size_t released = malloc_zone_pressure_relief(nullptr, 0);
    sampleLocked("after_pressure_relief", released);
}
class Probe {
public:
    Probe() {
        if (state) throw std::runtime_error("Duplicate memory probe");
        const char* path = std::getenv("FPM_MEMORY_TRACE");
        if (!path) throw std::runtime_error("FPM_MEMORY_TRACE is required");
        auto* s = new State;
        s->file = std::fopen(path, "wx");
        if (!s->file) { delete s; throw std::runtime_error("Cannot create memory trace"); }
        const char* relief = std::getenv("FPM_MEMORY_RELIEF");
        s->relief = relief && relief[0] == '1';
        state = s;
        stage("probe_started");
        const char* value = std::getenv("FPM_MEMORY_INTERVAL_MS");
        const int interval = value ? std::atoi(value) : 100;
        if (interval > 0) s->sampler = std::thread([s, interval] {
            std::unique_lock<std::mutex> lock(s->mutex);
            while (!s->wake.wait_for(lock, std::chrono::milliseconds(interval), [s] { return s->stopping; }))
                sampleLocked("sample");
        });
    }
    ~Probe() {
        auto* s = state;
        if (!s) return;
        {
            std::lock_guard<std::mutex> lock(s->mutex);
            s->stopping = true; s->wake.notify_all();
        }
        if (s->sampler.joinable()) s->sampler.join();
        stage("probe_finished");
        std::fclose(s->file); state = nullptr; delete s;
    }
    Probe(const Probe&) = delete;
    Probe& operator=(const Probe&) = delete;
};
} // namespace fpm_memory
