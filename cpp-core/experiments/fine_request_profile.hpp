// SPDX-License-Identifier: BSD-2-Clause
#pragma once
#include "internal.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace fpm_coarse_detail {
struct FineRequestProfileStats {
    std::uint64_t calls = 0, requests = 0;
    // Disjoint counts: all repeat occurrences within a call, then one per
    // distinct call key that was already retained from an earlier call.
    std::uint64_t within_call_duplicates = 0, cross_call_duplicates = 0;
    // Distinct keys in each call that could not enter the bounded history.
    // These may repeat untracked keys from earlier calls; this is NOT a count
    // of globally unique requests. Cross-call duplicates are a lower bound
    // after saturation. Within-call duplicates remain exact.
    std::uint64_t untracked_call_keys = 0, tracked_unique_keys = 0;
    std::uint64_t history_limit_keys = 0, history_key_payload_bytes = 0;
    std::uint64_t history_bucket_count = 0, peak_call_key_capacity_bytes = 0;
};

// An observational profiler, not a cache. Equality describes the inputs to
// rotatedRoi only: templates with equal dimensions share ROI geometry but may
// have different pixels and NCC scores. Source identity is valid only while
// its bytes are immutable; reset at each model/frame lifetime or invalidation.
// A pointer reused for changed image data must not share this history.
class FineRequestProfiler {
public:
    static constexpr std::size_t maxHistoryKeys = 65536;

    explicit FineRequestProfiler(std::size_t historyLimit = maxHistoryKeys)
        : historyLimit_(historyLimit)
    {
        if (historyLimit > maxHistoryKeys)
            throw std::invalid_argument("fine request history limit exceeds 65536 keys");
    }

    void reset()
    {
        // Release nodes and buckets, including when profiling is disabled.
        decltype(history_) empty;
        history_.swap(empty);
        counts_ = {};
    }

    void observe(fpm::ImageView source, const fpm::TemplateLevel& templ,
        const std::vector<fpm::detail::FineRequest>& requests)
    {
        // This allocation exists only for this call. It is not retained as
        // batches grow/shrink; capacity bytes are recorded before destruction.
        std::vector<Key> keys;
        keys.reserve(requests.size());
        for (const auto& request : requests) {
            keys.push_back({reinterpret_cast<std::uintptr_t>(source.data),
                source.width, source.height, source.stride,
                templ.image.width(), templ.image.height(),
                bits(request.anchor.x), bits(request.anchor.y), bits(request.angle)});
        }
        std::sort(keys.begin(), keys.end());
        const auto uniqueEnd = std::unique(keys.begin(), keys.end());
        const auto uniqueCount = std::size_t(uniqueEnd - keys.begin());
        ++counts_.calls;
        counts_.requests += requests.size();
        counts_.within_call_duplicates += requests.size() - uniqueCount;
        counts_.peak_call_key_capacity_bytes = std::max(
            counts_.peak_call_key_capacity_bytes,
            std::uint64_t(keys.capacity()) * sizeof(Key));
        for (auto it = keys.begin(); it != uniqueEnd; ++it) {
            if (history_.find(*it) != history_.end()) ++counts_.cross_call_duplicates;
            else if (history_.size() < historyLimit_) history_.insert(*it);
            else ++counts_.untracked_call_keys;
        }
    }

    FineRequestProfileStats stats() const
    {
        auto value = counts_;
        value.tracked_unique_keys = history_.size();
        value.history_limit_keys = historyLimit_;
        // Payload only: unordered-set links, allocator metadata/alignment and
        // bucket storage are additional. Bucket count is reported separately;
        // its representation is implementation-defined. Neither quantity is
        // process RSS or exact heap allocation. Per-call vector capacity also
        // excludes allocator overhead and the caller-owned request vector.
        value.history_key_payload_bytes = std::uint64_t(history_.size()) * sizeof(Key);
        value.history_bucket_count = history_.bucket_count();
        return value;
    }

private:
    struct Key {
        std::uintptr_t source;
        int sourceWidth, sourceHeight;
        std::size_t sourceStride;
        int templateWidth, templateHeight;
        std::uint64_t x, y, angle;

        auto fields() const
        {
            return std::tie(source, sourceWidth, sourceHeight, sourceStride,
                templateWidth, templateHeight, x, y, angle);
        }
        bool operator==(const Key& other) const { return fields() == other.fields(); }
        bool operator<(const Key& other) const { return fields() < other.fields(); }
    };
    struct KeyHash {
        std::size_t operator()(const Key& key) const
        {
            std::size_t hash = 0;
            const auto mix = [&hash](std::uint64_t value) {
                hash ^= std::hash<std::uint64_t>{}(value) +
                    std::size_t(0x9e3779b97f4a7c15ULL) + (hash << 6) + (hash >> 2);
            };
            mix(key.source); mix(std::uint64_t(key.sourceWidth));
            mix(std::uint64_t(key.sourceHeight)); mix(key.sourceStride);
            mix(std::uint64_t(key.templateWidth)); mix(std::uint64_t(key.templateHeight));
            mix(key.x); mix(key.y); mix(key.angle);
            return hash;
        }
    };
    static std::uint64_t bits(double value)
    {
        static_assert(sizeof(double) == sizeof(std::uint64_t), "64-bit doubles required");
        std::uint64_t result;
        std::memcpy(&result, &value, sizeof(result));
        return result;
    }

    std::size_t historyLimit_;
    std::unordered_set<Key, KeyHash> history_;
    FineRequestProfileStats counts_;
};
} // namespace fpm_coarse_detail
