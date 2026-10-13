// SPDX-License-Identifier: BSD-2-Clause
// Appended to pose_ncc.metal only when the opt-in resident solver is first used.
// The original score/derivative kernels remain the pixel-computation oracle.
// A single 64-thread control group provides a real group barrier for compacting
// independent poses. It never spins on another threadgroup's progress.
struct ResidentState {
    float4 pose, lower, upper, packed, candidate, candidatePacked;
    float h[16], g[4];
    float damping, score;
    uint iterations, evaluations, status, active, pending, proposed;
};
static_assert(sizeof(ResidentState) == 208, "resident state layout mismatch");
struct ResidentControl { uint activeCount, candidateCount, reserved[2]; };
struct ResidentDispatch { uint x, y, z, padding; };
struct ResidentOptions { uint count, finishThreads, reserved[2]; };
struct ResidentTrial { uint first, final; };
// Each indirect entry has 16-byte stride; Metal reads the first three uints.
enum ResidentGrid { ResidentOuter, ResidentDerivativeTiles, ResidentDerivativeFinish,
    ResidentTrialControl, ResidentScoreTiles, ResidentScoreFinish };

inline ResidentDispatch resident_grid(uint x, uint y = 1, uint z = 1)
{
    return x && y && z ? ResidentDispatch{x, y, z, 0} : ResidentDispatch{0, 0, 0, 0};
}
inline uint resident_covariance(uint i, uint j) { return i*5-i*(i-1)/2+j-i; }

inline float4 resident_pack(float4 pose)
{
    // Public angles stay unwrapped and bounded relative to their original seed.
    // Reducing only the trig argument avoids FP32 loss at large legal angles.
    const float angle = fmod(pose.z, 360.f)*0.01745329251994329577f;
    const float scale = exp(pose.w);
    return float4(pose.xy, scale*cos(angle), scale*sin(angle));
}

inline bool resident_assemble(thread const DerivativePartial& p,
    device ResidentState& s, device const Config& k)
{
    const float n = float(k.tw*k.th), energy = max(0.f, p.covariance[0]);
    if (!isfinite(energy) || energy/n <= 1e-12f || k.templateEnergy/n <= 1e-12f) return false;
    float product = p.product[0], jx[4], jt[4];
    for (uint i = 0; i < 4; ++i) {
        jx[i] = p.covariance[resident_covariance(0, i+1)];
        jt[i] = p.product[i+1];
    }
    for (uint c = 0; c < k.channels; ++c) {
        product -= p.mean[c][0]*k.templateSum[c];
        for (uint i = 0; i < 4; ++i) jt[i] -= p.mean[c][i+1]*k.templateSum[c];
    }
    const float denominator = sqrt(energy*k.templateEnergy);
    const float score = clamp(product/denominator, -1.f, 1.f);
    if (!isfinite(score) || !isfinite(denominator)) return false;
    for (uint i = 0; i < 4; ++i) {
        s.g[i] = jt[i]/denominator-score*jx[i]/energy;
        for (uint j = i; j < 4; ++j) {
            const float value = p.covariance[resident_covariance(i+1, j+1)]/energy
                -(jx[i]/energy)*(jx[j]/energy);
            s.h[4*i+j] = value; s.h[4*j+i] = value;
        }
    }
    return true;
}

inline bool resident_solve(device const ResidentState& s, thread float4& step)
{
    // Solve D^-1 (H + lambda diag(max(Hii,1e-12))) D^-1 z = D^-1 g,
    // then step = D^-1 z. Translation and angle/scale have very different units;
    // diagonal equilibration keeps the FP32 elimination near unit scale.
    float m[4][5], inverseDiagonal[4];
    for (uint i = 0; i < 4; ++i) {
        inverseDiagonal[i] = 1.f/sqrt(max(s.h[4*i+i], 1e-12f));
        if (!isfinite(inverseDiagonal[i])) return false;
    }
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j)
            m[i][j] = s.h[4*i+j]*inverseDiagonal[i]*inverseDiagonal[j];
        m[i][i] += s.damping;
        m[i][4] = s.g[i]*inverseDiagonal[i];
    }
    for (uint i = 0; i < 4; ++i) {
        uint pivot = i;
        for (uint row = i+1; row < 4; ++row)
            if (abs(m[row][i]) > abs(m[pivot][i])) pivot = row;
        if (!isfinite(m[pivot][i]) || abs(m[pivot][i]) < 1e-12f) return false;
        for (uint j = i; j < 5; ++j) {
            const float temporary = m[i][j]; m[i][j] = m[pivot][j]; m[pivot][j] = temporary;
        }
        const float divisor = m[i][i];
        for (uint j = i; j < 5; ++j) m[i][j] /= divisor;
        for (uint row = 0; row < 4; ++row) if (row != i) {
            const float factor = m[row][i];
            for (uint j = i; j < 5; ++j) m[row][j] -= factor*m[i][j];
        }
    }
    for (uint i = 0; i < 4; ++i) step[i] = m[i][4]*inverseDiagonal[i];
    return all(isfinite(step));
}

kernel void resident_initialize(device ResidentState* states [[buffer(0)]],
    device const float* scores [[buffer(1)]], device ResidentControl& control [[buffer(2)]],
    device ResidentDispatch* grids [[buffer(3)]], constant ResidentOptions& options [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]])
{
    for (uint i = lane; i < options.count; i += 64) {
        device ResidentState& s = states[i];
        s.score = scores[i]; s.damping = 1e-3f;
        s.iterations = 0; s.evaluations = 1; s.pending = s.proposed = 0;
        s.status = !isfinite(s.score) ? 3u : (s.score == -2.f ? 2u : (s.score == 1.f ? 0u : 1u));
        s.active = s.status == 1u;
    }
    if (lane == 0) {
        control.activeCount = control.candidateCount = 0;
        for (uint i = 0; i < 6; ++i) grids[i] = resident_grid(0);
        grids[ResidentOuter] = resident_grid(1);
    }
}

kernel void resident_prepare(device ResidentState* states [[buffer(0)]],
    device ResidentControl& control [[buffer(1)]], device uint* activeMap [[buffer(2)]],
    device float4* packed [[buffer(3)]], device Config& k [[buffer(4)]],
    device ResidentDispatch* grids [[buffer(5)]], constant ResidentOptions& options [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup atomic_uint count;
    if (lane == 0) atomic_store_explicit(&count, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < options.count; i += 64) if (states[i].active) {
        const uint index = atomic_fetch_add_explicit(&count, 1u, memory_order_relaxed);
        activeMap[index] = i; packed[index] = states[i].packed;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (lane == 0) {
        const uint n = atomic_load_explicit(&count, memory_order_relaxed);
        control.activeCount = n; control.candidateCount = 0; k.count = n;
        grids[ResidentOuter] = resident_grid(n ? 1u : 0u);
        grids[ResidentDerivativeTiles] = resident_grid(k.tiles, n);
        grids[ResidentDerivativeFinish] = resident_grid((n+options.finishThreads-1)/options.finishThreads);
        grids[ResidentTrialControl] = resident_grid(n ? 1u : 0u);
        grids[ResidentScoreTiles] = grids[ResidentScoreFinish] = resident_grid(0);
    }
}

// first: assemble this iteration's statistics and propose trial zero.
// otherwise: accept/reject the previous score batch, then propose the next trial.
// final: only accept/reject trial nine and terminate any still-pending pose.
kernel void resident_trial(device ResidentState* states [[buffer(0)]],
    device ResidentControl& control [[buffer(1)]], device const uint* activeMap [[buffer(2)]],
    device uint* candidateMap [[buffer(3)]], device float4* packed [[buffer(4)]],
    device const uchar* results [[buffer(5)]], device Config& k [[buffer(6)]],
    device ResidentDispatch* grids [[buffer(7)]], constant ResidentOptions& options [[buffer(8)]],
    constant ResidentTrial& trial [[buffer(9)]], uint lane [[thread_index_in_threadgroup]])
{
    threadgroup atomic_uint candidates, pending;
    if (lane == 0) {
        atomic_store_explicit(&candidates, 0u, memory_order_relaxed);
        atomic_store_explicit(&pending, 0u, memory_order_relaxed);
    }
    if (!trial.first) {
        const device float* scores = reinterpret_cast<const device float*>(results);
        for (uint j = lane; j < control.candidateCount; j += 64) {
            device ResidentState& s = states[candidateMap[j]];
            const float next = scores[j]; ++s.evaluations;
            // Both comparison scores use the unchanged pose_partials/finish
            // kernels. Ignore changes smaller than two float epsilons.
            if (isfinite(next) && next > s.score+2.384185791015625e-7f) {
                s.pose = s.candidate; s.packed = s.candidatePacked; s.score = next;
                s.damping = max(1e-8f, s.damping*.3f); s.pending = 0;
            } else s.damping *= 10.f;
            s.proposed = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint j = lane; j < control.activeCount; j += 64) {
        const uint owner = activeMap[j];
        device ResidentState& s = states[owner];
        if (trial.first) {
            const device DerivativePartial* derivatives = reinterpret_cast<const device DerivativePartial*>(results);
            const DerivativePartial value = derivatives[j];
            ++s.iterations; ++s.evaluations;
            if (!resident_assemble(value, s, k)) {
                s.active = s.pending = 0; s.status = 2; continue;
            }
            s.pending = 1;
        }
        if (!s.active || !s.pending) continue;
        if (trial.final) { s.active = s.pending = 0; s.status = 3; continue; }
        float4 step;
        if (!resident_solve(s, step)) {
            s.damping *= 10.f;
            atomic_fetch_add_explicit(&pending, 1u, memory_order_relaxed);
            continue;
        }
        step.z /= 0.01745329251994329577f;
        const float ratio = max(1.f, max(max(abs(step.x)/2.f, abs(step.y)/2.f),
            max(abs(step.z)/5.f, abs(step.w)/.1f)));
        // lower/upper were rounded INWARD on the CPU from double seed-relative
        // intervals; an ordinary float clamp cannot escape those exact bounds.
        const float4 proposal = clamp(s.pose+step/ratio, s.lower, s.upper);
        step = proposal-s.pose;
        if (max(abs(step.x), abs(step.y)) < 1e-4f && abs(step.z) < 1e-4f && abs(step.w) < 1e-6f) {
            s.active = s.pending = 0; s.status = 0; continue;
        }
        s.candidate = proposal; s.candidatePacked = resident_pack(proposal);
        if (!all(isfinite(s.candidatePacked))) { s.active = s.pending = 0; s.status = 3; continue; }
        const uint index = atomic_fetch_add_explicit(&candidates, 1u, memory_order_relaxed);
        candidateMap[index] = owner; packed[index] = s.candidatePacked; s.proposed = 1;
        atomic_fetch_add_explicit(&pending, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (lane == 0) {
        const uint n = atomic_load_explicit(&candidates, memory_order_relaxed);
        const uint remains = atomic_load_explicit(&pending, memory_order_relaxed);
        control.candidateCount = n; k.count = n;
        grids[ResidentScoreTiles] = resident_grid(k.tiles, n);
        grids[ResidentScoreFinish] = resident_grid((n+options.finishThreads-1)/options.finishThreads);
        grids[ResidentTrialControl] = resident_grid(remains ? 1u : 0u);
    }
}
