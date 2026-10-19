// SPDX-License-Identifier: BSD-2-Clause
// Appended after pose_ncc.metal and pose_resident.metal for the opt-in,
// grayscale fused solver. The host enforces one 256-thread group per candidate
// and a nonempty template of at most 16,384 pixels. Groups never communicate.
// Pixel order, tile size and reduction trees match the separate kernels. Only
// the first 128 lanes evaluate derivatives; every lane executes every barrier.

inline void fused_score(device const float* source, device const float* templ,
    float4 p, constant Config& k, uint lane,
    threadgroup DerivativePartial* scratch, threadgroup float& result)
{
    // Scores are grayscale, so four scalar arrays fit inside the derivative
    // scratch. All accesses to the previous interpretation finish at a group
    // barrier before this function is entered or returns.
    threadgroup float* means = reinterpret_cast<threadgroup float*>(scratch);
    threadgroup float* variances = means+256;
    threadgroup float* products = means+512;
    threadgroup uint* counts = reinterpret_cast<threadgroup uint*>(means+768);
    float totalMean = 0, totalVariance = 0, totalProduct = 0;
    uint totalCount = 0;
    for (uint tile = 0; tile < k.tiles; ++tile) {
        float mean = 0, m2 = 0, product = 0;
        uint count = 0;
        for (uint pixel = tile*1024+lane;
            pixel < min((tile+1)*1024, k.tw*k.th); pixel += 256) {
            const float u = float(pixel%k.tw)-float(k.tw)*.5f;
            const float v = float(k.th)*.5f-float(pixel/k.tw);
            const float sx = float(k.sw)*.5f+p.x+p.z*u-p.w*v;
            const float sy = float(k.sh)*.5f-p.y-(p.w*u+p.z*v);
            const int ix = int(floor(sx)), iy = int(floor(sy));
            const float dx = sx-float(ix), dy = sy-float(iy);
            const float a = load_pixel(source, ix, iy, 0, k);
            const float b = load_pixel(source, ix+1, iy, 0, k);
            const float e = load_pixel(source, ix, iy+1, 0, k);
            const float f = load_pixel(source, ix+1, iy+1, 0, k);
            const float sample = (a+(b-a)*dx)*(1-dy)+(e+(f-e)*dx)*dy;
            product += sample*templ[pixel];
            ++count;
            const float delta = sample-mean;
            mean += delta/float(count);
            m2 += delta*(sample-mean);
        }
        means[lane] = mean; variances[lane] = m2;
        products[lane] = product; counts[lane] = count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = 128; step > 0; step >>= 1) {
            if (lane < step) {
                const uint a = counts[lane], b = counts[lane+step];
                const float delta = means[lane+step]-means[lane];
                if (a+b > 0) {
                    means[lane] += delta*(float(b)/float(a+b));
                    variances[lane] += variances[lane+step]
                        +delta*delta*(float(a)*float(b)/float(a+b));
                }
                products[lane] += products[lane+step]; counts[lane] = a+b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            const float delta = means[0]-totalMean;
            const uint total = totalCount+counts[0];
            totalMean += delta*(float(counts[0])/float(total));
            totalVariance += variances[0]
                +delta*delta*(float(totalCount)*float(counts[0])/float(total));
            totalCount = total; totalProduct += products[0];
        }
        // The next tile must not overwrite lane zero's unread partial.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        const float n = float(k.tw*k.th), energy = max(0.f, totalVariance);
        const float numerator = totalProduct-totalMean*k.templateSum.x;
        result = energy/n <= 1e-12f || k.templateEnergy/n <= 1e-12f ? -2.f :
            clamp(numerator/sqrt(energy*k.templateEnergy), -1.f, 1.f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void fused_derivatives(device const float* source, device const float* templ,
    float4 p, constant Config& k, uint lane,
    threadgroup DerivativePartial* scratch, threadgroup DerivativePartial& total)
{
    if (lane == 0) total = {};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tile = 0; tile < k.tiles; ++tile) {
        DerivativePartial state = {};
        if (lane < 128) {
            for (uint pixel = tile*1024+lane;
                pixel < min((tile+1)*1024, k.tw*k.th); pixel += 128) {
                const float u = float(pixel%k.tw)-float(k.tw)*.5f;
                const float v = float(k.th)*.5f-float(pixel/k.tw);
                const float rx = p.z*u-p.w*v, ry = p.w*u+p.z*v;
                const float sx = float(k.sw)*.5f+p.x+p.z*u-p.w*v;
                const float sy = float(k.sh)*.5f-p.y-(p.w*u+p.z*v);
                const int ix = int(floor(sx)), iy = int(floor(sy));
                const float dx = sx-float(ix), dy = sy-float(iy);
                ++state.count;
                const float a = load_pixel(source, ix, iy, 0, k);
                const float b = load_pixel(source, ix+1, iy, 0, k);
                const float e = load_pixel(source, ix, iy+1, 0, k);
                const float f = load_pixel(source, ix+1, iy+1, 0, k);
                const float sample = (a+(b-a)*dx)*(1-dy)+(e+(f-e)*dx)*dy;
                const float gx = (b-a)*(1-dy)+(f-e)*dy;
                const float gy = (e-a)*(1-dx)+(f-b)*dx;
                float values[5] = {sample, gx, -gy, -gx*ry-gy*rx, gx*rx-gy*ry};
                if (!(k.unused & 1u)) values[1] = values[2] = 0;
                if (!(k.unused & 2u)) values[3] = 0;
                if (!(k.unused & 4u)) values[4] = 0;
                const float t = templ[pixel];
                float delta[5];
                for (uint i = 0; i < 5; ++i) {
                    delta[i] = values[i]-state.mean[0][i];
                    state.mean[0][i] += delta[i]/float(state.count);
                    state.product[i] += values[i]*t;
                }
                uint entry = 0;
                for (uint i = 0; i < 5; ++i) for (uint j = i; j < 5; ++j)
                    state.covariance[entry++] += delta[i]*(values[j]-state.mean[0][j]);
            }
        }
        if (lane >= 64 && lane < 128) scratch[lane-64] = state;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 64) {
            const DerivativePartial right = scratch[lane];
            merge_derivatives(state, right, 1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 64) scratch[lane] = state;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = 32; step > 0; step >>= 1) {
            if (lane < step) {
                DerivativePartial left = scratch[lane], right = scratch[lane+step];
                merge_derivatives(left, right, 1);
                scratch[lane] = left;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            DerivativePartial accumulated = total, next = scratch[0];
            merge_derivatives(accumulated, next, 1);
            total = accumulated;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Address-space-specific versions of the resident control helpers. Keeping
// the old helpers unchanged preserves the separate-kernel reference path.
inline bool fused_assemble(thread const DerivativePartial& p,
    threadgroup ResidentState& s, constant Config& k)
{
    const float n = float(k.tw*k.th), energy = max(0.f, p.covariance[0]);
    if (!isfinite(energy) || energy/n <= 1e-12f || k.templateEnergy/n <= 1e-12f) return false;
    float product = p.product[0], jx[4], jt[4];
    for (uint i = 0; i < 4; ++i) {
        jx[i] = p.covariance[resident_covariance(0, i+1)];
        jt[i] = p.product[i+1];
    }
    product -= p.mean[0][0]*k.templateSum[0];
    for (uint i = 0; i < 4; ++i) jt[i] -= p.mean[0][i+1]*k.templateSum[0];
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

inline bool fused_solve(threadgroup const ResidentState& s, thread float4& step)
{
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

kernel void pose_refine_fused(device const float* source [[buffer(0)]],
    device const float* templ [[buffer(1)]], device ResidentState* states [[buffer(2)]],
    constant Config& k [[buffer(3)]], constant uint& maxIterations [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]])
{
    if (group >= k.count) return;
    // 9,216 + 144 + 208 + 4 bytes, rounded for alignment: below 10 KiB.
    threadgroup DerivativePartial scratch[64], derivative;
    threadgroup ResidentState s;
    threadgroup float trialScore;
    if (lane == 0) s = states[group];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fused_score(source, templ, s.packed, k, lane, scratch, s.score);
    if (lane == 0) {
        s.damping = 1e-3f; s.iterations = 0; s.evaluations = 1;
        s.pending = s.proposed = 0;
        s.status = !isfinite(s.score) ? 3u : (s.score == -2.f ? 2u : (s.score == 1.f ? 0u : 1u));
        s.active = s.status == 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint iteration = 0; iteration < maxIterations; ++iteration) {
        const bool runIteration = s.active != 0;
        // Latch shared control before lane zero is allowed to change it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!runIteration) break;
        fused_derivatives(source, templ, s.packed, k, lane, scratch, derivative);
        if (lane == 0) {
            const DerivativePartial value = derivative;
            ++s.iterations; ++s.evaluations;
            if (!fused_assemble(value, s, k)) {
                s.active = s.pending = 0; s.status = 2;
            } else s.pending = 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint trial = 0; trial < 10; ++trial) {
            const bool runTrial = s.active && s.pending;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!runTrial) break;
            if (lane == 0) {
                s.proposed = 0;
                float4 step;
                if (!fused_solve(s, step)) {
                    s.damping *= 10.f;
                } else {
                    step.z /= 0.01745329251994329577f;
                    const float ratio = max(1.f, max(max(abs(step.x)/2.f, abs(step.y)/2.f),
                        max(abs(step.z)/5.f, abs(step.w)/.1f)));
                    // Bounds were rounded inward from the seed on the host.
                    const float4 proposal = clamp(s.pose+step/ratio, s.lower, s.upper);
                    step = proposal-s.pose;
                    if (max(abs(step.x), abs(step.y)) < 1e-4f && abs(step.z) < 1e-4f
                        && abs(step.w) < 1e-6f) {
                        s.active = s.pending = 0; s.status = 0;
                    } else {
                        s.candidate = proposal; s.candidatePacked = resident_pack(proposal);
                        if (!all(isfinite(s.candidatePacked))) {
                            s.active = s.pending = 0; s.status = 3;
                        } else s.proposed = 1;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // The shared predicate was published by lane zero; all 256 lanes
            // either execute the entire scorer (including barriers), or skip it.
            if (s.proposed) {
                fused_score(source, templ, s.candidatePacked, k, lane, scratch, trialScore);
                if (lane == 0) {
                    ++s.evaluations;
                    if (isfinite(trialScore) && trialScore > s.score+2.384185791015625e-7f) {
                        s.pose = s.candidate; s.packed = s.candidatePacked; s.score = trialScore;
                        s.damping = max(1e-8f, s.damping*.3f); s.pending = 0;
                    } else s.damping *= 10.f;
                    s.proposed = 0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0 && s.active && s.pending) {
            s.active = s.pending = 0; s.status = 3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) states[group] = s;
}
