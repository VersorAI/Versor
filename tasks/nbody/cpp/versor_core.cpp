#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

// Sign matrix (32x32)
static float S[32][32];
static bool initialized = false;

int popcount_local(int x) {
    // Portable popcount
    int c = 0;
    while(x) {
        x &= (x-1);
        c++;
    }
    return c;
}

void init_sign_matrix() {
    if (initialized) return;
    for (int a = 0; a < 32; a++) {
        for (int b = 0; b < 32; b++) {
             // Calculate sign(a, b) for geometric product Ea * Eb
             int swaps = 0;
             for (int bit = 0; bit < 5; bit++) {
                 if ((b >> bit) & 1) {
                     // Check bits in 'a' that are more significant than 'bit'
                     // These are the ones 'b's bit jumps over.
                     // mask of bits > bit: (~((1 << (bit + 1)) - 1)) & 31
                     int mask_gt = (~((1 << (bit + 1)) - 1)) & 31;
                     swaps += popcount_local(a & mask_gt);
                 }
             }
             float comm_sign = (swaps % 2 == 1) ? -1.0f : 1.0f;
             // Metric Sign (e4*e4 = -1). e4 corresponds to 16 (1<<4).
             float metric_sign = ((a & 16) && (b & 16)) ? -1.0f : 1.0f;
             S[a][b] = comm_sign * metric_sign;
        }
    }
    initialized = true;
}

// Manifold Normalization operating on a single 32-float array
inline void manifold_normalization_single(float* x, float eps=1e-6) {
    float norm_sq = 0.0f;
    float l2_sum = 0.0f;
    for (int i = 0; i < 32; i++) {
        float val = x[i];
        float sq = val * val;
        // Metric signature is S[i][i]
        norm_sq += sq * S[i][i];
        l2_sum += sq;
    }
    
    float abs_norm = std::sqrt(std::abs(norm_sq) + eps);
    float l2_norm = std::sqrt(l2_sum) + eps;
    float denom = std::max(abs_norm, l2_norm);
    denom = std::max(denom, 1.0f);
    
    float inv_denom = 1.0f / denom;
    for (int i = 0; i < 32; i++) {
        x[i] *= inv_denom;
    }
}

// Geometric Product: Out = A * B
inline void geometric_product_single(const float* A, const float* B, float* Out) {
    // Assume Out is zeroed by caller
    // Unrolling or optimizing this loop would be good, but this is already C++ speed.
    for (int i = 0; i < 32; i++) {
        float Ai = A[i];
        if (std::abs(Ai) < 1e-9f) continue; 
        for (int j = 0; j < 32; j++) {
            // k = i ^ j
            // Out[k] += Ai * Bj * S[i][j]
            Out[i ^ j] += Ai * B[j] * S[i][j];
        }
    }
}

torch::Tensor rra_scan_forward(torch::Tensor input) {
    // input: (B, S, N, H, 32)
    // Ensures CPU floats
    auto input_contig = input.contiguous().cpu();
    
    if (!initialized) init_sign_matrix();
    
    auto B = input_contig.size(0);
    auto S_len = input_contig.size(1);
    auto N = input_contig.size(2);
    auto H = input_contig.size(3);
    
    auto output = torch::zeros_like(input_contig);
    
    float* in_ptr = input_contig.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    
    // Total parallel tasks: B * N * H
    int64_t total_tasks = B * N * H;
    
    // Provide a parallel loop
    at::parallel_for(0, total_tasks, 1, [&](int64_t start, int64_t end) {
        for (int64_t task_idx = start; task_idx < end; task_idx++) {
            // Decode task_idx -> b, n, h
            // index layout is (b, s, n, h, 32) ? No.
            // Ptr arithmetic is easier if we just track strides.
            // Input stride logic:
            // dim 0: S*N*H*32
            // dim 1: N*H*32  <-- S dimension iterates inside
            // dim 2: H*32
            // dim 3: 32
            // dim 4: 1
            
            // task_idx covers (b, n, h).
            // h = task_idx % H
            // n = (task_idx / H) % N
            // b = (task_idx / (H * N))
            
            int64_t rem = task_idx;
            int64_t h = rem % H;
            rem /= H;
            int64_t n = rem % N;
            int64_t b = rem / N;
            
            // Base pointer for this sequence
            // input[b, :, n, h, :]
            // Offset = b * (S*N*H*32) + n * (H*32) + h * (32)
            // Wait, input is (B, S, N, H, 32).
            // So S is the second dimension.
            // Stride for B: S*N*H*32
            // Stride for S: N*H*32
            // Stride for N: H*32
            // Stride for H: 32
            
            int64_t batch_stride = S_len * N * H * 32;
            int64_t step_stride = N * H * 32;
            int64_t n_stride = H * 32;
            int64_t h_stride = 32;
            
            int64_t base_offset = b * batch_stride + n * n_stride + h * h_stride;
            
            // Local state
            float psi[32] = {0};
            psi[0] = 1.0f;
            
            float delta_r[32];
            float next_psi[32];
            
            for (int t = 0; t < S_len; t++) {
                int64_t data_offset = base_offset + t * step_stride;
                float* current_in = in_ptr + data_offset;
                float* current_out = out_ptr + data_offset;
                
                // Copy u_t
                for(int i=0; i<32; i++) delta_r[i] = current_in[i];
                
                // delta_r[0] += 1.0
                delta_r[0] += 1.0f;
                
                // Normalize
                manifold_normalization_single(delta_r);
                
                // Psi update
                std::fill(next_psi, next_psi + 32, 0.0f);
                geometric_product_single(delta_r, psi, next_psi);
                
                // Normalize Psi
                manifold_normalization_single(next_psi);
                
                // Store and Update
                for(int i=0; i<32; i++) {
                    psi[i] = next_psi[i];
                    current_out[i] = psi[i];
                }
            }
        }
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rra_scan_forward", &rra_scan_forward, "Versor RRA Scan Forward (CPU)");
}
