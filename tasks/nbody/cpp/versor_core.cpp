#include "mapping.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <torch/extension.h>
#include <vector>

// Sign matrix (32x32) for Clifford basis products Ea * Eb = sign * E(a^b)
static float S[32][32];
static bool initialized = false;

// Portable bit counting
int popcount_local(int x) {
  int c = 0;
  while (x) {
    x &= (x - 1);
    c++;
  }
  return c;
}

// Initialize the Clifford sign matrix Cl(4,1)
void init_sign_matrix() {
  if (initialized)
    return;
  for (int a = 0; a < 32; a++) {
    for (int b = 0; b < 32; b++) {
      int swaps = 0;
      for (int bit = 0; bit < 5; bit++) {
        if ((b >> bit) & 1) {
          // Count basis vector swaps
          int mask_gt = (~((1 << (bit + 1)) - 1)) & 31;
          swaps += popcount_local(a & mask_gt);
        }
      }
      float comm_sign = (swaps % 2 == 1) ? -1.0f : 1.0f;
      // Metric Sign for Cl(4,1): e4*e4 = -1 (where e4 is the 5th basis vector,
      // index 16)
      float metric_sign = ((a & 16) && (b & 16)) ? -1.0f : 1.0f;
      S[a][b] = comm_sign * metric_sign;
    }
  }
  initialized = true;
}

// Project multivector back onto the unit manifold
inline void manifold_normalization_single(float *x, float eps = 1e-8f) {
  float norm_sq = 0.0f;
  float l2_sum = 0.0f;
  for (int i = 0; i < 32; i++) {
    float val = x[i];
    float sq = val * val;
    norm_sq += sq * S[i][i];
    l2_sum += sq;
  }
  float abs_norm = std::sqrt(std::abs(norm_sq) + eps);
  float l2_norm = std::sqrt(l2_sum) + eps;

  // Stable max
  float denom = abs_norm;
  if (l2_norm > denom)
    denom = l2_norm;
  if (1.0f > denom)
    denom = 1.0f;

  float inv_denom = 1.0f / denom;
  for (int i = 0; i < 32; i++)
    x[i] *= inv_denom;
}

// Standard bit-masked geometric product
inline void geometric_product_single(const float *A, const float *B,
                                     float *Out) {
  for (int i = 0; i < 32; i++) {
    float Ai = A[i];
    if (std::abs(Ai) < 1e-9f)
      continue;
    for (int j = 0; j < 32; j++) {
      Out[i ^ j] += Ai * B[j] * S[i][j];
    }
  }
}

// Complex Matrix Multiplication for 4x4 representation
inline void complex_matmul_4x4(const float *A, const float *B, float *Out) {
  std::fill(Out, Out + 32, 0.0f);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      float re = 0.0f, im = 0.0f;
      for (int k = 0; k < 4; k++) {
        // A[i,k] * B[k,j]
        float are = A[(i * 4 + k) * 2];
        float aim = A[(i * 4 + k) * 2 + 1];
        float bre = B[(k * 4 + j) * 2];
        float bim = B[(k * 4 + j) * 2 + 1];
        re += are * bre - aim * bim;
        im += are * bim + aim * bre;
      }
      Out[(i * 4 + j) * 2] = re;
      Out[(i * 4 + j) * 2 + 1] = im;
    }
  }
}

// Map Geometric Vector to 4x4 Complex Matrix
inline void ga_to_matrix_local(const float *x, float *m) {
  std::fill(m, m + 32, 0.0f);
  for (int i = 0; i < 32; i++) {
    float val = x[i];
    if (std::abs(val) < 1e-9f)
      continue;
    const float *base = M_MAP + i * 32;
    for (int k = 0; k < 32; k++)
      m[k] += val * base[k];
  }
}

// Map 4x4 Complex Matrix back to Geometric Vector
inline void matrix_to_ga_local(const float *m, float *x) {
  for (int i = 0; i < 32; i++) {
    float val = 0.0f;
    const float *base = M_MAP + i * 32;
    for (int k = 0; k < 32; k++)
      val += m[k] * base[k];
    x[i] = val / 4.0f;
  }
}

// Main Parallelized Scan: Recursive Rotor Accumulator
torch::Tensor rra_scan_forward(torch::Tensor input, bool use_matrix) {
  auto input_contig = input.contiguous().cpu();
  if (!initialized)
    init_sign_matrix();

  auto B_count = input_contig.size(0);
  auto S_len = input_contig.size(1);
  auto N = input_contig.size(2);
  auto H = input_contig.size(3);

  auto output = torch::zeros_like(input_contig);
  float *in_ptr = input_contig.data_ptr<float>();
  float *out_ptr = output.data_ptr<float>();

  int64_t total_tasks = B_count * N * H;

  at::parallel_for(0, total_tasks, 1, [&](int64_t start, int64_t end) {
    for (int64_t task_idx = start; task_idx < end; task_idx++) {
      int64_t rem = task_idx;
      int64_t h = rem % H;
      rem /= H;
      int64_t n = rem % N;
      int64_t b = rem / N;

      int64_t batch_stride = S_len * N * H * 32;
      int64_t step_stride = N * H * 32;
      int64_t n_stride = H * 32;
      int64_t h_stride = 32;

      int64_t base_offset = b * batch_stride + n * n_stride + h * h_stride;

      float psi[32] = {0.0f};
      psi[0] = 1.0f; // Identity

      if (use_matrix) {
        // Matrix Turbo Path: Reduces Flops from 1024 to 256
        float m_psi[32];
        ga_to_matrix_local(psi, m_psi);

        for (int t = 0; t < S_len; t++) {
          float *current_in = in_ptr + base_offset + t * step_stride;
          float *current_out = out_ptr + base_offset + t * step_stride;

          float vec_r[32];
          for (int i = 0; i < 32; i++)
            vec_r[i] = current_in[i];
          vec_r[0] += 1.0f; // Delta-rotor mapping
          manifold_normalization_single(vec_r);

          float m_r[32];
          ga_to_matrix_local(vec_r, m_r);

          float m_next_psi[32];
          complex_matmul_4x4(m_r, m_psi, m_next_psi);

          // Normalize in matrix space (Frobenius norm proxy for stability)
          float sum_sq = 0.0f;
          for (int i = 0; i < 32; i++)
            sum_sq += m_next_psi[i] * m_next_psi[i];
          float inv_f_norm = 2.0f / (std::sqrt(sum_sq) + 1e-8f);
          for (int i = 0; i < 32; i++)
            m_psi[i] = m_next_psi[i] * inv_f_norm;

          matrix_to_ga_local(m_psi, current_out);
        }
      } else {
        // Bitmasked Path: Standard Clifford product
        for (int t = 0; t < S_len; t++) {
          int64_t data_offset = base_offset + t * step_stride;
          float *current_in = in_ptr + data_offset;
          float *current_out = out_ptr + data_offset;

          float delta_r[32];
          for (int i = 0; i < 32; i++)
            delta_r[i] = current_in[i];
          delta_r[0] += 1.0f;
          manifold_normalization_single(delta_r);

          float next_psi[32] = {0.0f};
          geometric_product_single(delta_r, psi, next_psi);
          manifold_normalization_single(next_psi);

          for (int i = 0; i < 32; i++) {
            psi[i] = next_psi[i];
            current_out[i] = psi[i];
          }
        }
      }
    }
  });

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rra_scan_forward", &rra_scan_forward,
        "Versor RRA Scan Forward (CPU/Matrix/Bitmasked)");
}
