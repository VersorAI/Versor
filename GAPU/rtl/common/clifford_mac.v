// ============================================================
// clifford_mac.v - Clifford Multiply-Accumulate Unit for Cl(4,1)
// ============================================================
// 
// This is the fundamental building block of ALL GAPU variants.
// It computes one term of the Geometric Product:
//   C[k] += sign(i,j) * metric(i,j) * A[i] * B[j]
//
// where k = i XOR j, and sign/metric are computed from bit logic.
//
// Pipeline: 2 stages
//   Stage 1: Sign + Index computation (combinational logic)
//   Stage 2: FP32 multiply-accumulate
//
// (c) 2026 Versor AI - Open Hardware License (CERN-OHL-P-2.0)
// ============================================================

module clifford_mac #(
    parameter N_BASIS = 5,         // Number of basis vectors (Cl(4,1) -> 5)
    parameter GA_DIM  = 32,        // 2^N_BASIS = 32 blades
    parameter BLADE_W = 5          // log2(GA_DIM) = width of blade index
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                valid_in,
    
    // Blade indices
    input  wire [BLADE_W-1:0]  blade_i,      // Index of A's blade
    input  wire [BLADE_W-1:0]  blade_j,      // Index of B's blade
    
    // FP32 coefficients
    input  wire [31:0]         coeff_a,       // A[i] in IEEE-754 FP32
    input  wire [31:0]         coeff_b,       // B[j] in IEEE-754 FP32
    
    // Accumulator input (for chaining)
    input  wire [31:0]         acc_in,
    
    // Outputs
    output reg  [BLADE_W-1:0]  blade_k,       // Result blade index = i XOR j
    output reg  [31:0]         acc_out,        // Updated accumulator
    output reg                 valid_out
);

    // --------------------------------------------------------
    // Stage 1: Combinational Sign & Index Logic
    // --------------------------------------------------------
    // This is the KEY innovation that replaces Cayley table lookups.
    // Sign = (-1)^(number_of_swaps) * metric_sign
    
    wire [BLADE_W-1:0] target_blade;
    wire               geo_sign;      // Geometric (anti-commutation) sign
    wire               metric_sign;   // Metric contraction sign
    wire               total_sign;    // Combined sign bit
    
    // Target blade: XOR of input blades
    assign target_blade = blade_i ^ blade_j;
    
    // Geometric sign: count swaps needed to canonicalize
    // For each set bit in blade_j, count how many higher bits are set in blade_i
    wire [2:0] swap_count;  // Max swaps for 5-bit system = 10, need 4 bits
    
    // Unrolled swap counting for Cl(4,1) - 5 basis vectors
    // For bit k of blade_j: count popcount(blade_i >> (k+1))
    wire [2:0] swaps_bit0, swaps_bit1, swaps_bit2, swaps_bit3;
    
    // Bit 0 of j: count bits 1,2,3,4 of i
    assign swaps_bit0 = blade_j[0] ? (blade_i[1] + blade_i[2] + blade_i[3] + blade_i[4]) : 3'd0;
    
    // Bit 1 of j: count bits 2,3,4 of i
    assign swaps_bit1 = blade_j[1] ? (blade_i[2] + blade_i[3] + blade_i[4]) : 3'd0;
    
    // Bit 2 of j: count bits 3,4 of i
    assign swaps_bit2 = blade_j[2] ? (blade_i[3] + blade_i[4]) : 3'd0;
    
    // Bit 3 of j: count bit 4 of i
    assign swaps_bit3 = blade_j[3] ? blade_i[4] : 3'd0;
    
    // Bit 4 of j: nothing higher -> 0 swaps
    
    wire [3:0] total_swaps;
    assign total_swaps = swaps_bit0 + swaps_bit1 + swaps_bit2 + swaps_bit3;
    
    // Sign = (-1)^total_swaps
    assign geo_sign = total_swaps[0];  // LSB determines parity
    
    // Metric contraction: occurs when a basis vector appears in BOTH blades.
    // In Cl(4,1), only e- (bit 4) squares to -1. All others square to +1.
    wire [BLADE_W-1:0] intersection;
    assign intersection = blade_i & blade_j;
    
    // Metric sign flips only if e- (bit 4) is in the intersection
    assign metric_sign = intersection[4];
    
    // Total sign: XOR of geometric sign and metric sign
    assign total_sign = geo_sign ^ metric_sign;
    
    // --------------------------------------------------------
    // Stage 2: Pipeline Register + FP32 MAC
    // --------------------------------------------------------
    // In a real ASIC, this would use a hardened FP32 multiplier.
    // Here we model the pipeline behavior.
    
    reg [BLADE_W-1:0] blade_k_s1;
    reg               sign_s1;
    reg [31:0]        coeff_a_s1, coeff_b_s1, acc_s1;
    reg               valid_s1;
    
    // Stage 1 -> Stage 2 pipeline register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            blade_k_s1 <= 0;
            sign_s1    <= 0;
            coeff_a_s1 <= 0;
            coeff_b_s1 <= 0;
            acc_s1     <= 0;
            valid_s1   <= 0;
        end else begin
            blade_k_s1 <= target_blade;
            sign_s1    <= total_sign;
            coeff_a_s1 <= coeff_a;
            coeff_b_s1 <= coeff_b;
            acc_s1     <= acc_in;
            valid_s1   <= valid_in;
        end
    end
    
    // FP32 multiply (sign-adjusted)
    // In real hardware: use a dedicated FP32 multiplier IP
    // The sign bit of the product is XORed with the algebra sign
    wire [31:0] product;
    wire [31:0] signed_product;
    
    // FP32 multiply: product = coeff_a * coeff_b
    // Sign adjustment: flip MSB (sign bit) if total_sign is 1
    assign product = {coeff_a_s1[31] ^ coeff_b_s1[31], 
                      coeff_a_s1[30:23] + coeff_b_s1[30:23] - 8'd127,
                      coeff_a_s1[22:0]}; // Simplified - real impl uses full mantissa mult
    
    assign signed_product = {product[31] ^ sign_s1, product[30:0]};
    
    // Output pipeline register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            blade_k   <= 0;
            acc_out   <= 0;
            valid_out <= 0;
        end else begin
            blade_k   <= blade_k_s1;
            // acc_out = acc_in + signed_product (FP32 add - simplified)
            acc_out   <= valid_s1 ? signed_product : 32'd0; // Placeholder for FP32 adder
            valid_out <= valid_s1;
        end
    end

endmodule
