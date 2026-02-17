// ============================================================
// sign_logic.v - Hardcoded Clifford Sign Logic for Cl(4,1)
// ============================================================
//
// Pure combinational circuit. Given two blade indices (i, j),
// outputs the result blade index (k) and the combined sign.
//
// This replaces the Cayley table entirely.
// Latency: 0 cycles (combinational)
// Area: ~200 gates (trivial silicon cost)
//
// (c) 2026 Versor AI - Open Hardware License (CERN-OHL-P-2.0)
// ============================================================

module sign_logic #(
    parameter N_BASIS = 5,
    parameter BLADE_W = 5
)(
    input  wire [BLADE_W-1:0]  blade_i,
    input  wire [BLADE_W-1:0]  blade_j,
    
    output wire [BLADE_W-1:0]  blade_k,      // Result index
    output wire                sign_bit,      // 0 = positive, 1 = negative
    output wire                has_contraction // 1 if blades share basis vectors
);

    // Result blade = symmetric difference
    assign blade_k = blade_i ^ blade_j;
    
    // Intersection = basis vectors in both blades
    wire [BLADE_W-1:0] intersection;
    assign intersection = blade_i & blade_j;
    assign has_contraction = |intersection;
    
    // --------------------------------------------------------
    // Swap count: For each bit in j, count higher bits in i
    // Unrolled for 5 basis vectors of Cl(4,1)
    // --------------------------------------------------------
    
    // Bit 0 of j set: count bits {1,2,3,4} of i
    wire s00 = blade_j[0] & blade_i[1];
    wire s01 = blade_j[0] & blade_i[2];
    wire s02 = blade_j[0] & blade_i[3];
    wire s03 = blade_j[0] & blade_i[4];
    
    // Bit 1 of j set: count bits {2,3,4} of i
    wire s10 = blade_j[1] & blade_i[2];
    wire s11 = blade_j[1] & blade_i[3];
    wire s12 = blade_j[1] & blade_i[4];
    
    // Bit 2 of j set: count bits {3,4} of i
    wire s20 = blade_j[2] & blade_i[3];
    wire s21 = blade_j[2] & blade_i[4];
    
    // Bit 3 of j set: count bit {4} of i
    wire s30 = blade_j[3] & blade_i[4];
    
    // Total swap parity (only need LSB for sign)
    wire swap_parity = s00 ^ s01 ^ s02 ^ s03 
                     ^ s10 ^ s11 ^ s12
                     ^ s20 ^ s21
                     ^ s30;
    
    // Metric sign: e- (bit 4) squares to -1
    wire metric_neg = intersection[4];
    
    // Combined sign
    assign sign_bit = swap_parity ^ metric_neg;

endmodule
