// ============================================================
// gspa_pim.v - Grade-Sparse Processing-In-Memory (GSPA)
// ============================================================
//
// Models a bank of memory with integrated Clifford arithmetic
// units (GACUs). This architecture performs "Near-Data Processing"
// to eliminate memory bandwidth bottlenecks.
//
// Key Feature:
//   - Each memory row (serving one multivector blade) has a
//     dedicated "Score Unit" that computes partial dot products.
//   - Only the final accumulated scalar leaves the memory bank.
//
// (c) 2026 Versor AI - Open Hardware License (CERN-OHL-P-2.0)
// ============================================================

module gspa_pim #(
    parameter N_BANKS = 32, // One bank per basis blade (Cl4,1 dim=32)
    parameter DATA_W  = 32
)(
    input  wire                clk,
    input  wire                rst_n,
    
    // Command Interface (Broadcast to all banks)
    input  wire [31:0]         query_broadcast, // Q[i] broadcast to all banks
    input  wire [4:0]          query_blade_idx, // Index i
    input  wire                cmd_score,       // Enable scoring
    
    // Result Interface (Wire-OR or Reduction Tree)
    output wire [31:0]         scalar_result
);

    // --------------------------------------------------------
    // PIM Bank Array
    // --------------------------------------------------------
    // Each bank stores V[j] for a specific blade j.
    // When Q[i] arrives, bank j computes:
    //   term = Q[i] * V[j] * sign(i,j)
    // AND contributes it to the sum ONLY IF (i ^ j == 0).
    
    wire [31:0] partial_scores [N_BANKS-1:0];
    
    genvar j;
    generate
        for (j=0; j<N_BANKS; j=j+1) begin : BANKS
            gspa_bank #(
                .BANK_ID(j)
            ) bank_inst (
                .clk(clk),
                .rst_n(rst_n),
                .query_val(query_broadcast),
                .query_idx(query_blade_idx),
                .enable(cmd_score),
                .partial_out(partial_scores[j])
            );
        end
    endgenerate

    // --------------------------------------------------------
    // Reduction Tree (Adder Tree)
    // --------------------------------------------------------
    // Sum all partial scores to get final scalar result.
    // In real hardware, this is a distinct reduction network.
    // Here we model it behaviorally.
    
    reg [31:0] tree_sum;
    integer k;
    
    always @(*) begin
        tree_sum = 0;
        for (k=0; k<N_BANKS; k=k+1) begin
            tree_sum = tree_sum + partial_scores[k];
        end
    end
    
    assign scalar_result = tree_sum;

endmodule

// ============================================================
// Internal PIM Bank Module
// ============================================================
module gspa_bank #(
    parameter BANK_ID = 0 // Represents blade index j
)(
    input wire clk, rst_n,
    input wire [31:0] query_val, // Q[i]
    input wire [4:0]  query_idx, // i
    input wire        enable,
    
    output reg [31:0] partial_out
);

    // Local Storage (modeling the memory row)
    reg [31:0] stored_value; // V[j]
    
    // Sign Logic
    // Compute sign(i, j) where j is fixed (BANK_ID)
    wire [4:0] k_blade;
    wire sign_bit;
    wire has_contraction; // Metric
    
    sign_logic #(5, 5) u_sign (
        .blade_i(query_idx),
        .blade_j(BANK_ID[4:0]),
        .blade_k(k_blade),
        .sign_bit(sign_bit),
        .has_contraction(has_contraction) // Not used for scalar check?
    );

    // Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stored_value <= 0; // In reality, this is RAM content
            partial_out <= 0;
        end else if (enable) begin
            // Core GSPA Rule: Only compute if result is Scalar (Grade 0)
            // i.e., blade_k == 0 (which implies i == j in Euclidean,
            // but in Cl(4,1) with null basis, scalar part logic is complex.
            // For standard inner product, scalar part exists iff i == j?
            // Yes, basis blades are orthogonal. E_i * E_j = scalar iff i=j.
            
            // Wait, what about E_o * E_inf = -1?
            // E_o (index?) and E_inf (index?) are distinct indices.
            // If i=E_o and j=E_inf, i^j is NOT 0 (it's E_o^E_inf bivector? No).
            // Clifford product: e_o . e_inf = -1.
            // So dot product has non-zero scalar part even if i != j.
            // This is handled by 'common/sign_logic.v' ?
            // sign_logic outputs k_blade = i ^ j.
            // If i != j, then k_blade != 0.
            // So sign_logic says result is a bivector.
            // BUT inner product <A B>_0 selects the grade-0 part of the result.
            // Is the result of e_o * e_inf purely scalar?
            // e_o e_inf = -1 - e_o ^ e_inf.
            // It has a scalar part AND a bivector part.
            // So standard "i==j" check is insufficient for Conformal GA?
            
            // Actually, for pure retrieval of scalar score <Q K>_0:
            // We want the scalar term.
            // Most terms Q[i]*K[j] produce result of grade |grade(i)-grade(j)| or higher.
            // Only terms that can produce scalars are those where i and j are "inverse" or "conjugate".
            // In standard GA, basis blades square to scalars. So i==j is the main source.
            // The only exception is null basis e_o, e_inf.
            // If our basis set is {1, e1, e2, e3, e+, e-}, they are orthogonal.
            // e+^2 = 1, e-^2 = -1.
            // And i==j covers all scalar-producing pairs.
            //
            // So yes, checking k_blade == 0 (i^j == 0 => i == j) is correct
            // provided we use the orthogonal basis {e+, e-} and not {eo, einf}.
            // The paper mentions "lattice" basis.
            // Assuming {e+, e-} basis usage here for hardware simplicity (diagonal metric).
            
            if (k_blade == 0) begin
                partial_out <= sign_bit ? -(query_val * stored_value) : (query_val * stored_value);
            end else begin
                partial_out <= 0;
            end
        end
    end

endmodule
