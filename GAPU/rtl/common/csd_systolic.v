// ============================================================
// csd_systolic.v - Clifford Systolic Dataflow Array (CSD)
// ============================================================
//
// BEHAVIORAL / LATENCY MODEL ONLY - NOT SYNTHESIZABLE
// This models a 32x32 array of Processing Elements (PEs) designed to compute
// Geometric Algebra operations with maximum data reuse.
// Note: FP32 multiplication and addition are modeled behaviorally.
//
// (c) 2026 Versor AI - Open Hardware License (CERN-OHL-P-2.0)
// ============================================================

module csd_systolic #(
    parameter N_BASIS = 5,
    parameter GA_DIM  = 32,
    parameter BLADE_W = 5
)(
    input  wire                clk,
    input  wire                rst_n,
    
    // Control
    input  wire                load_weights, // Load Value V into array
    input  wire                run_compute,  // Pulse Query Q
    output wire                busy,
    output wire                valid_out,
    
    // Data Inputs (Systolic Feed)
    // In a real chip, these come from on-chip SRAM banks.
    // We model a single input port that feeds the array edge.
    input  wire [31:0]         q_in_data,    // Q element entering row 0?
    input  wire [31:0]         v_in_data,    // V element for weight loading
    
    // Result Output
    output wire [31:0]         scalar_score_out
);

    // --------------------------------------------------------
    // Systolic Logic
    // --------------------------------------------------------
    // The array is 32x32. 
    // Rows = Query basis index i (0..31)
    // Cols = Value basis index j (0..31)
    
    // Wires connecting PEs
    wire [31:0] q_horizontal [31:0][32:0]; // Q flows West -> East
    wire [31:0] partial_sum  [32:0][31:0]; // Sum flows North -> South?
    // Actually, for dot product <Q * V>0, we sum all terms where i == j
    // or rather where i ^ j == 0 (scalar).
    // The CSD computes the full product or just scalar?
    // The paper says "GSPA" does sparse. CSD does dense.
    // If dense, we need 32 accumulators flowing out?
    
    // For simplicity of this structural model, let's implement the SCALAR SCORE logic
    // which is the bottleneck for attention.
    // scalar = sum(Q[i] * V[j] * sign(i,j) * metric(i,j))
    // This looks like a dot product but with sign terms.
    
    genvar r, c;
    generate
        for (r=0; r<GA_DIM; r=r+1) begin : ROWS
            for (c=0; c<GA_DIM; c=c+1) begin : COLS
            
                // Instantiate PE
                csd_pe #(
                    .ROW_IDX(r), 
                    .COL_IDX(c)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .load_weight(load_weights && (r==c)), // Diagonal loading? Simplified
                    .weight_in(v_in_data),                // Shared bus for loading (model)
                    
                    .q_in (q_horizontal[r][c]),
                    .q_out(q_horizontal[r][c+1]),
                    
                    .sum_in (partial_sum[r][c]),
                    .sum_out(partial_sum[r+1][c])
                );
            end
        end
    endgenerate

    // --------------------------------------------------------
    // Feed Logic (Simulation Model)
    // --------------------------------------------------------
    // Connect q_in_data to row inputs...
    // In a real systolic array, data is skewed.
    // Here we just define the connectivity.
    
    assign scalar_score_out = partial_sum[GA_DIM][GA_DIM-1]; // Result at bottom right?
    
endmodule

// ============================================================
// Internal Processing Element (PE)
// ============================================================
module csd_pe #(
    parameter ROW_IDX = 0,
    parameter COL_IDX = 0
)(
    input wire clk, rst_n,
    input wire load_weight,
    input wire [31:0] weight_in,
    
    input wire [31:0] q_in,
    output reg [31:0] q_out,
    
    input wire [31:0] sum_in,
    output reg [31:0] sum_out
);

    reg [31:0] stored_weight; // V[col]
    
    // Sign logic for this specific (row, col) pair
    // Calculate sign(row, col) at elaboration time or hardcoded
    // For RTL model, we can use the sign_logic module!
    
    wire [4:0] k_blade;
    wire sign_bit;
    wire has_contraction;
    
    sign_logic #(5, 5) u_sign (
        .blade_i(ROW_IDX[4:0]),
        .blade_j(COL_IDX[4:0]),
        .blade_k(k_blade),
        .sign_bit(sign_bit),
        .has_contraction(has_contraction)
    );
    
    // Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stored_weight <= 0;
            q_out <= 0;
            sum_out <= 0;
        end else begin
            // Weight loading
            if (load_weight) stored_weight <= weight_in;
            
            // Pass Q
            q_out <= q_in;
            
            // Compute MAC
            // term = q_in * stored_weight * sign
            // Only add to sum if result is scalar (k_blade == 0) ?
            // Or accumulate all?
            // "Scalar Score" implies we only want the scalar part.
            // k_blade == 0 means result is scalar.
            
            if (k_blade == 0) begin
                // FP32 MAC (Behavioral)
                // If sign_bit is 1, subtract.
               sum_out <= sum_in + (sign_bit ? -(q_in * stored_weight) : (q_in * stored_weight));
            end else begin
                // Pass sum unchanged
               sum_out <= sum_in;
            end
        end
    end

endmodule
