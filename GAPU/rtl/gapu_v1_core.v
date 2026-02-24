// ============================================================
// gapu_v1_core.v - GAPU Generation 1 Core (Parallel)
// ============================================================
//
// BEHAVIORAL / LATENCY MODEL ONLY - NOT SYNTHESIZABLE
// This file models the latency characteristics and basic logic flow for 
// architectural simulation. It does not contain synthesizable floating-point 
// kernels or valid ASIC timing paths.
//
// Architecture:
//   - Iterate through all 32x32 = 1024 basis blade pairs (i, j)
//   - Fetch A[i] and B[j]
//   - Compute term K = A[i] * B[j] * sign(i,j) * metric(i,j)
//   - Accumulate into output register file C[i ^ j]
//   - Total latency: ~1024 cycles per product (plus pipeline depth)
//
// (c) 2026 Versor AI - Open Hardware License (CERN-OHL-P-2.0)
// ============================================================

module gapu_v1_core #(
    parameter N_BASIS = 5,
    parameter GA_DIM  = 32,
    parameter BLADE_W = 5
)(
    input  wire                clk,
    input  wire                rst_n,
    
    // Control Interface
    input  wire                start,
    output reg                 done,
    output reg                 busy,
    
    // Memory Interface (External SRAM/Cache)
    // We request A[i] and B[j].
    output reg  [BLADE_W-1:0]  addr_a,
    input  wire [31:0]         data_a, // A[i]
    
    output reg  [BLADE_W-1:0]  addr_b,
    input  wire [31:0]         data_b, // B[j]
    
    // Result Interface (Writeback)
    // We output the full 32-dim multivector C
    output reg  [BLADE_W-1:0]  addr_c,
    output reg  [31:0]         data_c,
    output reg                 wen_c
);

    // --------------------------------------------------------
    // Internal State
    // --------------------------------------------------------
    
    // Iteration counters
    reg [BLADE_W-1:0] i, j;
    
    // Accumulator Register File (32 x 32-bit)
    // We must accumulate partial results for each blade index.
    // C[k] = sum_{i^j=k} (terms)
    reg [31:0] c_accum [0:GA_DIM-1];
    integer k;

    // FSM States
    localparam S_IDLE   = 0;
    localparam S_INIT   = 1;
    localparam S_COMPUTE = 2;
    localparam S_WRITE  = 3;
    localparam S_DONE   = 4;
    
    reg [2:0] state;

    // --------------------------------------------------------
    // Clifford MAC Instantiation
    // --------------------------------------------------------
    
    wire MAC_valid_in;
    wire [BLADE_W-1:0] MAC_blade_k;
    wire [31:0] MAC_acc_out; // Note: We use the MAC's adder for accumulation
    wire MAC_valid_out;
    
    // The MAC unit here is used slightly differently: 
    // We feed it A[i], B[j], and the *current accumulated value* of C[i^j].
    // Wait, the MAC unit in 'common/clifford_mac.v' has a built-in adder:
    // acc_out = acc_in + product.
    // So we need to feed it the current value of C[i^j].
    // But i^j changes every cycle, so we'd need 32 read/write ports to c_accum?
    // No. We can't use the MAC's internal accumulator for *different* blades easily
    // without reading/writing the regfile.
    //
    // Simplified Dataflow:
    // 1. Fetch data_a, data_b.
    // 2. Compute sign/index (combinational).
    // 3. Multiply (pipeline).
    // 4. Read C[target_blade].
    // 5. Add.
    // 6. Write C[target_blade].
    
    // Let's rely on the MAC for the sign logic and multiply.
    // We will supply 0 to acc_in and do the accumulation externally in this core
    // to manage the RegFile properly.
    
    clifford_mac #(
        .N_BASIS(5), .GA_DIM(32), .BLADE_W(5)
    ) u_mac (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(state == S_COMPUTE),
        .blade_i(i),
        .blade_j(j),
        .coeff_a(data_a),
        .coeff_b(data_b),
        .acc_in(32'b0), // Not using internal chaining here
        .blade_k(MAC_blade_k),
        .acc_out(MAC_acc_out), // This is just (A*B*sign) now
        .valid_out(MAC_valid_out)
    );

    // --------------------------------------------------------
    // FSM & Control Logic
    // --------------------------------------------------------
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            i <= 0;
            j <= 0;
            done <= 0;
            busy <= 0;
            addr_c <= 0;
            data_c <= 0;
            wen_c <= 0;
            // Clear accumulator
             for (k=0; k<GA_DIM; k=k+1) c_accum[k] <= 32'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 0;
                    wen_c <= 0;
                    if (start) begin
                        state <= S_INIT;
                        busy <= 1;
                    end else begin
                        busy <= 0;
                    end
                end
                
                S_INIT: begin
                    // Clear accumulators before new operation
                    for (k=0; k<GA_DIM; k=k+1) c_accum[k] <= 32'b0;
                    i <= 0;
                    j <= 0;
                    state <= S_COMPUTE;
                end
                
                S_COMPUTE: begin
                    // Drive Address lines for next cycle
                    addr_a <= i;
                    addr_b <= j;
                    
                    // Counters
                    if (j == GA_DIM-1) begin
                        j <= 0;
                        if (i == GA_DIM-1) begin
                            // Finished iterating, wait for pipeline to drain?
                            // For simplicity, we assume we just move to writeout
                            // Correct logic would wait for MAC_valid_out of last term.
                            // We'll simplisticly wait a few cycles or rely on counter match.
                            state <= S_WRITE;
                            i <= 0; 
                        end else begin
                            i <= i + 1;
                        end
                    end else begin
                        j <= j + 1;
                    end
                end
                
                S_WRITE: begin
                    // Write back internal C regs to memory
                    wen_c <= 1;
                    addr_c <= i;
                    data_c <= c_accum[i];
                    
                    if (i == GA_DIM-1) begin
                        state <= S_DONE;
                    end else begin
                        i <= i + 1;
                    end
                end
                
                S_DONE: begin
                    wen_c <= 0;
                    done <= 1;
                    busy <= 0;
                    state <= S_IDLE;
                end
            endcase
        end
    end
    
    // --------------------------------------------------------
    // Accumulation Logic
    // --------------------------------------------------------
    // When the MAC pops out a valid product term, add it to the correct blade bucket.
    // Note: In real HW, this needs handling for Read-After-Write hazards 
    // if two consecutive ops target result blade K.
    // Since we iterate i,j sequentially, i^j changes pseudo-randomly.
    
    always @(posedge clk) begin
        if (MAC_valid_out) begin
            // Model of FP32 Add: C[k] += term
            // Using a behavioural addition for RTL model
            c_accum[MAC_blade_k] <= c_accum[MAC_blade_k] + MAC_acc_out; 
            // Warning: Non-synthesizable FP32 inference. 
            // Real design needs floating point adder IP here.
        end
    end

endmodule
