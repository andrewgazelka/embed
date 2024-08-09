// WGSL (WebGPU Shading Language) shader for cosine similarity computation

// Binding declarations:
// These define the input and output buffers, as well as uniform parameters.
// The @group(0) attribute specifies the bind group index.
// The @binding(x) attribute specifies the binding index within the group.

// Input buffer A (read-only storage buffer)
@group(0) @binding(0) var<storage, read> a: array<f32>;

// Input buffer B (read-only storage buffer)
@group(0) @binding(1) var<storage, read> b: array<f32>;

// Output buffer for results (read-write storage buffer)
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

// Uniform buffer for parameters (read-only uniform buffer)
// This contains the dimensions of our computation: m, n, and k
@group(0) @binding(3) var<uniform> params: vec3<u32>;

// Main compute shader function
// @compute attribute marks this as a compute shader
// @workgroup_size(16, 16) specifies the size of the workgroup (16x16 threads)
@compute @workgroup_size(16, 16)
fn main(
    // @builtin(global_invocation_id) is a built-in input that gives us
    // the global ID of the current thread in the dispatch
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    // Extract dimensions from the params uniform
    let m = params.x;  // Number of rows in matrix A
    let n = params.y;  // Number of columns in matrix B
    let k = params.z;  // Number of columns in A / rows in B

    // Determine which element this thread is responsible for
    let row = global_id.x;
    let col = global_id.y;

    // Only perform computation if this thread maps to a valid matrix element
    if (row < m && col < n) {
        // Initialize accumulators for dot product and norms
        var dot_product: f32 = 0.0;
        var norm_a: f32 = 0.0;
        var norm_b: f32 = 0.0;

        // Compute dot product and norms
        for (var i: u32 = 0u; i < k; i = i + 1u) {
            // Get corresponding elements from A and B
            let a_val = a[row * k + i];
            let b_val = b[col * k + i];

            // Accumulate dot product
            dot_product += a_val * b_val;

            // Accumulate squared norms
            norm_a += a_val * a_val;
            norm_b += b_val * b_val;
        }

        // Compute cosine similarity:
        // cos(theta) = (a · b) / (||a|| * ||b||)
        // where a · b is the dot product, and ||a|| and ||b|| are the magnitudes (L2 norms)
        result[row * n + col] = dot_product / (sqrt(norm_a) * sqrt(norm_b));
    }
}