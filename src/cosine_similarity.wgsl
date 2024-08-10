// WGSL Shader: Cosine Similarity Computation
//
// This shader calculates the cosine similarity between vectors in two matrices.
// Cosine similarity measures the cosine of the angle between two vectors,
// providing a value between -1 and 1, where:
//   1 indicates vectors pointing in the same direction
//   0 indicates orthogonal vectors
//  -1 indicates vectors pointing in opposite directions

struct Params {
    m: u32,  // Number of rows
    n: u32,  // Number of columns
    k: u32,  // Length of each vector
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// The workgroup size is set to 16x16:
// - This is a common choice for compute shaders as it balances parallelism and resource usage.
// - 16x16 = 256 total threads, which is often the maximum supported by GPUs.
// - It's a power of 2, which can lead to efficient memory access patterns.
// - This size allows for good occupancy on many GPU architectures.
// - However, the optimal size can vary based on the specific GPU and the nature of the computation.
//   Experimentation with different sizes (e.g., 8x8, 32x32) might be beneficial for performance tuning.
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Imagine a big grid where each cell represents a computation we need to do.
    // 'row' tells us which row of this grid we're working on.
    // It corresponds to a vector in matrix A.
    let row = global_id.x;

    // 'col' tells us which column of the grid we're working on.
    // It corresponds to a vector in matrix B.
    let col = global_id.y;

    // We only want to do work if we're within the bounds of our matrices.
    // If our 'row' or 'col' is too big, we stop here.
    if (row >= params.m || col >= params.n) {
        return;
    }

    // Now, let's find where our vectors start in the big array of numbers.
    // For matrix A, we move down 'row' rows, and each row has 'k' elements.
    let a_offset = row * params.k;

    // For matrix B, we move over 'col' columns, and each column has 'k' elements.
    let b_offset = col * params.k;

    // These variables will help us calculate the similarity.
    var dot_product = 0.0;
    var norm_a = 0.0;
    var norm_b = 0.0;

    // We go through each element of our two vectors.
    for (var i = 0u; i < params.k; i++) {
        // Get the i-th element of the vector from matrix A.
        let a_val = a[a_offset + i];

        // Get the i-th element of the vector from matrix B.
        let b_val = b[b_offset + i];

        // Update our calculations.
        dot_product += a_val * b_val;
        norm_a += a_val * a_val;
        norm_b += b_val * b_val;
    }

    // Calculate the final similarity.
    let similarity = dot_product / (sqrt(norm_a) * sqrt(norm_b));

    // Store the result in the right place.
    // We're filling in the cell at (row, col) in our result grid.
    result[row * params.n + col] = similarity;
}
