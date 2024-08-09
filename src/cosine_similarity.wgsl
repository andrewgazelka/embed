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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= params.m || col >= params.n) {
        return;
    }
    
    let a_offset = row * params.k;
    let b_offset = col * params.k;

    var dot_product = 0.0;
    var norm_a = 0.0;
    var norm_b = 0.0;

    for (var i = 0u; i < params.k; i++) {
        let a_val = a[a_offset + i];
        let b_val = b[b_offset + i];

        dot_product += a_val * b_val;
        norm_a += a_val * a_val;
        norm_b += b_val * b_val;
    }

    let similarity = dot_product / (sqrt(norm_a) * sqrt(norm_b));
    result[row * params.n + col] = similarity;
}