use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use ndarray::{Array2, ArrayView2};

fn cosine_similarity(a: ArrayView2<'_, f32>, b: ArrayView2<'_, f32>) -> Array2<f32> {
    let norm_a = a.map_axis(ndarray::Axis(1), |row| row.dot(&row).sqrt());
    let norm_b = b.map_axis(ndarray::Axis(1), |row| row.dot(&row).sqrt());

    let dot_product = a.dot(&b.t());

    dot_product / (&norm_a.insert_axis(ndarray::Axis(1)) * &norm_b)
}

fn main() -> anyhow::Result<()> {
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()?;

    let queries = vec![
        "What are some ways to reduce stress?",
        "What are the benefits of drinking green tea?",
    ];
    let docs = vec![
        "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
        "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
    ];

    let query_embeddings = model.encode(&queries)?;
    let doc_embeddings = model.encode(&docs)?;

    println!(
        "Query embeddings shape: {:?}",
        (query_embeddings.len(), query_embeddings[0].len())
    );
    println!(
        "Doc embeddings shape: {:?}",
        (doc_embeddings.len(), doc_embeddings[0].len())
    );

    let query_array = Array2::from_shape_vec((queries.len(), query_embeddings[0].len()), query_embeddings.into_iter().flatten().collect())?;
    let doc_array = Array2::from_shape_vec((docs.len(), doc_embeddings[0].len()), doc_embeddings.into_iter().flatten().collect())?;

    let similarities = cosine_similarity(query_array.view(), doc_array.view());
    println!("Similarities:\n{similarities:?}");

    Ok(())
}
