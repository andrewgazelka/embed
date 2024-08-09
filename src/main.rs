use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use tch::Device;

mod similarity;

fn main() -> anyhow::Result<()> {
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .with_device(Device::Mps)
        .create_model()?;

    let queries = vec![
        "What are some ways to reduce stress?",
        "What are the benefits of drinking green tea?",
    ];
    let docs = vec![
        "There are many effective ways to reduce stress. Some common techniques include deep \
         breathing, meditation, and physical activity. Engaging in hobbies, spending time in \
         nature, and connecting with loved ones can also help alleviate stress. Additionally, \
         setting boundaries, practicing self-care, and learning to say no can prevent stress from \
         building up.",
        "Green tea has been consumed for centuries and is known for its potential health \
         benefits. It contains antioxidants that may help protect the body against damage caused \
         by free radicals. Regular consumption of green tea has been associated with improved \
         heart health, enhanced cognitive function, and a reduced risk of certain types of \
         cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss \
         properties.",
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

    let query_embeddings_len = query_embeddings[0].len() as u32;

    let query_array: Vec<f32> = query_embeddings.into_iter().flatten().collect();
    let doc_array: Vec<f32> = doc_embeddings.into_iter().flatten().collect();

    // create tokio runtine
    let rt = tokio::runtime::Runtime::new().unwrap();

    println!("blocking on");
    rt.block_on(async {
        let similarities = similarity::cosine(
            &query_array,
            &doc_array,
            u32::try_from(queries.len()).unwrap(),
            u32::try_from(docs.len()).unwrap(),
            query_embeddings_len,
        )
        .await?;

        for (i, similarity) in similarities.iter().enumerate() {
            println!("{i}: {similarity}");
        }

        anyhow::Ok(())
    })
    .unwrap();

    Ok(())
}
