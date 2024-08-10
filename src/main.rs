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

    let query_embeddings_len = u32::try_from(query_embeddings[0].len())?;

    let query_array: Vec<f32> = query_embeddings.into_iter().flatten().collect();
    let doc_array: Vec<f32> = doc_embeddings.into_iter().flatten().collect();

    // create tokio runtine
    let rt = tokio::runtime::Runtime::new()?;

    let similarities = rt.block_on(async {
        similarity::cosine(
            &query_array,
            &doc_array,
            u32::try_from(queries.len()).unwrap(),
            u32::try_from(docs.len()).unwrap(),
            query_embeddings_len,
        )
        .await
    })?;

    print_similarities_grid(&similarities, &queries, &docs);

    Ok(())
}

fn print_similarities_grid(similarities: &[f32], queries: &[&str], docs: &[&str]) {
    use prettytable::{format, Cell, Row, Table};

    let mut table = Table::new();
    table.set_format(*format::consts::FORMAT_BOX_CHARS);

    // Add header row with document texts
    let mut header = Row::new(vec![Cell::new("Query / Document").style_spec("bFc")]);
    for doc in docs {
        header.add_cell(Cell::new(&truncate_text(doc, 30)).style_spec("bFc"));
    }
    table.add_row(header);

    // Add rows for each query
    for (i, query) in queries.iter().enumerate() {
        let mut row = Row::new(vec![Cell::new(&truncate_text(query, 30)).style_spec("bFr")]);
        for j in 0..docs.len() {
            let similarity = similarities[i * docs.len() + j];
            row.add_cell(Cell::new(&format!("{similarity:.4}")));
        }
        table.add_row(row);
    }

    // Print the table
    table.printstd();
}

fn truncate_text(text: &str, max_length: usize) -> String {
    if text.len() <= max_length {
        text.to_string()
    } else {
        format!("{}...", &text[..max_length - 3])
    }
}
