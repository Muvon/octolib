// Copyright 2026 Muvon Un Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::model_load_lock;
use std::sync::Arc;

impl super::HuggingFaceProvider {
    pub(crate) async fn get_model_for_test(
        model_name: &str,
    ) -> anyhow::Result<Arc<super::HuggingFaceModel>> {
        Self::get_model(model_name).await
    }
}

#[tokio::test]
async fn test_model_load_locks_are_scoped_per_model() {
    let first = model_load_lock("same-model").await;
    let second = model_load_lock("same-model").await;
    let other = model_load_lock("other-model").await;

    assert!(Arc::ptr_eq(&first, &second));
    assert!(!Arc::ptr_eq(&first, &other));

    let first_guard = first.lock().await;
    assert!(second.try_lock().is_err());
    assert!(other.try_lock().is_ok());
    drop(first_guard);
    assert!(second.try_lock().is_ok());
}

#[test]
fn test_roberta_tokenizer_building() {
    // Test that we can build a RoBERTa-style tokenizer using BPE::from_file approach
    use tokenizers::{
        models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel,
        processors::roberta::RobertaProcessing, Tokenizer,
    };

    // Create temporary files for testing
    let vocab_file = std::env::temp_dir().join("test_vocab.json");
    let merges_file = std::env::temp_dir().join("test_merges.txt");

    // Write test vocab - must include all tokens used in merges
    let vocab_content = r#"{"<s>":0,"<pad>":1,"</s>":2,"<unk>":3,"h":4,"e":5,"l":6,"o":7,"r":8,"he":9,"ll":10,"or":11,"hello":12,"world":13}"#;
    std::fs::write(&vocab_file, vocab_content).expect("Failed to write vocab");

    // Write test merges
    let merges_content = "#version: 0.2\nh e\nl l\no r";
    std::fs::write(&merges_file, merges_content).expect("Failed to write merges");

    // Build BPE model using from_file
    let bpe = BPE::from_file(vocab_file.to_str().unwrap(), merges_file.to_str().unwrap())
        .unk_token("<unk>".to_string())
        .build()
        .expect("Failed to build BPE tokenizer");

    let mut tokenizer = Tokenizer::new(bpe);

    // Add ByteLevel pre-tokenizer (for RoBERTa)
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));

    // Add RoBERTa post-processing
    let post_processor = RobertaProcessing::new(
        ("</s>".to_string(), 2), // SEP token
        ("<s>".to_string(), 0),  // CLS token
    )
    .trim_offsets(false)
    .add_prefix_space(true);
    tokenizer.with_post_processor(Some(post_processor));

    // Test that tokenizer works
    let test_text = "hello world";
    let encoding = tokenizer
        .encode(test_text, false)
        .expect("Failed to encode");

    assert!(
        !encoding.get_ids().is_empty(),
        "Encoding should produce tokens"
    );
    println!("✓ RoBERTa-style tokenizer built successfully using BPE::from_file");

    // Clean up
    let _ = std::fs::remove_file(vocab_file);
    let _ = std::fs::remove_file(merges_file);
}

#[test]
fn test_merges_parsing() {
    // Test that we correctly parse merges.txt format
    let merges_content = r#"#version: 0.2
Ġ t
Ġ a
h e
Ġt he
i n"#;

    let merges: Vec<(String, String)> = merges_content
        .lines()
        .skip(1) // Skip header line
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                Some((parts[0].to_string(), parts[1].to_string()))
            } else {
                None
            }
        })
        .collect();

    assert_eq!(merges.len(), 5);
    assert_eq!(merges[0], ("Ġ".to_string(), "t".to_string()));
    println!("✓ Merges parsing works correctly");
}
