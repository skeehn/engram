use engram_core::{
    error::{EngramError, Result},
    id::NodeId,
    types::Node,
};
use parking_lot::Mutex;
use std::path::Path;
use std::sync::Arc;
use tantivy::{
    collector::TopDocs,
    doc,
    query::QueryParser,
    schema::{Field, Schema, Value, FAST, STORED, STRING, TEXT},
    Index, IndexReader, IndexWriter, ReloadPolicy, TantivyError,
};

pub struct FtsIndex {
    index: Index,
    reader: IndexReader,
    writer: Arc<Mutex<IndexWriter>>,
    f_node_id: Field,
    f_node_type: Field,
    f_body: Field,
    f_tags: Field,
    f_confidence: Field,
    f_tx_time: Field,
}

impl FtsIndex {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        std::fs::create_dir_all(path).map_err(|e| EngramError::Index(e.to_string()))?;

        let mut schema_builder = Schema::builder();
        let f_node_id = schema_builder.add_text_field("node_id", STRING | STORED);
        let f_node_type = schema_builder.add_text_field("node_type", STRING | STORED);
        let f_body = schema_builder.add_text_field("body", TEXT | STORED);
        let f_tags = schema_builder.add_text_field("tags", TEXT | STORED);
        let f_confidence = schema_builder.add_f64_field("confidence", FAST | STORED);
        let f_tx_time = schema_builder.add_i64_field("tx_time_micros", FAST | STORED);
        let schema = schema_builder.build();

        let index = if path.join("meta.json").exists() {
            Index::open_in_dir(path).map_err(|e| EngramError::Index(e.to_string()))?
        } else {
            Index::create_in_dir(path, schema).map_err(|e| EngramError::Index(e.to_string()))?
        };

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e: TantivyError| EngramError::Index(e.to_string()))?;

        let writer = index
            .writer(50_000_000)
            .map_err(|e| EngramError::Index(e.to_string()))?;

        Ok(Self {
            index,
            reader,
            writer: Arc::new(Mutex::new(writer)),
            f_node_id,
            f_node_type,
            f_body,
            f_tags,
            f_confidence,
            f_tx_time,
        })
    }

    pub fn index_node(&self, node: &Node) -> Result<()> {
        let writer = self.writer.lock();
        // Delete any existing doc for this node_id first
        let id_term = tantivy::Term::from_field_text(self.f_node_id, node.id.as_ref());
        writer.delete_term(id_term);
        // Add the new document
        let tags_str = node.tags.join(" ");
        writer
            .add_document(doc!(
                self.f_node_id => node.id.as_ref(),
                self.f_node_type => node.node_type.to_string(),
                self.f_body => node.body.as_str(),
                self.f_tags => tags_str.as_str(),
                self.f_confidence => node.confidence as f64,
                self.f_tx_time => node.tx_time.timestamp_micros(),
            ))
            .map_err(|e| EngramError::Index(e.to_string()))?;
        Ok(())
    }

    pub fn remove_node(&self, node_id: &NodeId) -> Result<()> {
        let writer = self.writer.lock();
        let id_term = tantivy::Term::from_field_text(self.f_node_id, node_id.as_ref());
        writer.delete_term(id_term);
        Ok(())
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<(NodeId, f32)>> {
        self.commit()?;
        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.f_body, self.f_tags]);
        let parsed = query_parser
            .parse_query(query)
            .map_err(|e| EngramError::Query(e.to_string()))?;
        let top_docs = searcher
            .search(&parsed, &TopDocs::with_limit(limit))
            .map_err(|e| EngramError::Query(e.to_string()))?;
        let mut results = Vec::new();
        for (score, doc_addr) in top_docs {
            let retrieved: tantivy::TantivyDocument = searcher
                .doc(doc_addr)
                .map_err(|e| EngramError::Query(e.to_string()))?;
            if let Some(val) = retrieved.get_first(self.f_node_id) {
                if let Some(id_str) = val.as_str() {
                    results.push((NodeId::from(id_str.to_string()), score));
                }
            }
        }
        Ok(results)
    }

    pub fn search_typed(
        &self,
        query: &str,
        node_type: &str,
        limit: usize,
    ) -> Result<Vec<(NodeId, f32)>> {
        let combined = format!("node_type:{} AND ({})", node_type, query);
        self.search(&combined, limit)
    }

    pub fn search_by_tag(&self, tag: &str, limit: usize) -> Result<Vec<(NodeId, f32)>> {
        let query = format!("tags:{}", tag);
        self.search(&query, limit)
    }

    pub fn commit(&self) -> Result<()> {
        let mut writer = self.writer.lock();
        writer
            .commit()
            .map_err(|e| EngramError::Index(e.to_string()))?;
        Ok(())
    }

    pub fn reload(&self) -> Result<()> {
        self.reader
            .reload()
            .map_err(|e| EngramError::Index(e.to_string()))
    }

    pub fn doc_count(&self) -> Result<u64> {
        self.commit()?;
        let searcher = self.reader.searcher();
        Ok(searcher.num_docs())
    }
}
