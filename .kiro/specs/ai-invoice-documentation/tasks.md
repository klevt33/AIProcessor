# Implementation Plan

- [x] 1. Create Solution Overview document



  - Write `docs/01-solution-overview.md` explaining overall architecture, processing flow, stage execution, special cases, and final output consolidation
  - Include sections: Overview, Key Concepts, Python Modules, Configuration, Business Logic, Processing Flow, Dependencies, Examples
  - Analyze `app.py`, `pipelines.py`, `ai_engine.py`, `constants.py`, `config.yaml`, `special_cases.yaml`
  - Include Mermaid diagram showing component interactions and pipeline flow
  - Add examples from `docs/sample_documents.txt` showing how invoice details flow through stages
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_




- [x] 2. Create Indexer document

  - Write `docs/02-indexer.md` explaining the semantic search indexer module
  - Include sections: Overview, Key Concepts, Python Modules, Configuration, Business Logic, Processing Flow, Dependencies, Examples
  - Analyze `indexer/semantic_search_indexer.py`, `azure_search_utils.py`, `config.yaml`
  - Explain multi-threaded architecture with data readers, processors, and writers
  - Document vector embeddings generation and Azure AI Search integration
  - Explain error handling: retry logic, circuit breakers, quota management



  - Document configuration parameters: batch sizes, queue sizes, thread counts
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 3. Create SQL Writer document

  - Write `docs/03-sql-writer.md` explaining the background database update service
  - Include sections: Overview, Key Concepts, Python Modules, Configuration, Business Logic, Processing Flow, Dependencies, Examples
  - Analyze `sql_writer.py`, `worker.py`, `cdb_utils.py`, `sql_utils.py`
  - Explain asynchronous processing from Cosmos DB to SQL database
  - Document retry logic with exponential backoff
  - Explain circuit breaker pattern for SQL outage handling
  - Document poison pill handling for problematic records
  - Explain duplicate invoice detail handling
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 4. Create COMPLETE_MATCH Stage document



  - Write `docs/04-complete-match-stage.md` explaining exact matching from database
  - Include sections: Overview, Key Concepts, Python Modules, Configuration, Business Logic, Processing Flow, Dependencies, Examples
  - Analyze `ai_stages.py` (COMPLETE_MATCH stage), `matching_utils.py`, `sql_utils.py`
  - Explain manufacturer extraction and part number detection logic
  - Document database querying strategy and SQL queries used
  - Explain confidence scoring based on match quality and verification status
  - Document fallback logic for manufacturer-only matching
  - List output fields: manufacturer, part number, UNSPSC, UPC, verification flag
  - Add examples showing exact matches vs manufacturer-only matches
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 5. Create CONTEXT_VALIDATOR Stage document



  - Write `docs/05-context-validator-stage.md` explaining match validation logic
  - Include sections: Overview, Key Concepts, Python Modules, Configuration, Business Logic, Processing Flow, Dependencies, Examples
  - Analyze `ai_stages.py` (CONTEXT_VALIDATOR stage), `prompts.py`, `llm.py`
  - Explain how LLM validates contextual relevance of COMPLETE_MATCH results
  - Document relationship types: DIRECT_MATCH, LOT_OR_KIT, REPLACEMENT_PART, ACCESSORY_PART, UNRELATED
  - Explain validation outcomes and when matches are accepted vs invalidated
  - Document how invalidation allows subsequent stages to provide alternatives
  - Add examples showing different relationship types and validation decisions
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 6. Create SEMANTIC_SEARCH Stage document



  - Write `docs/06-semantic-search-stage.md` explaining vector-based similarity search
  - Include sections: Overview, Key Concepts, Python Modules, Configuration, Business Logic, Processing Flow, Dependencies, Examples
  - Analyze `ai_stages.py` (SEMANTIC_SEARCH stage), `semantic_matching.py`, `azure_search_utils.py`
  - Explain embedding generation for invoice descriptions
  - Document hybrid search combining vector similarity and keyword matching
  - Explain confidence scoring for manufacturer name and UNSPSC
  - Document dependency on indexer having populated Azure AI Search
  - List output fields: manufacturer name, UNSPSC, confidence scores
  - Add examples showing semantic matches for ambiguous descriptions
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 7. Create FINETUNED_LLM Stage document



  - Write `docs/07-finetuned-llm-stage.md` explaining custom AI model extraction
  - Include sections: Overview, Key Concepts, Python Modules, Configuration, Business Logic, Processing Flow, Dependencies, Examples
  - Analyze `ai_stages.py` (FINETUNED_LLM stage), `training/fine_tuned_llm/infer_fine_tuned_llm.py`, `prompts.py`
  - Explain fine-tuned LLM extraction of manufacturer name, part number, UNSPSC
  - Document how manufacturer aliases and training examples are provided as context
  - Explain RAG (Retrieval Augmented Generation) for retrieving relevant examples
  - Document confidence boosting when RPA-provided values match AI-extracted values
  - List output fields: manufacturer name, part number, UNSPSC, confidence scores
  - Add examples showing extraction from complex descriptions
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_




- [x] 8. Create EXTRACTION_WITH_LLM_AND_WEBSEARCH Stage document
  - Write `docs/08-extraction-with-llm-and-websearch-stage.md` explaining web search with AI agent
  - Include sections: Overview, Key Concepts, Python Modules, Configuration, Business Logic, Processing Flow, Dependencies, Examples
  - Analyze `ai_stages.py` (EXTRACTION_WITH_LLM_AND_WEBSEARCH stage), `agents.py`, `matching_scores.py`
  - Explain AI agent with Bing search for finding product information online
  - Document how agent formulates queries and retrieves web results
  - Explain result ranking algorithm: manufacturer match, part number match, description similarity, data source priority
  - Document confidence calculation combining individual scores
  - List output fields: manufacturer name, part number, UNSPSC, UPC, source URL, ranked results
  - Add examples showing web search for obscure products
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 9. Add configuration documentation to all documents
  - Review all 8 documents to ensure configuration parameters are documented
  - Add tables showing YAML parameters from `config.yaml`, `special_cases.yaml`, `thresholds.yaml`, `confidences.yaml`
  - Explain how special cases modify the processing pipeline
  - Document confidence thresholds and their impact on final status
  - Explain batch sizes, retry settings, and other operational parameters
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 10. Add business logic documentation to stage documents
  - Review stage documents (tasks 4-8) to ensure business logic is thoroughly explained
  - Document confidence score calculation formulas for each stage
  - Explain manufacturer confidence: match type, relationship type considerations
  - Explain part number confidence: length, match quality, detection accuracy
  - Explain UNSPSC confidence: data source reliability, match quality
  - Document RPA data boosting logic
  - Explain final value selection based on highest confidence across stages
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 11. Add dependency documentation to all documents
  - Review all 8 documents to ensure dependencies are clearly documented
  - Document SEMANTIC_SEARCH dependency on indexer
  - Document COMPLETE_MATCH dependency on SQL database
  - Document SQL Writer dependency on Cosmos DB
  - Explain data passing between stages using shared cache
  - Document stage failure handling and continuation logic
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [ ] 12. Add conceptual examples to all documents
  - Review all 8 documents to add real processing examples from `docs/sample_documents.txt`
  - Add example for simple material items with clear manufacturer and part number
  - Add example for complex descriptions requiring web search
  - Add example for fee/labor items that skip certain stages
  - Add example for lot/kit items requiring special handling
  - Show progression of confidence scores and extracted values through stages
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [ ] 13. Final review and cross-referencing
  - Review all 8 documents for completeness against requirements
  - Add cross-references between related documents
  - Ensure terminology consistency across all documents
  - Verify all code references are accurate
  - Proofread for clarity and professionalism
  - Ensure no placeholder or TODO content remains
  - Validate examples against `docs/sample_documents.txt`
