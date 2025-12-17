# Requirements Document

## Introduction

This specification defines the requirements for creating comprehensive documentation for an AI-based invoice item matching solution. The documentation targets Business Analysts and QA professionals who need to understand the system's architecture, processing logic, and business rules without requiring deep technical implementation details.

The system processes invoice line items through multiple AI-powered stages to extract and match manufacturer names, part numbers, UNSPSC codes, and other product information.

The documentation will be delivered as **8 separate markdown documents**:
1. **Solution Overview** - Overall architecture and processing flow
2. **Indexer** - Semantic search indexer module
3. **SQL Writer** - Background service for database updates
4. **COMPLETE_MATCH Stage** - Exact matching from database
5. **CONTEXT_VALIDATOR Stage** - Match validation logic
6. **SEMANTIC_SEARCH Stage** - Vector-based similarity search
7. **FINETUNED_LLM Stage** - Custom AI model extraction
8. **EXTRACTION_WITH_LLM_AND_WEBSEARCH Stage** - Web search with AI agent

## Glossary

- **System**: The AI-based invoice item matching solution
- **Invoice Detail**: A single line item from an invoice containing product description and metadata
- **Processing Stage**: A discrete step in the invoice processing pipeline that performs specific extraction or validation tasks
- **UNSPSC**: United Nations Standard Products and Services Code - a standardized product classification system
- **Semantic Search**: Vector-based similarity search using AI embeddings to find matching products
- **LLM**: Large Language Model - AI model used for text understanding and extraction
- **Azure AI Search**: Microsoft's cloud-based search service with vector search capabilities
- **Cosmos DB**: NoSQL database used for storing processing logs and intermediate results
- **SDP**: SQL Data Platform - the SQL database containing master product data
- **RPA**: Robotic Process Automation - the upstream system that extracts initial invoice data
- **Confidence Score**: A numerical value (0-100) indicating the system's certainty in an extracted value

## Requirements

### Requirement 1

**User Story:** As a Business Analyst, I want a Solution Overview document, so that I can understand the overall system architecture, processing flow, and how all components work together.

#### Acceptance Criteria

1. WHEN the document describes the architecture THEN the system SHALL explain the main components (API, Pipeline, Stages, Indexer, SQL Writer, Databases)
2. WHEN the document describes the processing flow THEN the system SHALL show how invoice details flow through the pipeline stages
3. WHEN the document describes stage execution THEN the system SHALL explain how stages execute sequentially and pass data between them
4. WHEN the document describes special cases THEN the system SHALL explain how the pipeline is modified for generic items, lot items, and other scenarios
5. WHEN the document describes final output THEN the system SHALL explain how the best results from all stages are consolidated and written to databases

### Requirement 2

**User Story:** As a QA professional, I want an Indexer document, so that I can understand how product data is synchronized to the search index and verify indexing operations.

#### Acceptance Criteria

1. WHEN the document describes the indexer purpose THEN the system SHALL explain that it synchronizes product data from SQL to Azure AI Search
2. WHEN the document describes the indexing process THEN the system SHALL explain the multi-threaded architecture with data readers, processors, and writers
3. WHEN the document describes vector embeddings THEN the system SHALL explain how product descriptions are converted to embeddings for semantic search
4. WHEN the document describes error handling THEN the system SHALL explain retry logic, circuit breakers, and quota management
5. WHEN the document describes configuration THEN the system SHALL explain batch sizes, queue sizes, and indexer settings

### Requirement 3

**User Story:** As a Business Analyst, I want a SQL Writer document, so that I can understand how processing results are written to the database and troubleshoot update issues.

#### Acceptance Criteria

1. WHEN the document describes the SQL Writer purpose THEN the system SHALL explain that it asynchronously writes AI processing results from Cosmos DB to SQL database
2. WHEN the document describes the processing flow THEN the system SHALL explain how it polls Cosmos DB, processes documents, and updates SQL records
3. WHEN the document describes error handling THEN the system SHALL explain retry logic with exponential backoff and poison pill handling
4. WHEN the document describes the circuit breaker THEN the system SHALL explain how it prevents cascading failures during SQL outages
5. WHEN the document describes duplicate handling THEN the system SHALL explain how duplicate invoice details are updated with the same values

### Requirement 4

**User Story:** As a QA professional, I want a COMPLETE_MATCH Stage document, so that I can understand how exact matching identifies products from the database and verify the matching logic.

#### Acceptance Criteria

1. WHEN the document describes the stage purpose THEN the system SHALL explain that it performs exact matching against the product database
2. WHEN the document describes the matching process THEN the system SHALL explain manufacturer extraction, part number detection, and database querying
3. WHEN the document describes confidence scoring THEN the system SHALL explain how scores are calculated based on match quality and verification status
4. WHEN the document describes fallback logic THEN the system SHALL explain manufacturer-only matching when no part number match exists
5. WHEN the document describes output THEN the system SHALL explain the returned fields (manufacturer, part number, UNSPSC, UPC, verification flag)

### Requirement 5

**User Story:** As a Business Analyst, I want a CONTEXT_VALIDATOR Stage document, so that I can understand how the system validates that matched products are contextually appropriate.

#### Acceptance Criteria

1. WHEN the document describes the stage purpose THEN the system SHALL explain that it validates the contextual relevance of COMPLETE_MATCH results
2. WHEN the document describes the validation process THEN the system SHALL explain how an LLM compares invoice text to matched product descriptions
3. WHEN the document describes relationship types THEN the system SHALL explain DIRECT_MATCH, LOT_OR_KIT, REPLACEMENT_PART, ACCESSORY_PART, and UNRELATED classifications
4. WHEN the document describes validation outcomes THEN the system SHALL explain when matches are accepted versus invalidated
5. WHEN the document describes invalidation THEN the system SHALL explain how invalidated matches allow subsequent stages to provide alternative results

### Requirement 6

**User Story:** As a QA professional, I want a SEMANTIC_SEARCH Stage document, so that I can understand how vector-based similarity matching finds similar products and verify the search logic.

#### Acceptance Criteria

1. WHEN the document describes the stage purpose THEN the system SHALL explain that it uses vector embeddings to find semantically similar products
2. WHEN the document describes the search process THEN the system SHALL explain embedding generation and hybrid search (vector + keyword)
3. WHEN the document describes confidence scoring THEN the system SHALL explain how scores are calculated for manufacturer name and UNSPSC
4. WHEN the document describes dependencies THEN the system SHALL explain the requirement for the indexer to have populated the search index
5. WHEN the document describes output THEN the system SHALL explain the returned fields (manufacturer name, UNSPSC, confidence scores)

### Requirement 7

**User Story:** As a Business Analyst, I want a FINETUNED_LLM Stage document, so that I can understand how the custom-trained AI model extracts product information.

#### Acceptance Criteria

1. WHEN the document describes the stage purpose THEN the system SHALL explain that it uses a fine-tuned LLM to extract manufacturer name, part number, and UNSPSC
2. WHEN the document describes the extraction process THEN the system SHALL explain how manufacturer aliases and training examples are provided as context
3. WHEN the document describes RAG (Retrieval Augmented Generation) THEN the system SHALL explain how relevant examples are retrieved and included in prompts
4. WHEN the document describes confidence scoring THEN the system SHALL explain how RPA-provided values can boost confidence scores
5. WHEN the document describes output THEN the system SHALL explain the returned fields (manufacturer name, part number, UNSPSC, confidence scores)

### Requirement 8

**User Story:** As a QA professional, I want an EXTRACTION_WITH_LLM_AND_WEBSEARCH Stage document, so that I can understand how web search finds product information from online sources and verify the ranking logic.

#### Acceptance Criteria

1. WHEN the document describes the stage purpose THEN the system SHALL explain that it uses an AI agent with Bing search to find product information online
2. WHEN the document describes the search process THEN the system SHALL explain how the AI agent formulates queries and retrieves web results
3. WHEN the document describes result ranking THEN the system SHALL explain the scoring algorithm considering manufacturer match, part number match, description similarity, and data source priority
4. WHEN the document describes confidence calculation THEN the system SHALL explain how individual scores are combined into overall confidence
5. WHEN the document describes output THEN the system SHALL explain the returned fields (manufacturer name, part number, UNSPSC, UPC, source URL, ranked results)

### Requirement 9

**User Story:** As a Business Analyst, I want to understand the configuration parameters, so that I can adjust system behavior for different use cases.

#### Acceptance Criteria

1. WHEN configuration files are loaded THEN the system SHALL read settings from YAML files for stages, thresholds, and special cases
2. WHEN special cases are defined THEN the system SHALL modify the processing pipeline by skipping or adding stages
3. WHEN confidence thresholds are configured THEN the system SHALL use them to determine final invoice line status
4. WHEN batch sizes are configured THEN the system SHALL use them for database queries and index operations
5. WHEN retry settings are configured THEN the system SHALL use them for error handling and circuit breaker behavior

### Requirement 10

**User Story:** As a QA professional, I want to understand the business logic for confidence score calculation, so that I can validate that scores accurately reflect extraction quality.

#### Acceptance Criteria

1. WHEN calculating manufacturer confidence THEN the system SHALL consider match type (exact, partial, no match) and relationship type (direct, parent, child)
2. WHEN calculating part number confidence THEN the system SHALL consider part number length, match quality, and detection accuracy
3. WHEN calculating UNSPSC confidence THEN the system SHALL consider data source reliability and match quality
4. WHEN boosting confidence with RPA data THEN the system SHALL increase scores when RPA-provided values match AI-extracted values
5. WHEN determining final values THEN the system SHALL select the value with the highest confidence score from all stages

### Requirement 11

**User Story:** As a Business Analyst, I want to understand the dependencies between modules and stages, so that I can identify potential bottlenecks and failure points.

#### Acceptance Criteria

1. WHEN the semantic search stage runs THEN the system SHALL depend on the indexer having populated the Azure AI Search index
2. WHEN the complete match stage runs THEN the system SHALL depend on the SQL database containing up-to-date product data
3. WHEN the SQL writer runs THEN the system SHALL depend on Cosmos DB containing completed processing results
4. WHEN stages execute THEN the system SHALL pass data between stages using a shared cache object
5. WHEN a stage fails THEN the system SHALL continue to subsequent stages unless the failure is critical

### Requirement 12

**User Story:** As a QA professional, I want to see conceptual processing examples, so that I can understand how different types of invoice descriptions are handled.

#### Acceptance Criteria

1. WHEN documentation includes examples THEN the system SHALL provide examples for simple material items with clear manufacturer and part number
2. WHEN documentation includes examples THEN the system SHALL provide examples for complex descriptions requiring web search
3. WHEN documentation includes examples THEN the system SHALL provide examples for fee/labor items that skip certain stages
4. WHEN documentation includes examples THEN the system SHALL provide examples for lot/kit items that require special handling
5. WHEN documentation includes examples THEN the system SHALL show the progression of confidence scores and extracted values through stages
