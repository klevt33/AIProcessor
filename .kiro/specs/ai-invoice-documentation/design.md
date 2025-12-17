# Design Document

## Overview

This design document outlines the approach for creating comprehensive documentation for an AI-based invoice item matching solution. The documentation will consist of 8 separate markdown files, each targeting Business Analysts and QA professionals who need to understand the system without deep technical implementation details.

The documentation project is **documentation-only** - no code changes, refactoring, or bug fixes will be performed. The deliverables are markdown files that explain:
- System architecture and processing flow
- Key modules (Indexer, SQL Writer)
- Processing stages (COMPLETE_MATCH, CONTEXT_VALIDATOR, SEMANTIC_SEARCH, FINETUNED_LLM, EXTRACTION_WITH_LLM_AND_WEBSEARCH)
- Business logic and rules
- Configuration parameters
- Dependencies and relationships

## Architecture

### Documentation Structure

The documentation will be organized as 8 independent markdown files in a `docs/` directory:

```
docs/
├── 01-solution-overview.md
├── 02-indexer.md
├── 03-sql-writer.md
├── 04-complete-match-stage.md
├── 05-context-validator-stage.md
├── 06-semantic-search-stage.md
├── 07-finetuned-llm-stage.md
└── 08-extraction-with-llm-and-websearch-stage.md
```

### Documentation Approach

Each document will follow a consistent structure:
1. **Overview** - High-level purpose and role in the system
2. **Key Concepts** - Important concepts needed to understand the module/stage
3. **Python Modules** - Files and key functions/classes involved
4. **Configuration** - YAML parameters and their meanings
5. **Business Logic** - Detailed explanation of rules and decision-making
6. **Processing Flow** - Step-by-step walkthrough of operations
7. **Dependencies** - Relationships with other modules/stages
8. **Examples** - Conceptual examples showing typical scenarios

## Components and Interfaces

### Source Code Analysis

The documentation will be created by analyzing the following source files:

**Core Processing:**
- `app.py` - Main API application
- `pipelines.py` - Pipeline orchestration
- `ai_engine.py` - Stage execution engine
- `ai_stages.py` - Individual stage implementations
- `worker.py` - Background processing workers

**Key Modules:**
- `indexer/semantic_search_indexer.py` - Search index synchronization
- `sql_writer.py` - Database update service
- `azure_search_utils.py` - Search operations
- `matching_utils.py` - Product matching logic
- `semantic_matching.py` - Vector similarity matching

**Configuration:**
- `config.yaml` - Main configuration
- `special_cases.yaml` - Pipeline modifications
- `thresholds.yaml` - Confidence thresholds
- `confidences.yaml` - Scoring rules

**Supporting:**
- `constants.py` - Stage names, statuses, enums
- `prompts.py` - LLM prompts
- `agents.py` - AI agent implementations
- `llm.py` - LLM client wrappers

### Reference Data

The documentation will reference:
- `docs/sample_documents.txt` - Real processing examples showing stage progression
- SQL table schemas from `sql/tables.sql`
- Database queries from `sql/*.sql` files

## Data Models

### Document Structure

Each markdown document will contain:

**Metadata Section:**
- Document title
- Last updated date
- Target audience
- Related documents (cross-references)

**Content Sections:**
- Structured headings (H2, H3)
- Code blocks for configuration examples
- Tables for parameter descriptions
- Mermaid diagrams for flows (where applicable)
- Bullet lists for business rules
- Numbered lists for sequential processes

**Example Format:**
```markdown
# Module Name

**Last Updated:** YYYY-MM-DD  
**Audience:** Business Analysts, QA Professionals  
**Related:** [Other Document](link)

## Overview
[High-level description]

## Key Concepts
[Important concepts explained]

## Python Modules
- `module.py` - Description
  - `function_name()` - What it does

## Configuration
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| param_name | int | What it controls | 100 |

## Business Logic
[Detailed rules and decision-making]

## Processing Flow
1. Step one
2. Step two
3. Step three

## Dependencies
- Depends on: [Other modules]
- Used by: [Other stages]

## Examples
### Example 1: Simple Case
[Walkthrough]
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Since this is a documentation project, the correctness properties focus on verifying that the documentation contains the required content. These are example-based properties that can be verified through manual review or automated content checking.

### Property 1: Solution Overview Completeness
*For the* Solution Overview document, it should contain sections explaining the main components (API, Pipeline, Stages, Indexer, SQL Writer, Databases), processing flow, stage execution, special cases, and final output consolidation.
**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

### Property 2: Indexer Documentation Completeness
*For the* Indexer document, it should contain sections explaining the indexer purpose, multi-threaded architecture, vector embeddings, error handling mechanisms, and configuration parameters.
**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

### Property 3: SQL Writer Documentation Completeness
*For the* SQL Writer document, it should contain sections explaining the SQL Writer purpose, processing flow, error handling with retries, circuit breaker pattern, and duplicate handling logic.
**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

### Property 4: COMPLETE_MATCH Stage Documentation Completeness
*For the* COMPLETE_MATCH Stage document, it should contain sections explaining the stage purpose, matching process, confidence scoring, fallback logic, and output fields.
**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**

### Property 5: CONTEXT_VALIDATOR Stage Documentation Completeness
*For the* CONTEXT_VALIDATOR Stage document, it should contain sections explaining the stage purpose, validation process, relationship type classifications, validation outcomes, and invalidation behavior.
**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

### Property 6: SEMANTIC_SEARCH Stage Documentation Completeness
*For the* SEMANTIC_SEARCH Stage document, it should contain sections explaining the stage purpose, search process with embeddings, confidence scoring, dependencies on the indexer, and output fields.
**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

### Property 7: FINETUNED_LLM Stage Documentation Completeness
*For the* FINETUNED_LLM Stage document, it should contain sections explaining the stage purpose, extraction process with context, RAG technique, confidence boosting, and output fields.
**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

### Property 8: EXTRACTION_WITH_LLM_AND_WEBSEARCH Stage Documentation Completeness
*For the* EXTRACTION_WITH_LLM_AND_WEBSEARCH Stage document, it should contain sections explaining the stage purpose, search process with AI agent, result ranking algorithm, confidence calculation, and output fields.
**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**

## Error Handling

### Documentation Quality Assurance

**Content Verification:**
- Each document must be reviewed to ensure all required sections are present
- Technical accuracy must be verified against source code
- Business logic must be validated with domain experts
- Examples must be based on real processing data from `docs/sample_documents.txt`

**Consistency Checks:**
- Terminology must be consistent across all documents
- Cross-references must be valid
- Configuration parameter names must match actual YAML files
- Stage names must match constants defined in `constants.py`

**Completeness Validation:**
- All 8 documents must be created
- Each document must cover all topics specified in requirements
- No placeholder or "TODO" content should remain
- All code references must be accurate

## Testing Strategy

### Documentation Validation Approach

Since this is a documentation project, testing focuses on content validation rather than code testing.

**Manual Review Process:**
1. **Content Completeness Check** - Verify each document contains all required sections
2. **Technical Accuracy Review** - Validate against source code
3. **Business Logic Validation** - Confirm with domain experts
4. **Example Verification** - Ensure examples match real processing data
5. **Cross-Reference Check** - Verify all links and references are valid

**Automated Checks (Optional):**
- Markdown linting for formatting consistency
- Link validation for cross-references
- Spell checking for professional quality
- Table of contents generation

**Review Checklist per Document:**
- [ ] Overview section present and clear
- [ ] Key concepts explained for non-technical audience
- [ ] Python modules listed with descriptions
- [ ] Configuration parameters documented
- [ ] Business logic explained in detail
- [ ] Processing flow described step-by-step
- [ ] Dependencies identified
- [ ] Examples provided with real data
- [ ] No assumptions or generalizations
- [ ] Facts verified against code
- [ ] Terminology consistent with other documents

### Documentation Standards

**Writing Guidelines:**
- Use clear, professional language appropriate for Business Analysts and QA
- Avoid deep technical implementation details
- Focus on "what" and "why" rather than "how" (code-level)
- Use tables for structured information (parameters, fields)
- Use diagrams for complex flows (Mermaid format)
- Use bullet lists for business rules
- Use numbered lists for sequential processes
- Include real examples from `docs/sample_documents.txt`

**Formatting Standards:**
- Use H2 for main sections, H3 for subsections
- Use code blocks with language tags for configuration examples
- Use tables for parameter documentation
- Use bold for emphasis on key terms
- Use inline code for field names, file names, function names
- Use blockquotes for important notes or warnings

## Implementation Notes

### Documentation Creation Process

1. **Analysis Phase**
   - Read and understand source code for each module/stage
   - Extract business logic and decision rules
   - Identify configuration parameters
   - Map dependencies between components
   - Review sample processing documents

2. **Writing Phase**
   - Create each document following the standard structure
   - Write content targeting Business Analyst/QA audience
   - Include real examples from sample documents
   - Add configuration tables and parameter descriptions
   - Create flow diagrams where helpful

3. **Review Phase**
   - Verify technical accuracy against code
   - Check completeness against requirements
   - Validate examples against sample data
   - Ensure consistency across documents
   - Proofread for clarity and professionalism

4. **Finalization Phase**
   - Add cross-references between documents
   - Generate table of contents if needed
   - Final formatting and linting
   - Deliver all 8 markdown files

### Key Focus Areas

**For Solution Overview:**
- Explain the big picture architecture
- Show how components interact
- Describe the pipeline flow
- Explain special case handling

**For Indexer:**
- Explain synchronization process
- Describe multi-threading architecture
- Explain vector embeddings concept
- Document error handling and retries

**For SQL Writer:**
- Explain asynchronous processing
- Describe retry and circuit breaker patterns
- Explain poison pill handling
- Document duplicate record handling

**For Processing Stages:**
- Explain stage purpose and when it runs
- Describe inputs and outputs
- Detail business logic and rules
- Explain confidence score calculation
- Show example processing scenarios
- Document dependencies on other stages

### Content Sources

**Primary Sources:**
- Source code files (`.py` files)
- Configuration files (`.yaml` files)
- Sample processing documents (`docs/sample_documents.txt`)

**Secondary Sources:**
- Code comments and docstrings
- SQL queries and table definitions
- Constants and enumerations

**Validation Sources:**
- Real processing examples
- Domain expert knowledge
- Existing documentation (if any)
