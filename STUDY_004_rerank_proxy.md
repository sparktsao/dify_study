# Setting Up a Rerank Model Proxy for DIFY: A Complete Guide

## Why Rerank Models Are Critical for RAG Systems

Traditional RAG (Retrieval-Augmented Generation) systems rely heavily on embedding models for document retrieval. However, these systems face significant limitations:

**The Embedding Model Bottleneck:**
- Most embedding models are relatively small with limited dimensions (e.g., 1536)
- They struggle to capture complex semantic relationships and nuanced context
- Document chunking destroys important contextual information and relationships
- Simple cosine similarity often fails to identify truly relevant content

**The Cost vs. Quality Dilemma:**
- Using LLMs directly for every query is prohibitively expensive
- Embedding-only approaches deliver poor retrieval quality
- This creates a gap between cost-effective but low-quality retrieval and expensive but accurate LLM-based approaches

**Rerank Models: The Missing Link**
Rerank models bridge this gap by:
- Taking the top-k results from embedding-based retrieval
- Using more sophisticated models to re-score and re-order results
- Providing significantly better relevance scoring at a fraction of LLM costs
- Acting as a crucial quality gate that makes RAG systems actually work in production

## The Provider Support Problem

Despite their importance, rerank models face a significant ecosystem challenge:

**Limited Provider Support:**
- **OpenAI**: Doesn't provide rerank models, making OpenAI-compatible API interfaces unclear
- **Ollama**: Offers some rerank models but lacks clear serving documentation and interface definitions
- **DIFY**: Only considers LLM and embedding models from Ollama, ignoring rerank capabilities
- **Most Providers**: Either don't support rerank models or have poorly documented implementations

**The HuggingFace TEI Solution:**
After exploring various options, HuggingFace's Text Embeddings Inference (TEI) emerges as the most viable solution:
- Supports high-quality rerank models like BAAI/bge-reranker-v2-m3
- Provides excellent performance through Rust-based inference
- However, its API format doesn't align with DIFY's expectations

This guide shows how to bridge that gap with a proxy, enabling a purely local, high-performance rerank solution.

## Architecture Overview

Here's the complete architecture showing how DIFY communicates with the rerank model through our proxy:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐
│                 │    │                 │    │                             │
│      DIFY       │    │  Python Proxy   │    │  text-embeddings-inference  │
│   Application   │    │     Server      │    │        (Rust Server)        │
│                 │    │                 │    │                             │
└─────────────────┘    └─────────────────┘    └─────────────────────────────┘
         │                       │                           │
         │ POST /rerank          │ POST /rerank              │
         │ Port: 8086            │ Port: 8085                │
         │                       │                           │
         │ DIFY Format:          │ HuggingFace Format:       │
         │ {                     │ {                         │
         │   "model": "...",     │   "query": "...",         │
         │   "query": "...",     │   "texts": [...],         │
         │   "documents": [...], │   "truncate": true,       │
         │   "top_n": 3,         │   "truncation_direction": │
         │   "return_documents"  │   "Right",                │
         │ }                     │   "raw_scores": false     │
         │                       │ }                         │
         ▼                       ▼                           ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐
│                 │    │                 │    │                             │
│ Docker Network  │◄──►│  Format         │◄──►│    BAAI/bge-reranker-v2-m3  │
│ host.docker.    │    │  Conversion     │    │         Model               │
│ internal:8086   │    │  & Proxy Logic  │    │                             │
│                 │    │                 │    │                             │
└─────────────────┘    └─────────────────┘    └─────────────────────────────┘
         │                       │                           │
         │ DIFY Response:        │ HuggingFace Response:     │
         │ {                     │ [                         │
         │   "results": [        │   {"index": 0,            │
         │     {                 │    "score": 0.878},       │
         │       "index": 0,     │   {"index": 1,            │
         │       "document": {   │    "score": 0.385}        │
         │         "text": "..." │ ]                         │
         │       },              │                           │
         │       "relevance_     │                           │
         │       score": 0.878   │                           │
         │     }                 │                           │
         │   ]                   │                           │
         │ }                     │                           │
         ▲                       ▲                           ▲
         └───────────────────────┴───────────────────────────┘

Network Flow:
1. DIFY → Proxy (localhost:8086 or host.docker.internal:8086)
2. Proxy → text-embeddings-inference (localhost:8085)
3. text-embeddings-inference → Proxy (ranked results)
4. Proxy → DIFY (formatted response)

Components Purpose:
┌─────────────────────────────────────────────────────────────────────────┐
│ Component                │ Port │ Purpose                                │
├─────────────────────────────────────────────────────────────────────────┤
│ DIFY Application         │ N/A  │ Main LLM application platform          │
│ Python Proxy Server      │ 8086 │ Format conversion & API translation    │
│ text-embeddings-router   │ 8085 │ High-performance rerank inference      │
│ BAAI/bge-reranker-v2-m3  │ N/A  │ Multilingual rerank model              │
└─────────────────────────────────────────────────────────────────────────┘
```

## The Challenge

DIFY, a popular LLM application development platform, expects rerank models to follow a specific API format. However, Hugging Face's text-embeddings-inference server uses a different schema. To make them work together, we need a proxy that translates between these two formats.

## The Solution Overview

We'll set up:
1. A high-performance rerank model (BAAI/bge-reranker-v2-m3) using text-embeddings-inference
2. A Python proxy server that converts between DIFY and Hugging Face formats
3. Proper Docker networking to expose the service

## Step 1: Setting Up the Rerank Model

### Installing Dependencies

First, install Rust and clone the text-embeddings-inference repository:

```bash
brew install rust
git clone https://github.com/huggingface/text-embeddings-inference.git
cd text-embeddings-inference
cargo install --path router -F metal
```

### Choosing the Right Model

For this setup, we're using **BAAI/bge-reranker-v2-m3**, which offers:
- Excellent multilingual support
- High accuracy for document ranking
- Good performance characteristics
- Wide compatibility

### Starting the Rerank Server

Launch the text-embeddings-inference server:

```bash
text-embeddings-router \
  --model-id BAAI/bge-reranker-v2-m3 \
  --port 8085
```

This starts the server on port 8085, ready to accept rerank requests.

## Step 2: Testing the Base Setup

### Local Testing

Test the server locally to ensure it's working:

```bash
curl http://localhost:8085/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain gravity",
    "texts": ["Gravity is...", "A force that..."],
    "truncate": true,
    "truncation_direction": "Right",
    "raw_scores": false
  }'
```

Expected response:
```json
[{"index":0,"score":0.8782099},{"index":1,"score":0.38502774}]
```

### Docker Network Testing

For Docker services, test using the Docker internal hostname:

```bash
curl http://host.docker.internal:8085/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain gravity",
    "texts": ["Gravity is...", "A force that..."],
    "truncate": true,
    "truncation_direction": "Right",
    "raw_scores": false
  }'
```

## Step 3: Understanding the Format Differences

### DIFY Format (Input)
```json
{
  "model": "bge-reranker-v2-m3",
  "query": "What is the capital of the United States?",
  "documents": [
    "Carson City is the capital city of the American state of Nevada...",
    "The Commonwealth of the Northern Mariana Islands..."
  ],
  "top_n": 3,
  "return_documents": true
}
```

### Hugging Face Format (Internal)
```json
{
  "query": "What is the capital of the United States?",
  "texts": [
    "Carson City is the capital city of the American state of Nevada...",
    "The Commonwealth of the Northern Mariana Islands..."
  ],
  "truncate": true,
  "truncation_direction": "Right",
  "raw_scores": false
}
```

### DIFY Format (Output)
```json
{
  "results": [
    {
      "index": 1,
      "document": {
        "text": "The Commonwealth of the Northern Mariana Islands..."
      },
      "relevance_score": 0.064653486
    },
    {
      "index": 0,
      "document": {
        "text": "Carson City is the capital city of the American state of Nevada..."
      },
      "relevance_score": 0.024845436
    }
  ]
}
```

## Step 4: The Proxy Implementation

The proxy server handles the format conversion and acts as a bridge between DIFY and the Hugging Face server. Here's how it works:

### Key Features

1. **Format Translation**: Converts `documents` to `texts` and restructures the response
2. **Error Handling**: Gracefully handles connection issues and malformed requests
3. **Flexible Response Parsing**: Handles different Hugging Face response formats
4. **Sorting**: Automatically sorts results by relevance score
5. **Logging**: Provides detailed logging for debugging

### Core Conversion Logic

The proxy performs these transformations:

1. **Request Conversion**:
   - Maps DIFY's `documents` field to Hugging Face's `texts`
   - Adds required Hugging Face parameters (`truncate`, `truncation_direction`, `raw_scores`)

2. **Response Conversion**:
   - Wraps results in DIFY's expected `results` array
   - Adds document text back to each result
   - Converts score field names to `relevance_score`
   - Sorts by relevance score (highest first)

### Running the Proxy

The proxy runs on port 8086 and can be started with:

```bash
python dify_proxy.py
```

Output:
```
Native rerank proxy running on http://0.0.0.0:8086/rerank
This proxy converts between Dify format and Hugging Face rerank format
```

## Step 5: Integration with DIFY

### Docker Configuration

In your DIFY setup, configure the rerank model endpoint as:

```
http://host.docker.internal:8086/rerank
```

This points to your proxy server, which will handle the format conversion automatically.

### Testing the Complete Pipeline

Test the full integration:

```bash
curl http://localhost:8086/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-v2-m3",
    "query": "What is the capital of the United States?",
    "documents": [
      "Carson City is the capital city of the American state of Nevada...",
      "The Commonwealth of the Northern Mariana Islands..."
    ],
    "top_n": 3,
    "return_documents": true
  }'
```

## Benefits of This Approach

1. **Performance**: Leverages Rust-based text-embeddings-inference for optimal speed
2. **Flexibility**: Easy to swap different rerank models
3. **Compatibility**: Seamless integration with DIFY's expected format
4. **Debugging**: Comprehensive logging for troubleshooting
5. **Scalability**: Can handle multiple concurrent requests

## Troubleshooting Tips

### Common Issues

1. **Connection Refused**: Ensure the text-embeddings-router is running on port 8085
2. **Format Errors**: Check that your requests match the expected DIFY format
3. **Docker Networking**: Verify `host.docker.internal` resolves correctly in your Docker setup
4. **Model Loading**: Allow time for the model to download and load on first startup

### Debugging

Enable detailed logging by monitoring the proxy output. Each request shows:
- Incoming DIFY request
- Converted Hugging Face request
- Hugging Face response
- Final converted response

## The Result: A Purely Local Rerank Solution

After implementing this proxy setup, you now have:

**Complete Local Control:**
- No dependency on external API providers for rerank functionality
- Full control over model selection and performance tuning
- No data leaving your infrastructure for rerank operations
- Cost predictability without per-request charges

**Production-Ready Performance:**
- Rust-based inference engine for optimal speed
- Support for concurrent requests
- Minimal latency overhead from the proxy layer
- Scalable architecture that can handle production workloads

**Ecosystem Integration:**
- Seamless integration with DIFY's workflow
- Compatible with existing RAG pipelines
- Easy to extend for other platforms with similar format requirements
- Foundation for building more sophisticated retrieval systems

## Conclusion

This setup provides a robust bridge between DIFY and Hugging Face's text-embeddings-inference server, enabling you to leverage powerful rerank models in your DIFY applications. The proxy approach ensures compatibility while maintaining the performance benefits of the underlying Rust-based inference server.

By solving the provider support problem and format incompatibility issues, this solution demonstrates how to achieve what many consider essential for production RAG systems: high-quality reranking that actually works. The modular design makes it easy to adapt for other model formats or add additional features like caching, rate limiting, or authentication as your needs evolve.

## Next Steps

Consider these enhancements:
- Add authentication and rate limiting
- Implement response caching for frequently used queries
- Add metrics and monitoring
- Support for batch processing
- Configuration management for different model endpoints

---

*This setup was tested with BAAI/bge-reranker-v2-m3 model and DIFY platform. The proxy code is available and can be customized for other rerank models or API formats.*
