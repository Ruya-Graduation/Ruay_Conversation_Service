# Conversation API - Response Format

## Overview

The `/conversation` endpoint now returns structured response with **separated LLM metadata fields**, making it easy for the .NET server to access specific information.

---

## Response Structure

### Full Response Example

```json
{
  "answer": "The Mask of Tutankhamun was discovered in the Tomb of Tutankhamun, also known as KV62, which is located in the Valley of the Kings, Luxor.",
  "retrieved_chunks": 10,
  "model_id": "meta.llama3-3-70b-instruct-v1:0",
  "sources": [
    {
      "chunk_id": "eScholarship_UC_item_3f23c0q9-1234:chunk-0011",
      "title": "Tutankhamun's Burial",
      "text": "The golden burial mask of Tutankhamun was discovered...",
      "page_start": 12,
      "page_end": 15,
      "score": 0.892
    },
    // ... more chunks
  ],
  "request_id": "6193cfc6-f697-45e2-909c-3b26889b5f50",
  "region": "us-east-2",
  "usage": {
    "input_tokens": 1541,
    "output_tokens": 38,
    "total_tokens": 1579,
    "stop_reason": "end_turn",
    "budget_state": "ok",
    "fallback_used": false
  },
  "estimated_cost_usd": "0.001080",
  "actual_cost_usd": "0.001137",
  "status": "active",
  "llm_response_metadata": {
    // Any additional fields from LLM response
  }
}
```

---

## Response Fields

### Core Fields (Always Present)

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | **The actual answer text from the LLM** (from `output_text` in LLM response) |
| `retrieved_chunks` | integer | Number of knowledge chunks retrieved from the database |
| `model_id` | string | LLM model identifier that generated the answer |
| `sources` | array | List of retrieved chunks used as context (for transparency) |

### LLM Metadata Fields (From SBG API Response)

| Field | Type | Optional | Description |
|-------|------|----------|-------------|
| `request_id` | string | Yes | Unique identifier for this LLM request |
| `region` | string | Yes | Deployment region (e.g., "us-east-2") |
| `usage` | object | Yes | Token usage statistics (see below) |
| `estimated_cost_usd` | string | Yes | Estimated cost in USD |
| `actual_cost_usd` | string | Yes | Actual cost in USD |
| `status` | string | Yes | Request status (e.g., "active", "completed") |
| `llm_response_metadata` | object | Yes | Any additional metadata from LLM response |

### Usage Object Structure

```json
{
  "input_tokens": 1541,
  "output_tokens": 38,
  "total_tokens": 1579,
  "stop_reason": "end_turn",
  "budget_state": "ok",
  "fallback_used": false
}
```

---

## Sources Array Structure

Each source chunk contains:

```json
{
  "chunk_id": "eScholarship_UC_item_3f23c0q9-1234:chunk-0011",
  "title": "Tutankhamun's Burial",
  "text": "The golden burial mask of Tutankhamun was discovered by Howard Carter in 1925...",
  "page_start": 12,
  "page_end": 15,
  "score": 0.892
}
```

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | string | Unique identifier for the chunk |
| `title` | string | Title of the source article |
| `text` | string | The actual text content of the chunk |
| `page_start` | integer | Starting page number in the source |
| `page_end` | integer | Ending page number in the source |
| `score` | float | Similarity score from vector search (0-1) |

---

## Usage in .NET

### Deserialize to C# Class

```csharp
public class ConversationResponse
{
    [JsonPropertyName("answer")]
    public string Answer { get; set; }
    
    [JsonPropertyName("retrieved_chunks")]
    public int RetrievedChunks { get; set; }
    
    [JsonPropertyName("model_id")]
    public string ModelId { get; set; }
    
    [JsonPropertyName("sources")]
    public List<SourceChunk> Sources { get; set; }
    
    // LLM Metadata
    [JsonPropertyName("request_id")]
    public string RequestId { get; set; }
    
    [JsonPropertyName("region")]
    public string Region { get; set; }
    
    [JsonPropertyName("usage")]
    public UsageStats Usage { get; set; }
    
    [JsonPropertyName("estimated_cost_usd")]
    public string EstimatedCostUsd { get; set; }
    
    [JsonPropertyName("actual_cost_usd")]
    public string ActualCostUsd { get; set; }
    
    [JsonPropertyName("status")]
    public string Status { get; set; }
    
    [JsonPropertyName("llm_response_metadata")]
    public Dictionary<string, object> LlmResponseMetadata { get; set; }
}

public class SourceChunk
{
    [JsonPropertyName("chunk_id")]
    public string ChunkId { get; set; }
    
    [JsonPropertyName("title")]
    public string Title { get; set; }
    
    [JsonPropertyName("text")]
    public string Text { get; set; }
    
    [JsonPropertyName("page_start")]
    public int? PageStart { get; set; }
    
    [JsonPropertyName("page_end")]
    public int? PageEnd { get; set; }
    
    [JsonPropertyName("score")]
    public double? Score { get; set; }
}

public class UsageStats
{
    [JsonPropertyName("input_tokens")]
    public int InputTokens { get; set; }
    
    [JsonPropertyName("output_tokens")]
    public int OutputTokens { get; set; }
    
    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; set; }
    
    [JsonPropertyName("stop_reason")]
    public string StopReason { get; set; }
    
    [JsonPropertyName("budget_state")]
    public string BudgetState { get; set; }
    
    [JsonPropertyName("fallback_used")]
    public bool FallbackUsed { get; set; }
}
```

### Example Usage in .NET

```csharp
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;

public async Task<ConversationResponse> GetConversationResponse(ConversationRequest request)
{
    var httpClient = new HttpClient();
    var apiUrl = "http://localhost:8000/conversation";
    
    var json = JsonSerializer.Serialize(request);
    var content = new StringContent(json, Encoding.UTF8, "application/json");
    
    var response = await httpClient.PostAsync(apiUrl, content);
    response.EnsureSuccessStatusCode();
    
    var responseJson = await response.Content.ReadAsStringAsync();
    var conversationResponse = JsonSerializer.Deserialize<ConversationResponse>(responseJson);
    
    // Access specific fields
    Console.WriteLine($"Answer: {conversationResponse.Answer}");
    Console.WriteLine($"Request ID: {conversationResponse.RequestId}");
    Console.WriteLine($"Cost: ${conversationResponse.ActualCostUsd}");
    Console.WriteLine($"Tokens Used: {conversationResponse.Usage.TotalTokens}");
    Console.WriteLine($"Sources Retrieved: {conversationResponse.RetrievedChunks}");
    
    return conversationResponse;
}
```

---

## Backward Compatibility

### Before (Old Format)
```json
{
  "answer": "The Mask of Tutankhamun was discovered...",
  "retrieved_chunks": 10,
  "model_id": "meta.llama3-3-70b-instruct-v1:0",
  "sources": [...]
}
```

### After (New Format)
```json
{
  "answer": "The Mask of Tutankhamun was discovered...",
  "retrieved_chunks": 10,
  "model_id": "meta.llama3-3-70b-instruct-v1:0",
  "sources": [...],
  "request_id": "6193cfc6-f697-45e2-909c-3b26889b5f50",
  "region": "us-east-2",
  "usage": {...},
  "estimated_cost_usd": "0.001080",
  "actual_cost_usd": "0.001137",
  "status": "active"
}
```

**✅ Backward Compatible**: Old clients that only read `answer`, `retrieved_chunks`, `model_id`, and `sources` will continue to work. New clients can access the additional metadata fields.

---

## Benefits for .NET Server

1. **Cost Tracking**: Access `estimated_cost_usd` and `actual_cost_usd` for billing
2. **Usage Monitoring**: Track token usage via the `usage` object
3. **Request Tracing**: Use `request_id` for debugging and logging
4. **Budget Management**: Check `budget_state` in usage to monitor quota
5. **Transparency**: Access all source chunks used in the response
6. **Metadata Access**: Any additional LLM response data in `llm_response_metadata`

---

## Example Request/Response

### Request
```bash
POST http://localhost:8000/conversation
Content-Type: application/json

{
  "artifact": {
    "name": "Mask of Tutankhamun",
    "period": "New Kingdom"
  },
  "messages": [],
  "question": "Where was this discovered?"
}
```

### Response
```json
{
  "answer": "The Mask of Tutankhamun was discovered in the Tomb of Tutankhamun (KV62) in the Valley of the Kings, Luxor, Egypt.",
  "retrieved_chunks": 3,
  "model_id": "meta.llama3-3-70b-instruct-v1:0",
  "sources": [
    {
      "chunk_id": "tutankhamun_burial:chunk-001",
      "title": "Tutankhamun's Burial",
      "text": "Howard Carter discovered the tomb in 1922...",
      "page_start": 5,
      "page_end": 6,
      "score": 0.95
    }
  ],
  "request_id": "abc-123-def-456",
  "region": "us-east-2",
  "usage": {
    "input_tokens": 450,
    "output_tokens": 28,
    "total_tokens": 478,
    "stop_reason": "end_turn",
    "budget_state": "ok",
    "fallback_used": false
  },
  "estimated_cost_usd": "0.000340",
  "actual_cost_usd": "0.000345",
  "status": "active"
}
```
