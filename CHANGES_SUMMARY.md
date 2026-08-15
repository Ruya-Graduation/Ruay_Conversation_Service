# Changes Summary - Conversation API Restructuring

## What Changed

The `/conversation` endpoint has been restructured to properly separate **system prompt** (behavior) from **user message content** (data).

---

## Files Modified

### 1. `app/prompt_builder.py` ✅ REWRITTEN

**Old structure:**
- `build_system_prompt(artifact, question, chunks)` → Returned everything mixed together

**New structure:**
- `build_system_prompt()` → Returns **behavior/tone instructions only**
- `build_user_message(artifact, question, chunks)` → Returns **question + artifact + passages**

#### New `build_system_prompt()` (No Parameters)
```python
def build_system_prompt() -> str:
    """Defines HOW the assistant should behave (tone, style, guidelines)"""
    return (
        "You are an expert Egyptologist and archaeologist assistant "
        "specializing in Ancient Egyptian and Nubian artifacts...\n\n"
        "Your role:\n"
        "- Provide scholarly, precise answers...\n"
        "- Cite passage numbers when drawing from sources\n"
        "- Be honest when information is insufficient...\n"
        # ... behavior instructions only
    )
```

#### New `build_user_message()` 
```python
def build_user_message(artifact, question, chunks) -> str:
    """Builds the final user message containing data"""
    # Returns formatted string with:
    # - Artifact Information (name, period, material, place)
    # - Relevant Knowledge from UEE Database (retrieved chunks)
    # - Question (the actual user query)
```

---

### 2. `app/main.py` ✅ UPDATED

**Changes in the `/conversation` endpoint:**

#### Before:
```python
# Step 4: Build system prompt with everything
system_prompt = build_system_prompt(artifact, question, chunks)

# Step 5: Build messages
llm_messages = [... conversation history ...]
llm_messages.append({"role": "user", "content": body.question})  # Just the question

# Step 6: Call LLM
call_chat(system_prompt=system_prompt, messages=llm_messages, ...)
```

#### After:
```python
# Step 4: Build system prompt (behavior only)
system_prompt = build_system_prompt()

# Step 5: Build user message content (question + artifact + passages)
user_message_content = build_user_message(artifact, question, chunks)

# Step 6: Build messages
llm_messages = [... conversation history ...]
llm_messages.append({"role": "user", "content": user_message_content})  # Rich content

# Step 7: Call LLM
call_chat(system_prompt=system_prompt, messages=llm_messages, ...)
```

#### Updated imports:
```python
from app.prompt_builder import build_query_text, build_system_prompt, build_user_message
```

---

### 3. `app/chat_client.py` ✅ NO CHANGES NEEDED

The chat client already supports the correct format. No modifications required.

---

## Request Flow (New)

### 1. User sends request:
```json
{
  "artifact": { "name": "Mask of Tutankhamun", ... },
  "messages": [
    {"role": "user", "content": "Tell me about this artifact."},
    {"role": "assistant", "content": "The mask is..."}
  ],
  "question": "What materials were used?"
}
```

### 2. API processes:
1. **Embed query** (artifact + question)
2. **Retrieve chunks** from MongoDB vector DB
3. **Build system prompt** (behavior/tone) ← New: no parameters
4. **Build user message** (artifact + chunks + question) ← New function
5. **Assemble messages array**:
   - Previous conversation history
   - + New user message with all data
6. **Call LLM API**

### 3. LLM receives:
```json
{
  "model_id": "meta.llama3-3-70b-instruct-v1:0",
  "system_prompt": "You are an expert Egyptologist...",  // Behavior only
  "messages": [
    {"role": "user", "content": "Tell me about this artifact."},
    {"role": "assistant", "content": "The mask is..."},
    {"role": "user", "content": "**Artifact Information:**\nName: Mask...\n\n**Relevant Knowledge:**\n[1] Source: ...\n\n**Question:**\nWhat materials were used?"}
  ],
  "max_tokens": 1024
}
```

---

## Key Improvements

### ✅ Proper Separation of Concerns
- **System prompt**: Defines assistant behavior (HOW to answer)
- **User message**: Contains all the data (WHAT to answer about)

### ✅ Conversation History Preserved
- Previous messages pass through unchanged
- Only the final user message contains the enriched context

### ✅ RAG Context in User Message
- Artifact metadata clearly labeled
- Retrieved knowledge passages numbered for citation
- User's actual question at the end

### ✅ Cleaner Architecture
- Each function has a single responsibility
- Easy to customize behavior vs. data presentation separately
- Follows the standard chat API pattern

---

## Configuration

No changes to `.env` file needed. Existing settings work as-is:

```bash
# LLM Configuration (unchanged)
LLM_MODEL_ID=meta.llama3-3-70b-instruct-v1:0
LLM_MAX_TOKENS=1024

# SBG API (unchanged)
SBG_BASE_URL=http://apiaccess.iti.net.eg/api/v1
SBG_API_KEY=sbg_...
```

---

## Testing

The API is still running. Test with:

```bash
POST http://localhost:8000/conversation
{
  "artifact": {
    "name": "Test Artifact",
    "period": "Middle Kingdom"
  },
  "messages": [],
  "question": "Tell me about this artifact."
}
```

---

## Documentation

Created new documentation files:
- ✅ `CONVERSATION_API_MAPPING.md` - Detailed request/response examples
- ✅ `CHANGES_SUMMARY.md` - This file
- ✅ Updated code comments in `prompt_builder.py` and `main.py`

---

## Backward Compatibility

⚠️ **Breaking Change**: The internal structure changed, but the **API contract** (request/response format) remains the same.

- External clients don't need to change
- The `/conversation` endpoint still accepts the same request format
- The response format is unchanged

Only the **internal** mapping to the LLM API was restructured.
