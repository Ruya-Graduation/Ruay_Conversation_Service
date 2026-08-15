# Conversation API - Request/Response Mapping

## Overview

The `/conversation` endpoint now properly separates:
- **System prompt**: Behavior and tone instructions (HOW the assistant should respond)
- **Messages array**: Conversation history
- **Final user message**: Contains the question + artifact data + database passages

---

## Example 1: Simple Request (No History)

### Incoming Request to `/conversation`

```json
POST /conversation
{
  "artifact": {
    "name": "Mask of Tutankhamun",
    "period": "New Kingdom",
    "material": "Gold, Glass, Lapis Lazuli",
    "place_of_discovery": "Tomb of Tutankhamun (KV62), Valley of the Kings"
  },
  "messages": [],
  "question": "Where was the Mask of Tutankhamun discovered?"
}
```

### Outgoing Request to LLM API

```json
POST http://apiaccess.iti.net.eg/api/v1/student/chat
{
  "model_id": "meta.llama3-3-70b-instruct-v1:0",
  "system_prompt": "You are an expert Egyptologist and archaeologist assistant specializing in Ancient Egyptian and Nubian artifacts...",
  "messages": [
    {
      "role": "user",
      "content": "**Artifact Information:**\nName: Mask of Tutankhamun\nPeriod: New Kingdom\nMaterial: Gold, Glass, Lapis Lazuli\nPlace of discovery: Tomb of Tutankhamun (KV62), Valley of the Kings\n\n**Relevant Knowledge from UEE Database:**\n\n[1] Source: Tutankhamun's Burial, pp. 12–15\nThe golden burial mask of Tutankhamun was discovered by Howard Carter in 1925 within the innermost coffin of the pharaoh's tomb in the Valley of the Kings...\n\n[2] Source: Royal Masks, p. 8\nThe mask weighs approximately 10.23 kg and is one of the most recognized artifacts from ancient Egypt...\n\n**Question:**\nWhere was the Mask of Tutankhamun discovered?"
    }
  ],
  "max_tokens": 1024
}
```

---

## Example 2: With Conversation History

### Incoming Request to `/conversation`

```json
POST /conversation
{
  "artifact": {
    "name": "Mask of Tutankhamun",
    "period": "New Kingdom",
    "material": "Gold, Glass, Lapis Lazuli",
    "place_of_discovery": "Tomb of Tutankhamun (KV62), Valley of the Kings"
  },
  "messages": [
    {
      "role": "user",
      "content": "Tell me about the Mask of Tutankhamun."
    },
    {
      "role": "assistant",
      "content": "The Mask of Tutankhamun is one of ancient Egypt's most iconic artifacts. It was discovered in 1925 by Howard Carter..."
    },
    {
      "role": "user",
      "content": "What materials were used to make it?"
    },
    {
      "role": "assistant",
      "content": "The mask is primarily made of gold, with inlays of glass, lapis lazuli, obsidian, carnelian, and other precious materials..."
    }
  ],
  "question": "How much does it weigh?"
}
```

### Outgoing Request to LLM API

```json
POST http://apiaccess.iti.net.eg/api/v1/student/chat
{
  "model_id": "meta.llama3-3-70b-instruct-v1:0",
  "system_prompt": "You are an expert Egyptologist and archaeologist assistant specializing in Ancient Egyptian and Nubian artifacts...",
  "messages": [
    {
      "role": "user",
      "content": "Tell me about the Mask of Tutankhamun."
    },
    {
      "role": "assistant",
      "content": "The Mask of Tutankhamun is one of ancient Egypt's most iconic artifacts. It was discovered in 1925 by Howard Carter..."
    },
    {
      "role": "user",
      "content": "What materials were used to make it?"
    },
    {
      "role": "assistant",
      "content": "The mask is primarily made of gold, with inlays of glass, lapis lazuli, obsidian, carnelian, and other precious materials..."
    },
    {
      "role": "user",
      "content": "**Artifact Information:**\nName: Mask of Tutankhamun\nPeriod: New Kingdom\nMaterial: Gold, Glass, Lapis Lazuli\nPlace of discovery: Tomb of Tutankhamun (KV62), Valley of the Kings\n\n**Relevant Knowledge from UEE Database:**\n\n[1] Source: Royal Masks, p. 8\nThe mask weighs approximately 10.23 kg (22.5 lbs) and stands 54 cm tall...\n\n[2] Source: Tutankhamun's Burial, p. 14\nThe weight of the mask reflects the significant amount of gold used in its construction...\n\n**Question:**\nHow much does it weigh?"
    }
  ],
  "max_tokens": 1024
}
```

---

## Key Points

### System Prompt (Behavior)
- **Content**: Role definition, tone instructions, general guidelines
- **Does NOT contain**: Artifact data, passages, or the user's question
- **Purpose**: Defines HOW the assistant should behave

### Messages Array
- **Content**: Full conversation history + final user message
- **Conversation history**: Passed through exactly as received
- **Final user message**: Contains artifact info + database passages + question

### Final User Message Structure
The last message in the array contains:
1. **Artifact Information** (name, period, material, place of discovery)
2. **Relevant Knowledge from UEE Database** (retrieved chunks with citations)
3. **Question** (the actual user query)

---

## Code Files

- **`app/prompt_builder.py`**: Contains the prompt building logic
  - `build_system_prompt()`: Returns behavior/tone instructions
  - `build_user_message(artifact, question, chunks)`: Builds the final user message content

- **`app/main.py`**: Conversation endpoint
  - Retrieves chunks from vector DB
  - Calls `build_system_prompt()` for behavior instructions
  - Calls `build_user_message()` to create content for final user turn
  - Appends final user message to conversation history
  - Sends to LLM API

- **`app/chat_client.py`**: Handles the actual HTTP call to SBG API
  - Constructs the payload with `model_id`, `system_prompt`, `messages`, `max_tokens`
  - Posts to `{base_url}/student/chat`

---

## Customization

### To change behavior/tone:
Edit `build_system_prompt()` in `app/prompt_builder.py`

### To change how data is presented to the LLM:
Edit `build_user_message()` in `app/prompt_builder.py`

### To change the LLM model:
Update `LLM_MODEL_ID` in `.env`
