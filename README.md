# azure_inference_types
Message adapters for Azure AI Inference–based applications

## Overview

This module provides message adapters that bridge Azure AI Inference–based applications and the OpenAI SDK.

It enables seamless migration by allowing you to reuse familiar message structures while converting them into a format compatible with the OpenAI API.

## Before (Azure AI Inference style)

```python
messages = [
    AssistantMessage(content=message1),
    SystemMessage(content=message2),
    UserMessage(content=message3),
]

response = client.complete(
    messages=messages,
    max_tokens=4096,
    temperature=0,
    top_p=0.1,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    model=model_name
)
```

## After (Using message adapters)

```python
from .azure_inference_types import InferenceContext, SystemMessage, UserMessage, DeveloperMessage, AssistantMessage

messages = InferenceContext([
    AssistantMessage(content=message1),
    SystemMessage(content=message2),
    UserMessage(content=message3),
])

response = client.chat.completions.create(
    messages=messages.to_dict(),
    max_tokens=4096,
    temperature=0,
    top_p=0.1,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    model=model_name
)
```

## Key Features

* Message adapters for Azure AI Inference → OpenAI SDK compatibility
* Minimal changes required for existing codebases
* Clear and structured message construction
* Built-in conversion via `to_dict()` for OpenAI API requests
* Extensible design for additional backends or adapters

## How It Works

* `InferenceContext` acts as a container for messages
* Each message type (`SystemMessage`, `UserMessage`, etc.) represents a role-based structure
* `to_dict()` converts the entire message set into the format required by the OpenAI API

## Notes

* The API call changes from `client.complete()` to `client.chat.completions.create()`
* The adapter layer ensures compatibility without requiring a full rewrite of message handling logic
