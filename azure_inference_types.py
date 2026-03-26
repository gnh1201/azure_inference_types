# Wrapper for OpenAI message types for implementing Azure AI Inference-based integration
# Namhyeon Go <gnh1201@catswords.re.kr>
# https://github.com/gnh1201/azure_inference_types
from typing import List, Union


# ----------------------
# Message Classes
# ----------------------

class BaseMessage:
    # Allowed roles for OpenAI-compatible messages
    VALID_ROLES = ("system", "user", "assistant", "developer")

    def __init__(self, content: str):
        self.role = None
        self.content = content

    def to_dict(self) -> dict:
        # Validate role before conversion
        if self.role not in self.VALID_ROLES:
            raise ValueError("Invalid role: %s" % self.role)

        # Return generic dict representation
        return {
            "role": self.role,
            "content": self.content
        }

    def __repr__(self):
        return "%s(role=%s, content=%s)" % (
            self.__class__.__name__,
            self.role,
            self.content
        )


class SystemMessage(BaseMessage):
    def __init__(self, content: str):
        BaseMessage.__init__(self, content)
        self.role = "system"


class UserMessage(BaseMessage):
    def __init__(self, content: str):
        BaseMessage.__init__(self, content)
        self.role = "user"


class AssistantMessage(BaseMessage):
    def __init__(self, content: str):
        BaseMessage.__init__(self, content)
        self.role = "assistant"


class DeveloperMessage(BaseMessage):
    def __init__(self, content: str):
        BaseMessage.__init__(self, content)
        # Can be mapped to "system" if strict compatibility is needed
        self.role = "developer"


# ----------------------
# Context Class (Core)
# ----------------------

class InferenceContext:
    def __init__(self, messages: List[Union[BaseMessage, dict]] = None):
        self.messages = messages or []

    # Add message (chainable)
    def add(self, message: Union[BaseMessage, dict]):
        self.messages.append(message)
        return self

    # Convenience builders
    def system(self, content: str):
        return self.add(SystemMessage(content))

    def user(self, content: str):
        return self.add(UserMessage(content))

    def assistant(self, content: str):
        return self.add(AssistantMessage(content))

    def developer(self, content: str):
        return self.add(DeveloperMessage(content))

    # Generic dict export
    def to_dict(self) -> List[dict]:
        return [
            m.to_dict() if hasattr(m, "to_dict") else m
            for m in self.messages
        ]

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)

    def __repr__(self):
        return "InferenceContext(messages=%s)" % self.messages
