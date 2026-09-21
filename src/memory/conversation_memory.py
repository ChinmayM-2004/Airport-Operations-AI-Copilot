"""
Short-Term Conversation Memory

Stores recent user queries and assistant responses for the
current conversation session.

This memory is intentionally lightweight and in-memory.
It is not persistent across application restarts.
"""

from typing import Any, Dict, List, Optional


class ConversationMemory:
    """
    Manages short-term conversational memory.

    The memory stores recent user/assistant exchanges and can
    provide conversation context to the orchestrator.
    """

    def __init__(self, max_messages: int = 10):
        """
        Initialize conversation memory.

        Args:
            max_messages: Maximum number of messages to retain.
        """

        if max_messages <= 0:
            raise ValueError(
                "max_messages must be greater than zero."
            )

        self.max_messages = max_messages
        self.messages: List[Dict[str, str]] = []

    def add_user_message(self, message: str) -> None:
        """
        Add a user message to memory.
        """

        if not isinstance(message, str):
            raise TypeError(
                "User message must be a string."
            )

        message = message.strip()

        if not message:
            return

        self.messages.append(
            {
                "role": "user",
                "content": message,
            }
        )

        self._trim_memory()

    def add_assistant_message(self, message: str) -> None:
        """
        Add an assistant response to memory.
        """

        if not isinstance(message, str):
            raise TypeError(
                "Assistant message must be a string."
            )

        message = message.strip()

        if not message:
            return

        self.messages.append(
            {
                "role": "assistant",
                "content": message,
            }
        )

        self._trim_memory()

    def add_exchange(
        self,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Add a complete user/assistant exchange.
        """

        self.add_user_message(user_message)
        self.add_assistant_message(assistant_message)

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Return a copy of the stored conversation messages.
        """

        return list(self.messages)

    def get_context(self) -> str:
        """
        Convert conversation memory into a readable context string.
        """

        if not self.messages:
            return "No previous conversation context."

        context_lines = []

        for message in self.messages:
            role = message["role"].capitalize()
            content = message["content"]

            context_lines.append(
                f"{role}: {content}"
            )

        return "\n".join(context_lines)

    def get_recent_messages(
        self,
        count: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Return the most recent messages.
        """

        if count <= 0:
            return []

        return self.messages[-count:]

    def clear(self) -> None:
        """
        Clear all conversation memory.
        """

        self.messages.clear()

    def is_empty(self) -> bool:
        """
        Return True when there are no stored messages.
        """

        return len(self.messages) == 0

    def size(self) -> int:
        """
        Return the number of stored messages.
        """

        return len(self.messages)

    def _trim_memory(self) -> None:
        """
        Keep only the most recent messages.

        This prevents the short-term memory from growing
        indefinitely.
        """

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[
                -self.max_messages:
            ]


def create_memory(
    max_messages: int = 10,
) -> ConversationMemory:
    """
    Factory function for creating conversation memory.
    """

    return ConversationMemory(
        max_messages=max_messages
    )


def run_memory_test() -> None:
    """
    Test the short-term conversation memory.
    """

    print("\n" + "=" * 70)
    print("SHORT-TERM CONVERSATION MEMORY TEST")
    print("=" * 70)

    memory = create_memory(
        max_messages=6
    )

    memory.add_user_message(
        "What's happening at SFO?"
    )

    memory.add_assistant_message(
        "SFO is currently operating at low severity."
    )

    memory.add_user_message(
        "What about the surge?"
    )

    memory.add_assistant_message(
        "The current SFO surge multiplier is 1.26x."
    )

    print("\nStored messages:")
    print("-" * 70)

    for message in memory.get_messages():
        print(
            f"{message['role']}: "
            f"{message['content']}"
        )

    print("\nConversation context:")
    print("-" * 70)
    print(memory.get_context())

    print("\nMemory size:")
    print(memory.size())

    print("\nRecent messages:")
    print("-" * 70)

    for message in memory.get_recent_messages(2):
        print(
            f"{message['role']}: "
            f"{message['content']}"
        )

    memory.clear()

    print("\nAfter clear:")
    print(f"Memory empty: {memory.is_empty()}")

    print("\n" + "=" * 70)
    print("MEMORY TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_memory_test()