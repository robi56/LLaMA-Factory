"""
Chat utilities for turn detector training.
Adapted from your original chat_utils.py for LLaMA Factory compatibility.
"""

def normalized_chat(conversation):
    """
    Normalize conversation format for consistent processing.
    
    Args:
        conversation: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        List of normalized message dictionaries
    """
    normalized = []
    for message in conversation:
        if isinstance(message, dict) and 'role' in message and 'content' in message:
            # Ensure role is lowercase and content is string
            normalized.append({
                'role': message['role'].lower(),
                'content': str(message['content'])
            })
    return normalized


def format_chat_chatml(chat_context):
    """
    Format the chat into the ChatML template format.
    This matches your original Livekit ChatML template.
    
    Args:
        chat_context: List of normalized message dictionaries
        
    Returns:
        Formatted chat string
    """
    formatted_parts = []
    
    for message in chat_context:
        role = message['role']
        content = message['content']
        
        # Map roles to ChatML format
        if role == 'user':
            role_tag = '<|user|>'
        elif role == 'assistant':
            role_tag = '<|assistant|>'
        elif role == 'system':
            role_tag = '<|system|>'
        else:
            role_tag = f'<|{role}|>'
        
        # Format as ChatML
        formatted_parts.append(f'<|im_start|>{role_tag}{content}<|im_end|>')
    
    return ''.join(formatted_parts)


def format_chat_sharegpt(chat_context):
    """
    Format the chat into ShareGPT format for LLaMA Factory.
    
    Args:
        chat_context: List of normalized message dictionaries
        
    Returns:
        List of ShareGPT formatted messages
    """
    formatted_messages = []
    
    for message in chat_context:
        role = message['role']
        content = message['content']
        
        # Map roles to ShareGPT format
        if role == 'user':
            sharegpt_role = 'user'
        elif role == 'assistant':
            sharegpt_role = 'assistant'
        elif role == 'system':
            sharegpt_role = 'system'
        else:
            sharegpt_role = role
        
        formatted_messages.append({
            'role': sharegpt_role,
            'content': content
        })
    
    return formatted_messages


def convert_to_sharegpt_format(conversations):
    """
    Convert conversations to ShareGPT format for LLaMA Factory.
    
    Args:
        conversations: List of conversation dictionaries
        
    Returns:
        List of ShareGPT formatted conversations
    """
    sharegpt_conversations = []
    
    for conversation in conversations:
        if 'conversations' in conversation:
            # Normalize the conversation
            normalized = normalized_chat(conversation['conversations'])
            # Format as ShareGPT
            formatted = format_chat_sharegpt(normalized)
            
            sharegpt_conversations.append({
                'conversations': formatted
            })
    
    return sharegpt_conversations


# Example usage and testing
if __name__ == "__main__":
    # Test conversation
    test_conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
        {"role": "user", "content": "Can you help me with turn detection?"},
        {"role": "assistant", "content": "Of course! Turn detection is about identifying when a speaker changes in a conversation. What specific aspect would you like help with?"}
    ]
    
    print("Original conversation:")
    print(test_conversation)
    print("\nNormalized conversation:")
    normalized = normalized_chat(test_conversation)
    print(normalized)
    print("\nChatML formatted:")
    chatml = format_chat_chatml(normalized)
    print(chatml)
    print("\nShareGPT formatted:")
    sharegpt = format_chat_sharegpt(normalized)
    print(sharegpt)
