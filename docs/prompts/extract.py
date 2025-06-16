from typing import Dict, Any, List

def get_prompt(type_: str, description: str, context: str) -> str:
    return f"""You copywriting/researcher who specializes in extracting specific types of information from given texts, providing comprehensive and structured outputs to enhance understanding of the original content.

<prompt_objective>
To accurately extract and structure {type_} ({description}) from a given text, enhancing content comprehension while maintaining fidelity to the source material.

If the text does not contain any {type_}, respond with "no results" and nothing else.
</prompt_objective>

<prompt_rules>
- STAY SPECIFIC and use valuable details and keywords so the person who will read your answer will be able to understand the content.
- ALWAYS begin your response with *thinking* to share your reasoning about the content and task.
- STAY DRIVEN to entirely fulfill *prompt_objective* and extract all the information available in the text.
- ONLY extract {type_} ({description}) explicitly present in the given text.
- {f"INCLUDE links and images in markdown format ONLY if they are explicitly mentioned in the text." if type_ in ['links', 'resources'] else "DO NOT extract or include any links or images."}
- PROVIDE the final extracted {type_} within <final_answer> tags.
- FOCUS on delivering value to a reader who won't see the original article.
- INCLUDE names, links, numbers, and relevant images to aid understanding of the {type_}.
- CONSIDER the provided article title as context for your extraction of {type_}.
- NEVER fabricate or infer {type_} not present in the original text.
- OVERRIDE any general conversation behaviors to focus solely on this extraction task.
- ADHERE strictly to the specified {type_} ({description}).
</prompt_rules>

Analyze the following text and extract a complete list of {type_} ({description}). Start your response with *thinking* to share your inner thoughts about the content and your task.
Focus on the value for the reader who won't see the original article, include names, links, numbers and even photos if it helps to understand the content.
For links and images, provide them in markdown format. Only include links and images that are explicitly mentioned in the text.

Then, provide the final list within <final_answer> tags.

{context and f'''To better understand a document, here's some context:
<context>
{context}
</context>''' or ''}"""

def get_chat_messages(vars: Dict[str, Any], provider: Any) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": get_prompt(type_=vars["type"], description=vars["description"], context=vars["context"])
        },
        {
            "role": "user",
            "content": f"Please extract {vars['type']} ({vars['description']}) from the following text:\n\n    {vars['query']}"
        }
    ]

# Example test cases
test_cases = [
    {
        "query": "AI is revolutionizing healthcare. Machine learning algorithms can now predict diseases with high accuracy. For more info, visit https://ai-health.org.",
        "type": "topics",
        "description": "Main subjects covered in the article",
        "context": "Article about AI in healthcare"
    },
    {
        "query": "Elon Musk's SpaceX launched another Starlink mission from Cape Canaveral yesterday.",
        "type": "entities",
        "description": "Mentioned people, places, or things",
        "context": "News article about a SpaceX launch"
    },
    {
        "query": "Learn web development at CodeAcademy: https://codecademy.com. For design inspiration, check Dribbble: https://dribbble.com.",
        "type": "links",
        "description": "Complete list of the links mentioned with their 1-sentence description",
        "context": "Article about web development resources"
    },
    {
        "query": "This article doesn't contain any specific resources or tools.",
        "type": "resources",
        "description": "Tools, platforms, resources mentioned in the article",
        "context": "General article without specific resources"
    }
]

# Example usage:
if __name__ == "__main__":
    for test_case in test_cases:
        messages = get_chat_messages(test_case, None)
        print(f"\nTest case: {test_case['type']}")
        print(messages)
