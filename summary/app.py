from dotenv import load_dotenv
import os
import asyncio
import re
from pathlib import Path
from typing import Optional, Dict
import aiofiles
from openai.types.chat import ChatCompletionMessageParam
from openai_service import OpenAIService

load_dotenv()
openai_service = OpenAIService(api_key=os.getenv('OPENAI_API_KEY'))

def get_result(content: str, tag_name: str) -> Optional[str]:
    """
    Extract the content within XML-like tags.
    
    Args:
        content: The string containing XML-like tags.
        tag_name: The name of the tag to extract content from.
        
    Returns:
        The content within the specified tags, or None if not found.
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else None


async def extract_information(
    title: str, 
    text: str, 
    extraction_type: str, 
    description: str
) -> str:
    """
    Extract information of a specified type from the provided text.
    
    Args:
        title: The title of the article.
        text: The input text from which to extract information.
        extraction_type: The type of information to extract.
        description: The description of the extraction type.
        
    Returns:
        A string containing the extracted information.
    """
    extraction_message: ChatCompletionMessageParam = {
        "content": f"""Extract {extraction_type}: {description} from user message under the context of "{title}".
        Transform the content into clear, structured yet simple bullet points without formatting except links and images.

        Format link like so: - name: brief description with images and links if the original message contains them.

        Keep full accuracy of the original message.""",
        "role": "system"
    }
    
    user_message: ChatCompletionMessageParam = {
        "content": f"Here's the articles you need to extract information from: {text}",
        "role": "user"
    }
    
    response = await openai_service.completion(
        [extraction_message, user_message], 
        'gpt-4o', 
        False
    )
    
    return response.choices[0].message.content or ''

async def draft_summary(
    title: str, 
    article: str, 
    context: str, 
    entities: str, 
    links: str, 
    topics: str, 
    takeaways: str
) -> str:
    """
    Create a comprehensive Polish-language article summary that preserves the original structure and style.
    """
    draft_message: ChatCompletionMessageParam = {
        "content": f"""As a copywriter, create a standalone, fully detailed article based on "{title}" that can be understood without reading the original. Write in markdown format, incorporating all images within the content. The article must:

        Write in Polish, ensuring every crucial element from the original is included while:
        - Stay driven and motivated, ensuring you never miss the details needed to understand the article
        - NEVER reference to the original article
        - Always preserve original headers and subheaders
        - Mimic the original author's writing style, tone, expressions and voice
        - Presenting ALL main points with complete context and explanation
        - Following the original structure and flow without omitting any details
        - Including every topic, subtopic, and insight comprehensively
        - Preserving the author's writing characteristics and perspective
        - Ensuring readers can fully grasp the subject matter without prior knowledge
        - Use title: "{title}" as the title of the article you create. Follow all other headers and subheaders from the original article
        - Include cover image

        Before writing, examine the original to capture:
        * Writing style elements
        * All images, links and vimeo videos from the original article
        * Include examples, quotes and keypoints from the original article
        * Language patterns and tone
        * Rhetorical approaches
        * Argument presentation methods

        Note: You're forbidden to use high-emotional language such as "revolutionary", "innovative", "powerful", "amazing", "game-changer", "breakthrough", "dive in", "delve in", "dive deeper" etc.

        Reference and integrate ALL of the following elements in markdown format:

        <context>{context}</context>
        <entities>{entities}</entities>
        <links>{links}</links>
        <topics>{topics}</topics>
        <key_insights>{takeaways}</key_insights>

        <original_article>{article}</original_article>

        Create the new article within <final_answer></final_answer> tags. The final text must stand alone as a complete work, containing all necessary information, context, and explanations from the original article. No detail should be left unexplained or assumed as prior knowledge.""",
        "role": "user"
    }
    
    response = await openai_service.completion([draft_message], 'gpt-4o', False)
    return response.choices[0].message.content or ''

async def critique_summary(summary: str, article: str, context: str) -> str:
    """
    Analyze the summary for factual accuracy, completeness, and adherence to the original content.
    """
    critique_message: ChatCompletionMessageParam = {
        "content": f"""Analyze the provided compressed version of the article critically, focusing solely on its factual accuracy, structure and comprehensiveness in relation to the given context.

        PRIMARY OBJECTIVE: Compare compressed version against original content with 100% precision requirement.

        VERIFICATION PROTOCOL:
        - Each statement must match source material precisely
        - Every concept requires direct source validation
        - No interpretations or assumptions permitted
        - Markdown formatting must be exactly preserved
        - All technical information must maintain complete accuracy

        CRITICAL EVALUATION POINTS:
        1. Statement-level verification against source
        2. Technical accuracy assessment
        3. Format compliance check
        4. Link and reference validation
        5. Image placement verification
        6. Conceptual completeness check

        <original_article>{article}</original_article>

        <context desc="It may help you to understand the article better.">{context}</context>

        <compressed_version>{summary}</compressed_version>

        RESPONSE REQUIREMENTS:
        - Identify ALL deviations, regardless of scale
        - Report exact location of each discrepancy
        - Provide specific correction requirements
        - Document missing elements precisely
        - Mark any unauthorized additions

        Your task: Execute comprehensive analysis of compressed version against source material. Document every deviation. No exceptions permitted.""",
        "role": "system"
    }
    
    response = await openai_service.completion([critique_message], 'gpt-4o', False)
    return response.choices[0].message.content or ''

async def create_final_summary(
    refined_draft: str, 
    topics: str, 
    takeaways: str, 
    critique: str, 
    context: str
) -> str:
    """
    Generate the final compressed version of the article incorporating feedback from the critique.
    """
    system_message: ChatCompletionMessageParam = {
        "role": "system",
        "content": "You are an expert summarizer. Your task is to create a polished, comprehensive Polish-language summary that addresses all critique points."
    }
    
    summarize_message: ChatCompletionMessageParam = {
        "content": f"""Create a final compressed version of the article that starts with an initial concise overview, then covers all the key topics using available knowledge in a condensed manner, and concludes with essential insights and final remarks.

        Consider the critique provided and address any issues it raises.

        Important: Include relevant links and images from the context in markdown format. Do NOT include any links or images that are not explicitly mentioned in the context.

        Note: You're forbidden to use high-emotional language such as "revolutionary", "innovative", "powerful", "amazing", "game-changer", "breakthrough", "dive in", "delve in", "dive deeper" etc.

        Requirement: Use Polish language.

        Guidelines for compression:
        - Maintain the core message and key points of the original article
        - Always preserve original headers and subheaders
        - Ensure that images, links and videos are present in your response
        - Eliminate redundancies and non-essential details
        - Use concise language and sentence structures
        - Preserve the original article's tone and style in a condensed form

        Provide the final compressed version within <final_answer></final_answer> tags.

        <refined_draft>{refined_draft}</refined_draft>
        <topics>{topics}</topics>
        <key_insights>{takeaways}</key_insights>
        <critique note="This is important, as it was created based on the initial draft of the compressed version. Consider it before you start writing the final compressed version">{critique}</critique>
        <context>{context}</context>

        Let's start.""",
        "role": "user"
    }
    
    try:
        # First try with o1-preview model
        response = await openai_service.completion(
            [system_message, summarize_message], 
            'o1-preview', 
            False
        )
        return response.choices[0].message.content or ''
    except Exception as e:
        print(f"Error with o1-preview model: {e}")
        # Fall back to gpt-4o if o1-preview fails
        try:
            response = await openai_service.completion(
                [system_message, summarize_message], 
                'gpt-4o', 
                False
            )
            return response.choices[0].message.content or ''
        except Exception as e:
            print(f"Error with fallback model: {e}")
            return f"<final_answer>Error generating summary: {e}</final_answer>"

async def generate_detailed_summary():
    """
    Generate a detailed summary by orchestrating all processing steps, including embedding relevant links and images within the content.
    """
    article_path = Path(__file__).parent / 'article.md'
    
    async with aiofiles.open(article_path, 'r', encoding='utf-8') as f:
        article = await f.read()
    
    title = 'AI_devs 3, Lekcja 1, Moduł 1 — Interakcja z dużym modelem językowym'
    
    extraction_types = [
        {
            'key': 'topics', 
            'description': 'Main subjects covered in the article. Focus here on the headers and all specific topics discussed in the article.'
        },
        {
            'key': 'entities', 
            'description': 'Mentioned people, places, or things mentioned in the article. Skip the links and images.'
        },
        {
            'key': 'keywords', 
            'description': 'Key terms and phrases from the content. You can think of them as hashtags that increase searchability of the content for the reader. Example of keyword: OpenAI, Large Language Model, API, Agent, etc.'
        },
        {
            'key': 'links', 
            'description': 'Complete list of the links and images mentioned with their 1-sentence description.'
        },
        {
            'key': 'resources', 
            'description': 'Tools, platforms, resources mentioned in the article. Include context of how the resource can be used, what the problem it solves or any note that helps the reader to understand the context of the resource.'
        },
        {
            'key': 'takeaways', 
            'description': 'Main points and valuable lessons learned. Focus here on the key takeaways from the article that by themselves provide value to the reader (avoid vague and general statements like "it\'s really important" but provide specific examples and context). You may also present the takeaway in broader context of the article.'
        },
        {
            'key': 'context', 
            'description': 'Background information and setting. Focus here on the general context of the article as if you were explaining it to someone who didn\'t read the article.'
        }
    ]
    
    # Run all extractions concurrently
    extraction_tasks = [
        extract_information(title, article, extraction['key'], extraction['description'])
        for extraction in extraction_types
    ]
    
    extraction_results = await asyncio.gather(*extraction_tasks)
    
    extracted_data: Dict[str, str] = {}
    
    # Process results and write to files
    for i, (extraction_type, content) in enumerate(zip(extraction_types, extraction_results)):
        key = extraction_type['key']
        extracted_data[key] = content or f"No {key} found"
        
        output_path = Path(__file__).parent / f"{i + 1}_{key}.md"
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(content or f"No {key} found")
    
    # Draft summary can start as soon as we have context, topics, and takeaways
    draft = await draft_summary(
        title, 
        article, 
        extracted_data['context'], 
        extracted_data['entities'], 
        extracted_data['links'], 
        extracted_data['topics'], 
        extracted_data['takeaways']
    )
    
    # Wait for draft and write it to file
    draft_content = get_result(draft, 'final_answer') or ''
    draft_path = Path(__file__).parent / '8_draft_summary.md'
    async with aiofiles.open(draft_path, 'w', encoding='utf-8') as f:
        await f.write(draft_content)
    
    # Generate critique first
    critique = await critique_summary(
        draft, 
        article, 
        '\n\n'.join(extracted_data.values())
    )
    
    critique_path = Path(__file__).parent / '9_summary_critique.md'
    async with aiofiles.open(critique_path, 'w', encoding='utf-8') as f:
        await f.write(critique)
    
    # Use critique and context in final summary generation
    final_summary = await create_final_summary(
        draft, 
        extracted_data['topics'], 
        extracted_data['takeaways'], 
        critique, 
        extracted_data['context']
    )
    
    final_summary_content = get_result(final_summary, 'final_answer') or ''
    final_path = Path(__file__).parent / '10_final_summary.md'
    async with aiofiles.open(final_path, 'w', encoding='utf-8') as f:
        await f.write(final_summary_content)
    
    print('All steps completed and saved to separate files.')

# Execute the summary generation process
if __name__ == "__main__":
    asyncio.run(generate_detailed_summary())
