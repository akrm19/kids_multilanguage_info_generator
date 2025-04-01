import sys
import os

from typing import List, Literal

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


from agent.state import GeneratorState, GeneratorInput, GeneratorOutput
from agent.models.validation_result import ValidationResult

# Util functions
def get_ollama(bind_tools: bool = True) -> ChatOllama:
    llm = ChatOllama(
        #model="phi4", temperature=0
        #model="llama3.2", temperature=0
        model="llama3-groq-tool-use:8b", temperature=0
    )
    if bind_tools:
        llm.bind_tools([validate_topic])

    return llm

# Tool functions
def validate_language(state: GeneratorState):
    """Validate the language inputs. This ensures that the language inputs are real languages and that the agent can translate to them."""

    # If there are no target languages, we don't need to validate anything
    if not state.target_languages:
        return {
            "is_valid": True,
            "existing_summary": ""
        }

    system_message = """
    You are an assistant that needs to validate the additional_languages input.

    The additional_languages input will be one or more languages separated by commas. For each language input, validate the following:
     - That each language is a real language, meaning it is not a non-existent language
     - That you can translate to the given language input 

    The output should be a JSON object with the following fields:
     - is_valid: a boolean. Should be True if the input is valid
     - reasons_for_failed_validation: An optional string. If the language input fails validation, this should contain the reasons why the validation is invalid. If the input is valid, this field should be an empty string.
    
    If the additional_languages fail validation:
      - set the is_valid field to False
      - reasons_for_failed_validation output should have the languages that failed validation and the reasons for rejection.

    If additional_languages pass validation, set the is_valid field to True and return an empty string for reasons_for_failed_validation.
    """

    human_message = """additional_languages: {languages}"""

    languages = state.target_languages if state.target_languages else []

    message = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message.format(languages=languages))
    ]

    structured_agent = get_ollama(bind_tools=False).with_structured_output(ValidationResult.model_json_schema())
    result = structured_agent.invoke(message)

    validation_result = result["is_valid"]
    validation_message = result.get("reason_for_failed_validation", "")

    return {
        "is_valid": validation_result,
        "existing_summary": validation_message
    }

def validate_topic(state: GeneratorState):
    """Validate the topic. This ensures that the topic is a real topic and is appropriate for target audience"""

    system_message = """
    You are an assitant that needs to validate the topic input.

    For the topic, validate the following:
    - Validate that the topic is a real topic. Meaning its not a random string of characters or gibberish.
    - Validate that the topic is appropriate for the target audience. The target audience are kids in the age group of 4-6 years old. 
   
    If topic passes validation, set the is_valid field to True and return an empty string for reasons_for_failed_validation.
    If topic fail validation, set the is_valid field to False and return a string containing the reasons for rejection in the reasons_for_failed_validation field of the output.

    The output should be a JSON object with the following fields:
    - is_valid: a boolean indicating whether the input is valid or not
    - reasons_for_failed_validation: a string containing the reasons why the validation is invalid. If the input is valid, this field should be an empty string.
    """

    human_message = """The topic is: {topic}"""

    topic = state.topic

    message = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message.format(topic=topic))
    ]

    structured_agent = get_ollama(bind_tools=False).with_structured_output(ValidationResult.model_json_schema())
    result = structured_agent.invoke(message)

    validation_result = result["is_valid"]
    validation_message = result.get("reason_for_failed_validation", "")

    return {
        "is_valid": validation_result,
        "existing_summary": validation_message
    }

def validation_router(state: GeneratorState) -> Literal["continue", "END"]:
    if state.is_valid:
        return "continue"
    else:
        return "END"

def translation_router(state: GeneratorState) -> Literal["continue", "END"]:
    if state.target_languages:
        return "continue"
    else:
        return "END"

# Graph functions
def generate_summary(state: GeneratorState):
    # Get
    system_message = """You are an assistant that needs to generate a childrens book for a given topic. 

    The generated book should be:
        - The length should be around 2000 words
        - The book should give facts about the topic
        - The book should be engaging for the target audience
        - The book should be written in a simple language that is easy to understand for the target audience
        - The book should explain what the topic is and what is used for

    The target audience is: kids in the age group of 4-6 years old.
    """

    topic = state.topic
    topic_prompt = system_message.format(topic=topic)
    languages = state.target_languages

    message = [
        SystemMessage(content=topic_prompt),
        HumanMessage(content=f"generate a book for the topic: {topic}")
    ]
    
    summary = get_ollama().invoke(message)

    return {"topic": topic, "existing_summary": summary.content}

def generate_translations(state: GeneratorState):
    """Generate translations for the summary in the given languages."""
    languages = state.target_languages
    translated_summaries = {
        "English": state.existing_summary
    }
    
    system_message = """
        You are an assistant that needs to translate the summary into {target_language}.
        Note that the summary is already in English, so you need to translate it into {target_language}.
        The summary is a book for kids in the age group of 4-6 years old.
        The summary should be engaging for the target audience and should be written in a simple language that is easy to understand for the target audience. 

        The summary is: 
        {summary}
        """
    
    for language in languages:
        if language.lower() == "english":
            continue


        message = [
            SystemMessage(content=system_message.format(target_language=language, summary=state.existing_summary)),
            HumanMessage(content=f"Translate the summary")
        ]

        translation = get_ollama().invoke(message)
        translated_summaries[language] = translation.content

    return {
        "output_languages_summaries": translated_summaries
    }

builder = StateGraph(state_schema=GeneratorState, input=GeneratorInput, output=GeneratorOutput)

#Define the nodes
builder.add_node("validate_language", validate_language)
builder.add_node("validate_topic", validate_topic)
builder.add_node("validation_router", validation_router)
builder.add_node("generate_summary", generate_summary)
builder.add_node("translation_router", translation_router)
builder.add_node("generate_translations", generate_translations)

# Define the edges
builder.add_edge(START, "validate_language")
builder.add_conditional_edges("validate_language", validation_router, {
    "continue": "validate_topic",
    "END": END
})
builder.add_conditional_edges("validate_topic", validation_router, {
    "continue": "generate_summary",
    "END": END
})

builder.add_conditional_edges("generate_summary", translation_router, {
    "continue": "generate_translations",
    "END": END
})
builder.add_edge("generate_translations", END)


graph = builder.compile()