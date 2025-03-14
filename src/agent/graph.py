from typing import List, Literal

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from src.agent.state import GeneratorState, GeneratorInput, GeneratorOutput
from src.agent.models.validation_result import ValidationResult

# Util functions
def get_ollama(bind_tools: bool = True) -> ChatOllama:
    llm = ChatOllama(
        #model="phi4", temperature=0
        model="llama3.2", temperature=0
    )
    if bind_tools:
        llm.bind_tools([validate_inputs])

    return llm

# Tool functions
def validate_inputs(state: GeneratorState):
    """Validate the topic. This ensures that the topic is appropriate for target audience and languages are supported"""

    system_message = """You are an assitant that needs to validate that the topic is appropriate for the target audience and that the languages are supported. 
    The target audience are kids in the age group of 4-6 years old. The topic should be appropriate for this age group, if not, you should reject the topic and provide a reason.
    There is also an optional "additional_languages" parameter that can be passed.

    If additional_languages is not empty, validate them by ensuring that they are supported and that you can translate to the language. 
    If a language is not supported, you should reject the language and specify the invalid language.
    If no additional_languages are provided, you can assume that the target language is English.

    If the topic does not pass validation, provide the reasons why the topic is invalid in the validation results
    """

    human_message = """The topic is: {topic}
    The additional_languages are: {languages}"""

    topic = state.topic
    languages = state.target_languages

    message = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message.format(topic=topic, languages=languages))
    ]
    agent = get_ollama(bind_tools=False)
    structured_agent = get_ollama(bind_tools=False).with_structured_output(ValidationResult.model_json_schema())
    result = structured_agent.invoke(message)
    print(f"Validation result: {result}")

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
    

# Graph functions
def generate_summary(state: GeneratorState):
    # Get
    system_message = """You are an assistant that needs to generate a childrens book for a given topic. 

    The generated book should be:
        - The length should be around 1000 words
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

    return {"topic": topic, "summary_outputs": summary.content}

def generate_translations(summary: str, languages: List[str]) -> List[str]:
    return (True, "This is a translation")

builder = StateGraph(state_schema=GeneratorState, input=GeneratorInput, output=GeneratorOutput)
builder.add_node("validate_inputs", validate_inputs)
builder.add_node("validation_router", validation_router)
builder.add_node("generate_summary", generate_summary)

builder.add_edge(START, "validate_inputs")
#builder.add_conditional_edges("validate_inputs", validation_router)
builder.add_conditional_edges("validate_inputs", validation_router, {
    "continue": "generate_summary",
    "END": END
})
builder.add_edge("generate_summary", END)

graph = builder.compile()