from src.agent.graph import graph
from src.agent.state import GeneratorState, GeneratorInput, GeneratorOutput

topic = input("Enter a topic: ").strip()
language_input = input("Enter one or more languages separated by commas: ").strip()
languages = [lang.strip() for lang in language_input.split(",")]

sumary_topic = GeneratorInput(topic=topic, target_languages=languages) #(topic="serial killers") #, target_languages={"English"})

summmay = graph.invoke(sumary_topic)

print(f"\n\nSummary:\n{summmay}")
#print(f"\n\nTool calls: {summmay.tool_calls}")
#print(f"summary outputs: {summmay["summary_outputs"]}")
#print(summmay)