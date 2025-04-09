import json
from src.agent.graph import graph
from src.agent.state import GeneratorState, GeneratorInput, GeneratorOutput

topic = input("Enter a topic: ").strip()
language_input = input("Enter one or more languages separated by commas: ").strip()

languages = [lang.strip() for lang in language_input.split(",") if lang.strip()]

sumary_topic = GeneratorInput(topic=topic, target_languages=languages) #(topic="serial killers") #, target_languages={"English"})

summmary = graph.invoke(sumary_topic)

output_file = "output.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(summmary, f, indent=4, ensure_ascii=False)

    # with open('output2.json', "r") as f:
    #     f.write(json.dumps(summmary, indent=4))
#print(f"\n\nSummary:\n{summmay}")
print(f"\n\nSummary:\n")
print(summmary)
#print(json.dumps(summmay, indent=4))
#print(f"\n\nTool calls: {summmay.tool_calls}")
#print(f"summary outputs: {summmay["summary_outputs"]}")
#print(summmay)