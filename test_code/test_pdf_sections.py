import json
import ollama

def load_json_file(filename):
    """Load JSON data from a file."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)
    
file = "pdf_split_sect.json"
    
sections = load_json_file(file)
for idx, (title, text) in enumerate(sections.items()):
    print(f"Section {idx+1}: {title}")
    # print(text)
    if idx == 1:
        #print(text)
        prompt=f"Summarize and explain the given section: {text}"
        response = ollama.generate(
            model="deepseek-r1:7b",
            prompt=prompt,
            options={'temperature': 0.1}
        )
        break

