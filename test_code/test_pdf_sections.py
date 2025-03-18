import json

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
        print(text)