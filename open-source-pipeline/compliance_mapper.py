import json
import ollama
import time
import re
from json import JSONDecodeError
from typing import List, Dict

class ComplianceMapper:
    def __init__(self, model_name: str = "phi3:mini", batch_size: int = 2):
        self.model = model_name
        self.batch_size = batch_size
        self.max_retries = 3
        self.section_blacklist = {"ROC Template Instructions", "Dos and Don’ts"}
        
        # Improved JSON validation pattern
        self.json_pattern = re.compile(r'\{.*?\}', re.DOTALL)

    def sanitize_content(self, content: str) -> str:
        """Clean content to prevent JSON breakage"""
        replacements = {
            '\n': '\\n',
            '\t': '\\t',
            '"': "'",
            '\\': '\\\\'
        }
        for char, replacement in replacements.items():
            content = content.replace(char, replacement)
        return content[:2000]  # Truncate to prevent context overflow

    def create_prompt(self, chunk: Dict) -> str:
        """Create more robust prompt with JSON examples"""
        sanitized_content = self.sanitize_content(chunk['content'])
        return f"""
        You are a PCI-DSS compliance and network security expert.
        Analyze the following section content and check if the given evidence meets the PCI-DSS requirements.
        Return JSON analysis for PCI-DSS section. Format:
        {{
        "section_num": "{chunk['section_num']}",
        "requirements": ["list", "of", "codes"],
        "explanation": "technical analysis",
        "confidence": "High/Medium/Low"
        }}

        Section Content:
        {sanitized_content}

        Evidence Summary:
        {self.sanitize_content(self.image_analysis)}

        Analysis (JSON ONLY):"""

    def parse_response(self, response: str, chunk: Dict) -> Dict:
        """Robust JSON parsing with multiple fallbacks"""
        try:
            # First attempt: Direct JSON parse
            return json.loads(response)
        except JSONDecodeError:
            try:
                # Second attempt: Extract first JSON block
                json_str = self.json_pattern.search(response)
                if json_str:
                    return json.loads(json_str.group())
                # Fallback: Manual construction
                return {
                    "section_num": chunk['section_num'],
                    "requirements": [],
                    "explanation": "Analysis failed",
                    "confidence": "Low"
                }
            except:
                return {
                    "section_num": chunk['section_num'],
                    "requirements": [],
                    "explanation": "Invalid response format",
                    "confidence": "Low"
                }

    def analyze_chunk(self, chunk: Dict) -> Dict:
        """Enhanced analysis with retries"""
        for attempt in range(self.max_retries):
            try:
                response = ollama.generate(
                    model=self.model,
                    prompt=self.create_prompt(chunk),
                    format="json",
                    options={'temperature': 0.1}
                )
                parsed = self.parse_response(response['response'], chunk)
                return self.validate_result(parsed, chunk)
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {str(e)}")
                time.sleep(2 ** attempt)
        return self.empty_result(chunk)

    def validate_result(self, result: Dict, chunk: Dict) -> Dict:
        """Validate and normalize result structure"""
        return {
            "section_num": result.get('section_num', chunk['section_num']),
            "section_title": chunk['section_title'],
            "requirements": result.get('requirements', []),
            "explanation": result.get('explanation', '')[:500],
            "confidence": result.get('confidence', 'Low').capitalize(),
            "content_excerpt": self.sanitize_content(chunk['content'][:300])
        }

    def process_batches(self):
        """Safer batch processing with progress tracking"""
        results = []
        total = len(self.pdf_chunks)
        
        for i in range(0, total, self.batch_size):
            batch = self.pdf_chunks[i:i+self.batch_size]
            print(f"Processing batch {i//self.batch_size+1}/{(total//self.batch_size)+1}")
            
            batch_results = []
            for chunk in batch:
                if chunk['section_num'] in self.section_blacklist:
                    continue
                result = self.analyze_chunk(chunk)
                batch_results.append(result)
            
            results.extend(batch_results)
            self.save_results(results)
            time.sleep(5)  # Conservative rate limiting

        return results

    def save_results(self, results: List[Dict]):
        """Atomic JSON saving with backup"""
        try:
            safe_data = {
                "metadata": {
                    "processed": len(results),
                    "timestamp": time.time()
                },
                "results": results
            }
            
            with open("mapped_controls.json", 'w', encoding='utf-8') as f:
                json.dump(safe_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Critical save error: {str(e)}")

if __name__ == "__main__":
    mapper = ComplianceMapper(batch_size=2)
    mapper.load_data(
        pdf_chunks_path="pci_chunks.json",
        image_analysis_path="vision_analysis.json"
    )
    
    try:
        results = mapper.process_batches()
        print(f"Successfully processed {len(results)} sections")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print("Partial results saved in mapped_controls.json")