import requests

class ExplanationTool:

    def __init__(self, model_name="gemma:2b"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"

    def explain(self, prompt):

        try:
            response = requests.post(
                self.url,
                json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 500,   
                    "temperature": 0.2    
                }},
                timeout=777
            )

            response.raise_for_status()
            return response.json().get("response", "").strip()

        except Exception as e:
            return f"Explanation unavailable: {str(e)}"