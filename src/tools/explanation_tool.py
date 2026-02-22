import requests

class ExplanationTool:

    def __init__(self, model_name="gemma:2b"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"

    def explain(self, prompt):

        response = requests.post(
            self.url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]