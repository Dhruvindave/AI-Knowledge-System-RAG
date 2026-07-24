from google import genai


class LLMResponseGenerator:

    def __init__(self):
        """
        Google Gemini GenAI configuration.
        """

        API_KEY = 'AQ.Ab8RN6LdZl8pxdRhnCTAMsjCvQuzFv1CRJt3JGnqdkmhUR-GhQ'

        self.client = genai.Client(api_key=API_KEY)

    def prompt_builder(self, context, query):
        """
        Creating the prompt for the Query

        Parameters
        ------------
        context: list[str]
            List containing retrieved text

        query: str
            User asked query


        Returns
        ----------
        prompt: str
            Well structured context-enabled prompt
        """

        prompt = f"""
        You are an AI assistant.

        Answer ONLY using the provided context.

        If the answer is not available,
        say "I don't know."

        Context:
        {context}

        Query:
        {query}
        """

        return prompt

    def generate_response(self, prompt, model="gemini-2.5-flash"):
        response = self.client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response
