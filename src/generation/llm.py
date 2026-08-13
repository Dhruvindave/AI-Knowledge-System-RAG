from google import genai


class LLMResponseGenerator:

    def __init__(self):
        """
        Google Gemini GenAI configuration.
        """

        API_KEY = 'AQ.Ab8RN6LdZl8pxdRhnCTAMsjCvQuzFv1CRJt3JGnqdkmhUR-GhQ'

        self.client = genai.Client(api_key=API_KEY)

    # def prompt_builder(self, context, query):
    #     """
    #     Creating the prompt for the Query

    #     Parameters
    #     ------------
    #     context: list[str]
    #         List containing retrieved text

    #     query: str
    #         User asked query


    #     Returns
    #     ----------
    #     prompt: str
    #         Well structured context-enabled prompt
    #     """

    #     prompt = f"""
    #     You are an AI assistant.

    #     Answer ONLY using the provided context.

    #     If the answer is not available,
    #     say "I don't know."

    #     Context:
    #     {context}

    #     Query:
    #     {query}
    #     """

    #     return prompt

    def build_context(self, retrieved_chunks):
        """
        retrieved_chunks: list of dicts from store.search(), e.g.
            {"title": ..., "breadcrumb": ..., "content": ..., "distance": ...}
        """
        blocks = []

        for i, chunk in enumerate(retrieved_chunks, start=1):
            blocks.append(
                f"[Source {i}] {chunk['breadcrumb']}\n{chunk['content']}"
            )

        return "\n\n---\n\n".join(blocks)

    def build_prompt(self, user_question, context):
        return f"""You are answering a question using only the context provided below.
If the answer isn't contained in the context, say so clearly instead of guessing.
When relevant, mention which source(s) your answer is based on.

Context:
{context}

Question: {user_question}

Answer:"""

    def generate_response(self, prompt, model="gemini-2.5-flash"):
        response = self.client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response

