from typing import Any


class ai_model:
    def __init__(self, model: str, client: Any):
        self.model = model
        self.client = client    # client: AsyncOpenAI

    async def answer(self, message: str) -> str:
        """Envoie `message` au modèle et renvoie le texte de la réponse."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}],
        )
        return response.choices[0].message.content or ""