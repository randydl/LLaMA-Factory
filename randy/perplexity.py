import math
from openai import OpenAI
from loguru import logger
from tqdm.auto import tqdm


class Perplexity:
    def __init__(self, model, base_url, api_key='EMPTY', temperature=1.0, max_retries=3):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries
        )
        self.model = model
        self.temperature = temperature

    def __call__(self, texts):
        return self.get_ppl(texts)

    def get_logps(self, text):
        res = self.client.completions.create(
            model=self.model,
            prompt=text,
            logprobs=0,
            max_tokens=0,
            temperature=self.temperature,
            echo=True
        )
        logps = res.choices[0].logprobs.token_logprobs
        return [x for x in logps if x is not None]

    def get_ppl(self, texts):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        ppls = []

        for text in tqdm(texts):
            try:
                logps = self.get_logps(text)
                avg_logp = sum(logps) / len(logps)
                ppls.append(math.exp(-avg_logp))
            except Exception as e:
                logger.error(f'API request failed: {e}')
                raise

        return ppls[0] if single else ppls
