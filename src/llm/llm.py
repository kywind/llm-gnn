import os
import sys
import openai
import re


class LLM:
    def __init__(self, args):
        self.args = args
        with open(args.api_key, 'r') as f:
            api_key = f.read().strip()
        openai.api_key = api_key

        self.model = args.llm

        self.prompt_history = []
        self.response_history = []

    def query(self, prompts):
        message = [{"role": "user", "content": prompts}]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=message,
        )
        response_text = response['choices'][0]['message']['content']

        self.prompt_history.append(prompts)
        self.response_history.append(response_text)
        return response_text
