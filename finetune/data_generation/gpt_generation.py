from datetime import datetime, timedelta
import logging
import time
import json
import os

# from azure.identity import DefaultAzureCredential
import openai
from tqdm import tqdm

def load_prompt():
    prompt = ''
    with open('prompt_new.txt') as f:
        for line in f:
            prompt = prompt + line.strip() + '\n'
    return prompt

class GPT4:
    def complete(
        self, prompt, max_tokens=5120, temperature=1.0, logprobs=None, n=1,
        frequency_penalty=0, presence_penalty=0, stop=None, rstrip=False,
        partition_id=None, **kwargs
    ) -> str:
        openai.api_base = ""
        openai.api_version = ''
        openai.api_key = ""
        openai.api_type = "azure"
        self.deployment_id = "gpt-4"
        if rstrip:
            # Remove heading whitespaces improves generation stability. It is
            # disabled by default to keep consistency.
            prompt = prompt.rstrip()
        retry_interval_exp = 1 
        while True:
            try:
                # Partition ID should only be set explictly when requests can
                # utilize server side cache. Caching helps reduce computational
                # cost for prompts with similar content. You may manually
                # assign the same partition ID to similar prompts.
                if not partition_id:
                    # Requests will be routed in round robin by default.
                    partition_id = f"sumscience-{datetime.now()}"
                response = openai.ChatCompletion.create(
                    engine=self.deployment_id,
                    messages=[
                        {"role": "system", "content": "You are an expert in chemistry and a helpful assistant."},
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    # top_p=1,  # Not recommended to change with temperature
                    # frequency_penalty=frequency_penalty,
                    # presence_penalty=presence_penalty,
                    # logprobs=logprobs,
                    # n=n,
                    # stop=stop,
                    headers={"partition-id": partition_id},
                    # **kwargs,
                )
                return response["choices"][0]["message"]["content"]
            except openai.error.RateLimitError as e:
                # NOTE: openai.error.RateLimitError: Requests to the
                # Deployments_Completion Operation under OpenAI API have
                # exceeded rate limit of your current OpenAI S0 pricing tier.
                # Please retry after 7 seconds. Please contact Azure support
                # service if you would like to further increase the default rate
                # limit.
                logging.warning("OpenAI rate limit error. Retry")
                # Expontial backoff
                time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
                retry_interval_exp += 1
            except openai.error.APIConnectionError as e:
                logging.warning("OpenAI API connection error. Retry")
                time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
                retry_interval_exp += 1
            except openai.error.Timeout as e:
                logging.warning("OpenAI timeout error. Sleep then retry.")
                time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
                retry_interval_exp += 1
def main():
    gpt4 = GPT4()
    prompt = load_prompt()
    n = 1000
    print(prompt)
    for i in tqdm(range(n)):
        response = gpt4.complete(prompt=prompt)
        print(i)
        print(response)
        cur = {}
        cur['gpt_output'] = response
        with open('generated_problems_new.jsonl', 'a') as f:
            print(json.dumps(cur, ensure_ascii=False), file=f)
            
if __name__ == "__main__":
    main()
