from datetime import datetime, timedelta
import logging
import time
import json
import random
import argparse
import os
# from azure.identity import DefaultAzureCredential
import openai
from tqdm import tqdm


def load_prompt(file):
    prompt = ''
    with open(file) as f:
        for line in f:
            prompt = prompt + line.strip() + '\n'
    return prompt

class GPT4:

    def __init__(self, max_tokens=1024, temperature=0.0, logprobs=None, n=1, engine='gpt-4',
        frequency_penalty=0, presence_penalty=0, stop=None, rstrip=False,
        partition_id=None, **kwargs):

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rstrip = rstrip
        self.engine = engine
        self.partition_id = partition_id


    def complete(self, prompt):

        openai.api_base = ""
        openai.api_version = '2023-03-15-preview'
        openai.api_key = ""
        openai.api_type = "azure"
        self.deployment_id = self.engine
        if self.rstrip:
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
                if not self.partition_id:
                    # Requests will be routed in round robin by default.
                    partition_id = f"sumscience-{datetime.now()}"
                response = openai.ChatCompletion.create(
                    engine=self.deployment_id,
                    messages=[
                        {"role": "system", "content": "You are an expert chemist. Your expertise lies in reasoning and addressing chemistry problems. "},
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature = self.temperature,
                    max_tokens = self.max_tokens,
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


def verify_formula(problem_statement: str, answer: str, max_attempts: int) -> str:

    gpt = GPT4()

    feedback_prompt = '# You are provided with a chemistry problem and a **Formula retrieval** process for solving the problem. Review the retireved formula and find any problems in it. Justify your answer with a confidence score in the scale of [0,1].'

    feedback_example = load_prompt("./prompts/feedback_formula.txt")

    def is_refinement_sufficient(prompt, feedback, initial, refined) -> bool:
        # Define stopping criteria here
        pass

    # answer = gpt.complete(prompt)
    flag = True

    n_attempts = 0
    max_confidence = 0

    while n_attempts < max_attempts:

        feedback_prompt = feedback_prompt + feedback_example + "# chemistry problem:\n" + problem_statement + "\n# formula retrieval results:\n" + answer
        feedback = gpt.complete(feedback_prompt)
        
        # if is_refinement_sufficient(prompt, feedback, answer, refined):
        #     break
        # if "correct" in feedback:
        #     break

        answer_new = feedback.lstrip("# Here is the correct formula:")
        answer_new, confidence = answer_new.split("**Confidence score:**")
        confidence = int(confidence.strip())

        if confidence > max_confidence:
            max_confidence = confidence
            answer = answer_new
        else:
            answer = answer

        n_attempts += 1

    if n_attempts > 0 :
        flag = False

    return answer, flag

def verify_reasoning(problem_statement: str, formula: str, reasoning: str, max_attempts: int) -> str:

    gpt = GPT4()

    feedback_prompt = '# You are provided with a chemistry problem and a **Reasoning/calculation process** for solving the problem based on the given **Formula retrieval**. Review the reasoning/calculation process step by step and find any problems in it. Justify your answer with a confidence score in the scale of [0,1].'

    feedback_example = load_prompt("./prompts/feedback_reasoning.txt")

    def is_refinement_sufficient(prompt, feedback, initial, refined) -> bool:
        # Define stopping criteria here
        pass

    # answer = gpt.complete(prompt)
    flag = True

    n_attempts = 0
    max_confidence = 0

    while n_attempts < max_attempts:

        feedback_prompt = feedback_prompt + feedback_example + "# chemistry problem:\n" + problem_statement + "# solution:\n" + formula + "\n" + reasoning
        feedback = gpt.complete(feedback_prompt)

        # if is_refinement_sufficient(prompt, feedback, answer, refined):
        #     break
        # if "correct" in feedback:
        #     break

        reasoning_new = feedback.lstrip("# Here is the correct reasoning process:")
        reasoning_new, confidence = reasoning_new.split("**Confidence score:**")
        confidence = int(confidence.strip())

        if confidence > max_confidence:
            max_confidence = confidence
            reasoning = reasoning_new
        else:
            reasoning = reasoning

        n_attempts += 1

    if n_attempts > 0 :
        flag = False

    return reasoning, flag


def run(file, max_attempts, base_lm, mode):

    gpt4 = GPT4(engine=base_lm)
    prompt = load_prompt("./prompts/instruction.txt")

    with open("./scibench/dataset/original/{}.json".format(file)) as f:
        test_data = json.load(f)

    for item in tqdm(test_data):
        problem_text = item['problem_text']
        unit_prob = item['unit']
        new_problem = "\n\n Now try to solve the following problem:\n" + problem_text + " The unit of the answer is " + unit_prob + "."
        
        if mode == 'zero-shot':
            problem_statement = prompt + new_problem
        elif mode == 'few-shot':
            # Randomly select three demonstrations
            txt_files = [file for file in os.listdir("./prompts/demonstrations") if file.endswith('.txt')]
            random_files = random.sample(txt_files, 3)
            demonstrations = "" 
            for demo in random_files:
                demo_content = load_prompt(os.path.join("./prompts/demonstrations", demo)) + "\n\n"
                demonstrations += demo_content
            problem_statement = prompt + "\n\nTo clearly explain the task, we provide the following example:" + demonstrations + new_problem
        
        ### 1. First decompose the problem solving process with formulae retrieval and reasoning process
        response = gpt4.complete(prompt=problem_statement)
        ## 1.1 Parse the generated results of formulae and reasoning
        formula_retrieval, reasoning_process = response.split("**Reasoning/calculation process:**")[0], response.split("**Reasoning/calculation process:**")[1]
        reasoning_process = "**Reasoning/calculation process:**" + reasoning_process.split("**Answer conclusion:**")[0]
        
        ### 2. Iterative review and refinement of formulae and reasoning
        feedback_problem = problem_text + " The unit of the answer is " + unit_prob + "."
        formula_refined, flag_formula = verify_formula(feedback_problem, formula_retrieval, max_attempts)
        reasoning_refined, flag_reasoning = verify_reasoning(feedback_problem, formula_refined, reasoning_process, max_attempts)

        ### 3. Conclude the answers
        if flag_formula and flag_reasoning:
            final_response = response

        else:
            verified_prompt = load_prompt("./prompts/verified_instruction.txt")
            final_response = gpt4.complete(prompt=verified_prompt+formula_refined+reasoning_refined)

        cur = {}
        cur['gpt_output'] = final_response

        with open('./outputs/{}_res.jsonl'.format(file), 'a') as f:

            print(json.dumps(cur, ensure_ascii=False), file=f)
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4')
    parser.add_argument('--refine_iteration', type=int, default=5)     
    parser.add_argument('--dataset', nargs='+', default=["atkins", "chemmc","matter","quan"])
    parser.add_argument('--mode', type=str, default='zero-shot')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    for f in args.dataset:
        run(f, max_attempts=args.refine_iteration, base_lm=args.engine, mode=args.mode)
