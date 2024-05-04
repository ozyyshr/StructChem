from datetime import datetime, timedelta
import logging
import time
import json
import random
import re
import io
import sys
from instruction import overall_instruction, overall_instruction_pot
import argparse
import os
from prompts.refine_prompt import refine_formulae_prompt, refine_reasoning_prompt, refine_reasoning_prompt_pot
# from azure.identity import DefaultAzureCredential
import openai
from openai import OpenAI
from tqdm import tqdm


def load_prompt(file):
    prompt = ''
    with open(file) as f:
        for line in f:
            prompt = prompt + line.strip() + '\n'
    return prompt

class GPT4:

    def __init__(self, max_tokens=1024, temperature=0.0, logprobs=None, n=1, engine='gpt-4',
        frequency_penalty=0, presence_penalty=0, stop=None, rstrip=False, **kwargs):

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rstrip = rstrip
        self.engine = engine
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", None),
        )


    def complete(self, prompt):

        openai.api_version = '2023-03-15-preview'
        self.deployment_id = self.engine
        
        if self.rstrip:
            # Remove heading whitespaces improves generation stability. It is
            # disabled by default to keep consistency.
            prompt = prompt.rstrip()
        retry_interval_exp = 1 

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_id,
                    messages=[
                        {"role": "system", "content": "You are an expert chemist. Your expertise lies in reasoning and addressing chemistry problems. "},
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature = self.temperature,
                    max_tokens = self.max_tokens,
                )
                return response.choices[0].message.content
            
            except openai.RateLimitError as e:
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

            except openai.APIConnectionError as e:
                logging.warning("OpenAI API connection error. Retry")
                time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
                retry_interval_exp += 1

            # except openai.Timeout as e:
            #     logging.warning("OpenAI timeout error. Sleep then retry.")
            #     time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
            #     retry_interval_exp += 1


def verify_formula(problem_statement: str, formulae: str, max_attempts: int) -> str:
    gpt = GPT4()
    formulae_retrieved = formulae

    def is_refinement_sufficient(prompt, feedback, initial, refined) -> bool:
        # Define stopping criteria here
        pass

    flag = True

    n_attempts = 0
    max_confidence = 0.0

    while n_attempts < max_attempts:
        
        with io.StringIO() as f:
            f.write(refine_formulae_prompt.strip() + "\n\n")
            f.write("Now try the following. Remember to strictly follow the output format:\n\n")
            f.write(f"### Chemistry problem:###\n {problem_statement}\n\n### Formulae retrieval:###\n{formulae_retrieved}")
            model_input = f.getvalue()

        refined_formulae = gpt.complete(model_input)

        formulae_new, conf_f = refined_formulae.split("**Confidence score:**")[0].strip("\n"), refined_formulae.split("**Confidence score:**")[1].strip()
        
        # extract the confidence score and the refined components
        conf = float(re.findall(r"\d+\.?\d*", conf_f)[0])
        formulae_new = "**Formula retrieval:**" + formulae_new.split("**Formula retrieval:**")[1]


        if conf > max_confidence:
            max_confidence = conf
            formulae = formulae_new
        else:
            formulae = formulae

        n_attempts += 1

    if n_attempts > 0 :
        flag = False

    return formulae, flag

def verify_reasoning(problem_statement: str, formula: str, reasoning: str, max_attempts: int, pot: bool) -> str:

    gpt = GPT4()

    def is_refinement_sufficient(prompt, feedback, initial, refined) -> bool:
        # Define stopping criteria here
        pass

    flag = True

    n_attempts = 0
    max_confidence = 0.0

    if pot:
        refine_reasoning_prompt = refine_formulae_prompt_pot

    while n_attempts < max_attempts:

        with io.StringIO() as f:
            f.write(refine_reasoning_prompt.strip() + "\n\n")
            f.write("Now try the following. Remember to strictly follow the output format:\n\n")
            f.write(f"### Chemistry problem:###\n {problem_statement}\n\n### Formulae retrieval:###\n{formula}\n\n###Reasoning process###\n{reasoning}")
            model_input = f.getvalue()
        
        refined_reasoning = gpt.complete(model_input)

        reasoning_new, conf_f = refined_reasoning.split("**Confidence score:**")[0].strip("\n"), refined_reasoning.split("**Confidence score:**")[1].strip()

        # extract the confidence score and the refined components
        conf = float(re.findall(r"\d+\.?\d*", conf_f)[0])
        reasoning_new = "**Reasoning/calculation process:**" + reasoning_new.split("**Reasoning/calculation process:**")[1]

        if conf > max_confidence:
            max_confidence = conf
            reasoning = reasoning_new
        else:
            reasoning = reasoning

        n_attempts += 1

    if n_attempts > 0 :
        flag = False

    return reasoning, flag


def run(file, max_attempts, base_lm, mode, pot):

    gpt4 = GPT4(engine=base_lm)
    if pot:
        prompt = overall_instruction_pot
    else:
        prompt = overall_instruction
    # prompt = load_prompt("./prompts/instruction.txt")

    with open("./datasets/{}.json".format(file)) as f:
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
        reasoning_refined, flag_reasoning = verify_reasoning(feedback_problem, formula_refined, reasoning_process, max_attempts, pot)

        ### 3. Conclude the answers
        if not pot:
            if flag_formula and flag_reasoning:
                final_response = response
            else:
                verified_prompt = load_prompt("./prompts/verified_instruction.txt")
                final_response = gpt4.complete(prompt=verified_prompt+formula_refined+reasoning_refined)
        else:
            old_stdout = sys.stdout
            redirected_output = sys.stdout = StringIO()
            try:
                reasoning_pot = reasoning_refined.split("**Reasoning/calculation process:**")[1]
                exec(reasoning_pot)
                sys.stdout = old_stdout
                final_response = redirected_output.getvalue().strip()
            except:
                final_response = "None"

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
    parser.add_argument('--pot', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    for f in args.dataset:
        run(f, max_attempts=args.refine_iteration, base_lm=args.engine, mode=args.mode, pot=args.pot)
