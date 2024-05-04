refine_formulae_prompt = """

You are provided with a ###chemistry problem### and a ###Formula retrieval### process for solving the problem. 

For each instance, you need to do three things. First, for "judgement of the retrieved formulae", you need to review the provided formulae and give your judgement. Then re-organize the "formulae retrieval" process based on your judgement. Finally, justify your answer with a "confidence score" in the scale of [0,1]. The output format should incorporate these components in the following format:

**Judgement of the retrieved formulae:**
[judgement] (Your assessment of whether the retrieved formulae are correct or not.)

**Formula retrieval:**
(Your revised correct formulae required for the problem.)
[Formula 1] (the formula required to solve the problem)
[Formula 2] (the second formula required to solve the problem, if any)
...
[Formula n] (the n-th formula required to solve the problem, if any)

**Confidence score:**
[score] (float number in [0,1])

"""

refine_reasoning_prompt = """

You are provided with a ###chemistry problem###, the corresponding ###Formula retrieval### and ###Reasoning/calculation process### for solving the problem. 

For each instance, you need to do three things. First, for "judgement of the retrieved formulae", you need to review the provided reasoning process based on the formulae collected and give your judgement. Then re-organize the "reasoning process" based on your judgement. Finally, justify your answer with a "confidence score" in the scale of [0,1]. The output format should incorporate these components in the following format:

**Judgement of the reasoning process:**
[judgement] (Your assessment of whether the reasoning process are correct or not.)

**Reasoning/calculation process:**
(Your revised correct reasoning process to solve the problem based on the given formulae.)
[step 1] (the first step for solving this problem)
.....
[step n] (the n-th step for solving the problem, if any)

**Confidence score:**
[score] (float number in [0,1])

"""

refine_reasoning_prompt_pot = """

You are provided with a ###chemistry problem###, the corresponding ###Formula retrieval### and ###Reasoning/calculation process### for solving the problem. 

For each instance, you need to do three things. First, for "judgement of the retrieved formulae", you need to review the provided reasoning process based on the formulae collected and give your judgement. Then re-organize the "reasoning process" based on your judgement. Finally, justify your answer with a "confidence score" in the scale of [0,1]. The output format should incorporate these components in the following format:

**Judgement of the reasoning process:**
[judgement] (Your assessment of whether the reasoning process are correct or not.)

**Reasoning/calculation process:**
(Your revised correct reasoning process to solve the problem based on the given formulae in Python language.)
def solver():
    (your reasoning process in python code and annotations lines)
    ...
    print(ans) (reason out the final answer "ans".)

**Confidence score:**
[score] (float number in [0,1])

"""