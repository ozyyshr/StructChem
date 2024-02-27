import json

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

gpt_output = load_jsonl('generated_problems_new.jsonl')

problems = []

for item in gpt_output:
    item = item['gpt_output']
    try:
        problem1 = item.split('Problem 2')[0].split("Problem 1")[1].strip("\n")
        problem2 = item.split('Problem 3')[0].split('Problem 2')[1].strip("\n")
        problem3 = item.split('Problem 3')[1].strip('\n')
    except:
        continue
    
    problems.append(problem1)
    problems.append(problem2)
    problems.append(problem3)

with open('problems_cleaned.json', 'w') as f:
    json.dump(problems, f)

    
