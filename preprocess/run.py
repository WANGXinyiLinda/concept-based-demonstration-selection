import os
import json

commands = []
with open('../config/tune.json') as f:
    tasks = json.load(f)
blimp = False
ethos = False
for task in tasks['train']:
    if task ==  "quartz-no_knowledge":
        task = "quartz"
    elif task.startswith("glue"):
        task = 'glue_' + task[5:]
    elif task == "wino_grande":
        task = "winogrande"
    elif task == "ag_news":
        task = "agnews"
    elif task.startswith("blimp"):
        task = "blimp"
        blimp = True
    elif task.startswith("ethos"):
        task = "ethos"
        ethos = True
    commands.append(f"python {task}.py --do_test --test_k 4")
    commands.append(f"python {task}.py --do_train --test_k 4")

for c in commands:
    os.system(c)
