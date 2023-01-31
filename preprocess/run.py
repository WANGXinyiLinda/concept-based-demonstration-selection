import os
import json
# commands = ['python search_qa.py --do_test --train_k 16384 --test_k 16', 
# 'python spider.py --do_test --train_k 16384 --test_k 16', 
# 'python amazon_polarity.py --do_test --train_k 16384 --test_k 16', 
# 'python dbpedia_14.py --do_test --train_k 16384 --test_k 16', 
# 'python yahoo_answers_topics.py --do_test --train_k 16384 --test_k 16', 
# 'python yelp_review_full.py --do_test --train_k 16384 --test_k 16']

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
