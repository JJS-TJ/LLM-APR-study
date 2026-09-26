import argparse
import sys
import torch
import os
import json
import time
import random
import numpy as np
import openai
from difflib import unified_diff

from Models.model import GPT2, starCoder, LLama2, CodeLLama,deepseek
from utils.parse_d4c import clean_parse_c, choose_prompt,clean_parse_java,clean_parse_97c,clean_parse_97java
from utils.prompt import JAVA_LONG_VARY_PROMPT3,PYTHON_VARY_PROMPT2
from codex_defects4j.api_request import request_engine, create_openai_config, create_gpt4_config, create_openai_config_suffix, create_openai_config_single

# os.environ['CURL_CA_BUNDLE'] = ''
API_KEY_FILE = './codex_defects4j/api_key_gpt4_2.txt'  # read private api key
openai.api_key = open(API_KEY_FILE, 'r').read().strip()
openai.api_base = "https://api.ai-yyds.com/v1"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_unified_diff(source, mutant):
    output = ""
    for line in unified_diff(source.split('\n'), mutant.split('\n'), lineterm=''):
        output += line + "\n"
    return output


def repair_loop(args, model, prompt, file_name, folder, bug, t_chances, skip_val=True):
    start = time.time()
    repair_result = []
    p_diff = {}
    print("Repairing bug {} ... ".format(file_name))
    print(prompt)
    if not model.check_input(prompt, bug['buggy']):
        return 0, False, False, repair_result

    total_times = 0
    while t_chances > 0:
        total_times += 1
        torch.cuda.empty_cache()
        print("Try :{}".format(total_times))
        well, length, outputs, entropies = model.model_prediction(prompt, bug['buggy'], do_sample=True,
                                                                  num_samples=t_chances)
        t_chances -= args.batch_size
        if well:
            for index, output in enumerate(outputs):
                diff = get_unified_diff(bug['buggy'], output)
                if diff in p_diff:
                    repair_result[p_diff[diff]]['num'] += 1
                    continue
                p_diff[diff] = len(repair_result)
                print(diff)
                repair_result.append({'output': output,
                                      'diff': diff,
                                      'finish_reason': 'stop',
                                      'entropy': "",
                                      'num': 1})

    end = time.time()

    print("{} Unique Patches Generated in {}s".format(len(repair_result), end - start))

    json_str = json.dumps(repair_result, indent=4)
    with open("{}/{}.json".format(folder, file_name.replace("/","_")), 'w') as f:
        f.write(json_str)

    return len(repair_result), False, False, repair_result


def repair(args, model, bugs, folder, used_prompt, chances, skip_val=True, only_same=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + "/prompt.txt", "w") as f:
        f.write(used_prompt)
    with open(folder + "/args.txt", "w") as f:
        f.write(str(args))

    result = {}
    t_generated = 0
    t_unique = 0
    start_t = time.time()
    for file_name, bugcontent in bugs.items():
        if os.path.exists("{}/{}.json".format(folder, file_name.replace("/","_"))):
            continue
        #if "Collections" in file_name:
            #example_bug, example_fix = choose_prompt(bugs, file_name)
        #else:
            #example_bug, example_fix = choose_prompt(bugs, file_name)
        with open("../results/cot/deepseek2222/{}.json".format(file_name),"r") as ff:
            an=json.load(ff)
        
        prompt = used_prompt.format(bug=bugcontent['buggy'])
       # time.sleep(600000)
        n_generated, valid, first_try, result[file_name] = repair_loop(args, model, prompt, file_name, folder, bugcontent,
                                                                       chances, skip_val)
        if n_generated >= 1:
            t_generated += chances
            t_unique += len(result[file_name])

    end_t = time.time()

    with open(folder + "/stats.txt", "w") as f:
        f.write("Total generated: {}\n".format(t_generated))
        f.write("Total unique: {}\n".format(t_unique))
        f.write("Total time: {}\n".format(end_t - start_t))

    with open(folder + "/lm_repair.json", "w") as f:  # write to file
        json.dump(result, f)


def repair_codex_loop(prompt, file_name, folder, bug, t_chances, stop="# Provide a fix for the buggy function",
                skip_val=True) -> (bool, bool, list):
    start = time.time()
    repair_result = []
    p_diff = {}
    print("Repairing bug {} ... ".format(file_name.split(".")[0]))
    print(prompt)
    temperature = 0.8
    top_p = 0.95
    config = create_openai_config(message=prompt, stop=stop, temperature=temperature, top_p=top_p)
    total_times = 0
    while t_chances > 0:
        total_times += 1
        t_chances -= 1
        print("Try: {}".format(total_times))
        ret = request_engine(config)
        if ret is None:
            return False, False, []
        output = ret["choices"][0]['message']['content'].strip()
        diff = get_unified_diff(bug['buggy'], output)
        finish_reason = ret["choices"][0]['finish_reason']
        if finish_reason != "stop":
            continue
        if diff in p_diff:
            repair_result[p_diff[diff]]['num'] += 1
            continue
        p_diff[diff] = len(repair_result)
        print(diff)
        repair_result.append({'output': output,
                              'diff': diff,
                              'finish_reason': finish_reason,
                              'num': 1})

    end = time.time()
    print("{} Unique Patches Generated in {}s".format(len(repair_result), end - start))
    json_str = json.dumps(repair_result, indent=4)
    with open("{}/{}.json".format(folder,file_name.split(".")[0]), 'w') as f:
        f.write(json_str)
    return False, False, repair_result

def repair_codex(args, bugs, folder, used_prompt, chances, stop, skip_val=True, only_same=False):
    """
    Codex repair loop, write each patch to corresponding file
    :param args: arguments
    :param bugs: dict of bugs
    :param folder: folder to save the files
    :param used_prompt: prompt as input to codex
    :param chances: number of chances to try to repair (0 means only try once with temp=1)
    :param vary: whether or not the prompt should be varied (specifically designed for d4j and complex bugs, where the
            we use the an example fix from the same project
    :param stop: stop condition for codex
    :param skip_val: if True, skip validation
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + "/prompt.txt", "w") as f:
        f.write(used_prompt)
    with open(folder + "/args.txt", "w") as f:
        f.write(str(args))

    result = {}
    t_generated = 0
    t_unique = 0
    start_t = time.time()

    for file_name, bug in bugs.items():
        # print(file_name, bug)
        example_bug, example_fix = choose_prompt(bugs, file_name)
        prompt = used_prompt.format(example_bug=example_bug, example_fix=example_fix, bug=bug['buggy'])
        valid, first_try, result[file_name] = repair_codex_loop(prompt, file_name, folder, bug, t_chances=chances,
                                                          stop=stop, skip_val=skip_val)
        if len(result[file_name]) != 0:
            t_generated += chances
            t_unique += len(result[file_name])
        # break
    end_t = time.time()

    with open(folder + "/stats.txt", "w") as f:
        f.write("Total generated: {}\n".format(t_generated))
        f.write("Total unique: {}\n".format(t_unique))
        f.write("Total time: {}\n".format(end_t - start_t))

    with open(folder + "/codex_repair.json", "w") as f:  # write to file
        json.dump(result, f)

def repair_gpt4_loop(prompt, file_name, folder, bug, t_chances, stop="# Provide a fix for the buggy function",
                skip_val=True) -> (bool, bool, list):
    start = time.time()
    repair_result = []
    p_diff = {}
    print("Repairing bug {} ... ".format(file_name.split(".")[0]))
    print(prompt)
    temperature = 0.8
    top_p = 0.95
    config = create_gpt4_config(message=prompt, stop=stop, temperature=temperature, top_p=top_p)
    total_times = 0
    while t_chances > 0:
        total_times += 1
        t_chances -= 1
        print("Try: {}".format(total_times))
        ret = request_engine(config)
        if ret is None:
            return False, False, []
        output = ret["choices"][0]['message']['content'].strip()
        diff = get_unified_diff(bug['buggy'], output)
        finish_reason = ret["choices"][0]['finish_reason']
        if finish_reason != "stop":
            continue
        if diff in p_diff:
            repair_result[p_diff[diff]]['num'] += 1
            continue
        p_diff[diff] = len(repair_result)
        print(diff)
        repair_result.append({'output': output,
                              'diff': diff,
                              'finish_reason': finish_reason,
                              'num': 1})

    end = time.time()
    print("{} Unique Patches Generated in {}s".format(len(repair_result), end - start))
    json_str = json.dumps(repair_result, indent=4)
    with open("{}/{}.json".format(folder,file_name.split(".")[0]), 'w') as f:
        f.write(json_str)
    return False, False, repair_result

def repair_gpt4(args, bugs, folder, used_prompt, chances, stop, skip_val=True, only_same=False):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + "/prompt.txt", "w") as f:
        f.write(used_prompt)
    with open(folder + "/args.txt", "w") as f:
        f.write(str(args))

    result = {}
    t_generated = 0
    t_unique = 0
    start_t = time.time()

    for file_name, bug in bugs.items():
        # print(file_name, bug)
        #example_bug, example_fix = choose_prompt(bugs, file_name)
        prompt = used_prompt.format(bug=bug['buggy'])
        valid, first_try, result[file_name] = repair_codex_loop(prompt, file_name, folder, bug, t_chances=chances,
                                                          stop=stop, skip_val=skip_val)
        if len(result[file_name]) != 0:
            t_generated += chances
            t_unique += len(result[file_name])
        # break
    end_t = time.time()

    with open(folder + "/stats.txt", "w") as f:
        f.write("Total generated: {}\n".format(t_generated))
        f.write("Total unique: {}\n".format(t_unique))
        f.write("Total time: {}\n".format(end_t - start_t))

    with open(folder + "/codex_repair.json", "w") as f:  # write to file
        json.dump(result, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="conj",
                        help="Dataset to use, current support: conj conp")
    parser.add_argument("--chances", type=int, default=10)
    parser.add_argument("--skip_val", action="store_true", default=False)
    parser.add_argument("--folder", type=str, default="../results/deepseek_zero425")
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--weight", type=str, default="float16")
    args = parser.parse_args()
    if args.dataset == "introc":
        dataset = clean_parse_c(folder="")
        prompt = C_VARY_PROMPT
        stop = "/* Provide a fix for the buggy function */"
        args.language = "c"
    elif args.dataset == "introjava":
        dataset = clean_parse_java(folder="")
        prompt = JAVA_LONG_VARY_PROMPT
        stop = "// Provide a fix for the buggy function"
        args.language = "java"
    elif args.dataset == "conj":
        dataset = clean_parse_97java(folder="")
        prompt = JAVA_LONG_VARY_PROMPT3
        stop = "// Provide a fix for the buggy function"
        args.language = "java"
    elif args.dataset == "conp":
        dataset = clean_parse_97c(folder="")
        prompt = JAVA_LONG_VARY_PROMPT2
        stop = "# Provide a fix for the buggy function"
        args.language = "python"
    else:
        print("Unsupported dataset!!!", file=sys.stderr)
        exit(-1)

    set_seed(args.seed)
    if args.model_name == "gpt-neo-1.3B":
        model = GPT2(batch_size=args.batch_size, pretrained=args.model_name, stop=stop, weight=args.weight)

    elif args.model_name == "starcoderbase":
        model = starCoder(batch_size= args.batch_size, pretrained=args.model_name, stop=stop, weight=args.weight)
    
    elif args.model_name == "Llama-2-7b-hf":
        model = LLama2(batch_size= args.batch_size, pretrained=args.model_name, stop=stop, weight=args.weight)
    elif args.model_name == "Llama-2-13b-hf":
        model = LLama2(batch_size= args.batch_size, pretrained=args.model_name, stop=stop, weight=args.weight)
    elif args.model_name == "CodeLlama-7b-hf":
        model = CodeLLama(batch_size= args.batch_size, pretrained=args.model_name, stop=stop, weight=args.weight)
    elif args.model_name == "deepseek":
        model = deepseek(batch_size= args.batch_size, pretrained=args.model_name, stop=stop, weight=args.weight)
    elif args.model_name == "gpt-3.5":
        repair_codex(args, dataset, args.folder, prompt, chances=args.chances,
                     stop=stop, skip_val=args.skip_val, only_same=args.dataset.startswith("bugscpp"))
    elif args.model_name == "gpt-4":
        repair_gpt4(args, dataset, args.folder, prompt, chances=args.chances,
                     stop=stop, skip_val=args.skip_val, only_same=args.dataset.startswith("bugscpp"))
    else:
        print("No processed model!!!", file=sys.stderr)
        exit(-1)
    repair(args, model, dataset, args.folder, prompt, args.chances, args.skip_val,
           only_same=args.dataset.startswith("bugscpp"))


if __name__ == '__main__':
    main()
