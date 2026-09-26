import json
import os

rootpath = '/data/gpt/gpt_out/defects4j/'

files = os.listdir(rootpath)
# print(files)
repos = ['Chart', 'Closure', 'Math', 'Lang', 'Time']

for file in files:
    if '.json' not in file:
        continue
    if file.split(".json")[0].split("_")[0] not in repos:
        continue
    print(file)
    tmpdict = json.load(open(rootpath+file, 'r'))
    outdict = {}
    maxtries=50
    counter = 0
    for dictkey in tmpdict.keys():
        # dictkey = str(i)
        # if counter >= 50:
        #     break
        functions = tmpdict[dictkey]
        if "// Fixed Function\n" not in functions:
            continue
        patch     = functions.split("// Fixed Function\n")[1]
        # counter += 1
        outdict[dictkey] = patch
    if outdict == {}:
        continue
    with open('/data/PLM4APR/codex_out/gptout/{}'.format(file), 'w') as jsonf:
        jsonstr = json.dumps(outdict, indent=4)
        jsonf.write(jsonstr)  