# this file trying to test how many bugs only depend on LLM's output 
import os
import json
import subprocess
from utils.DataSet import DataSet
import javalang
import time
import re
import signal

bugs = DataSet('../../d4j-info/single_function_repair.json', '../../d4j-info/filelist.json')


def run_d4j_test(source, testmethods, bug_id, workingdir):
    buggy = False
    compile_fail = False
    time_out = False
    entire_buggy = False
    error_string = ""
    
    # check syntax error
    try:
        tokens = javalang.tokenizer.tokenize(source)
        parser = javalang.parser.Parser(tokens)
        parser.parse()
    except:
        return compile_fail, time_out, buggy, entire_buggy, True
    
    for t in testmethods:
        cmd = "defects4j test -w {workingdir} -t {testmethod}".format(workingdir=workingdir, testmethod=t.strip())
        returncode = ""
        error_file = open("stderr.txt", "wb")
        child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=error_file, bufsize=-1,
                                 start_new_session=True)
        while_begin = time.time()
        while True:
            Flag = child.poll()
            if Flag == 0:
                # if child.stdout is not None:
                returncode = child.stdout.readlines()  # child.stdout.read()
                print(b"".join(returncode).decode('utf-8'))
                error_file.close()
                break
            elif Flag != 0 and Flag is not None:
                compile_fail = True
                error_file.close()
                with open("stderr.txt", "rb") as f:
                    r = f.readlines()
                for line in r:
                    if re.search(':\serror:\s', line.decode('utf-8')):
                        error_string = line.decode('utf-8')
                        break
                print(error_string)
                break
            elif time.time() - while_begin > 15:
                error_file.close()
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                time_out = True
                break
            else:
                time.sleep(0.01)
        log = returncode
        # print("log {}".format(log))
        if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
            continue
        else:
            buggy = True
            break
        
    if not buggy:
        print('So you pass the basic tests, Check if it passes all the test, include the previously passing tests')
        cmd = "defects4j test -w {workingdir}".format(workingdir= workingdir)
        returncode = ""
        child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1,
                                 start_new_session=True)
        while_begin = time.time()
        while True:
            Flag = child.poll()
            if Flag == 0:
                # if child.stdout is not None:
                returncode = child.stdout.readlines()  # child.stdout.read()
                break
            elif Flag != 0 and Flag is not None:
                buggy = True
                break
            elif time.time() - while_begin > 180:
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                buggy = True
                break
            else:
                time.sleep(0.01)
        log = returncode
        if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
            print('success')
        else:
            entire_buggy = True

    return compile_fail, time_out, buggy, entire_buggy, False


def test_all_patches():
    plausible = 0
    outcome = '../results/starcoderbase_test/'
    rootpath = '../results/starcoderbase/' # where generated patches are stored
    info = bugs.getBugList()
    # projs = ["Chart", "Closure", "Time", "Math", "Lang"]
    projs = ["Gson"]
    for key in info:
        ids = info[key]
        for idnum in ids:
            if key not in projs:
                continue
            if os.path.exists("{}/{}_{}.txt".format(outcome, key, idnum)):
                continue
            print("{}_{}".format(key, idnum))
            subprocess.run("rm -rf ./tmp/defects4j_buggy/{repol}/{repol}_{id}_buggy".format(repol=key.lower(), id=idnum), shell=True)
            subprocess.run('defects4j checkout -p {repo} -v {id}b -w ./tmp/defects4j_buggy/{repol}/{repol}_{id}_buggy'\
                           .format(repo=key, id=idnum, repol=key.lower()), shell=True)
            
            testmethods = os.popen("defects4j export "
                                   "-w ./tmp/defects4j_buggy/{repol}/{repol}_{id}_buggy "
                                   "-p tests.trigger".format(repol=key.lower(), id=idnum)).readlines()
            
            modifyfile = bugs.getOneBugFile("{}_{}".format(key, idnum))
            
            beginline, endline = bugs.getOneBugLine("{}-{}".format(key, idnum))
            # then check 50 output
            LLMoutputfile = rootpath+"{}-{}.json".format(key, idnum)
            if not os.path.exists(LLMoutputfile):
                with open("./error.txt" ,'a') as f:
                    f.write("No generated patches"+LLMoutputfile+"\n")
                continue
            outputdict = json.load(open(LLMoutputfile, 'r'))
            outputlist = []
            # Original file format
            # for num in outputdict.keys():
            #     if num.isdigit():
            #         outputlist.append(outputdict[num])
            # new file format
            for item in outputdict:
                outputlist.append(item["output"])
            # outputlist = list(set(outputlist))
            tries = len(outputlist)
            originalfile = './tmp/defects4j_buggy/{repo}/{repo}_{id}_buggy/{file}'.format(repo=key.lower(), id=idnum, file=modifyfile)
            
            try:
                with open(originalfile, 'r') as f:
                    sourcelines = f.readlines()
            except:
                with open(originalfile, 'r', encoding="ISO-8859-1") as f:
                    sourcelines = f.readlines()
            
            originalcontent = "".join(sourcelines)
            
            try:
                with open(originalfile, 'r') as f:
                    prior = f.readlines()[0:beginline-1]
            except:
                with open(originalfile, 'r', encoding='ISO-8859-1') as f:
                    prior = f.readlines()[0:beginline-1]
            prior = "".join(prior)

            try:
                with open(originalfile, 'r') as f:
                    after = f.readlines()[endline:]
            except:
                with open(originalfile, 'r', encoding='ISO-8859-1') as f:
                    after = f.readlines()[endline:]
            after = "".join(after)
            
            correctpathes = []
            for i in range(0, tries):
                newfile = prior + "\n" + outputlist[i] + "\n" + after
                with open(originalfile, 'w', encoding='utf-8') as f:
                    f.write(newfile)
                # Begin Testing
                compile_fail, timed_out, buggy, entire_buggy, syntax_error = run_d4j_test(newfile, testmethods, "{}_{}".format(key, idnum), "./tmp/defects4j_buggy/{repo}/{repo}_{id}_buggy/".format(repo=key.lower(), id=idnum))
                print("testoutcome: {}, {}, {}, {}, {}".format(
                    compile_fail, timed_out, buggy, entire_buggy, syntax_error))

                if not compile_fail and not timed_out and not buggy and not entire_buggy and not syntax_error:
                    plausible += 1
                    if outputlist[i] in correctpathes:
                        javaf = open(originalfile, 'w', encoding='utf-8')
                        javaf.write(originalcontent)
                        continue
                    if not os.path.exists("{}/{}_{}.txt".format(outcome, key, idnum)):
                        with open("{}/{}_{}.txt".format(outcome, key, idnum), 'w', encoding='utf-8') as f:
                            f.write("No.{} Patch\n".format(i))
                            f.write(outputlist[i]+"\n")
                    else:
                        with open("{}/{}_{}.txt".format(outcome, key, idnum), 'a', encoding='utf-8') as f:
                            f.write("No.{} Patch\n".format(i))
                            f.write(outputlist[i]+"\n")
                    correctpathes.append(outputlist[i])
                javaf = open(originalfile, 'w', encoding='utf-8')
                javaf.write(originalcontent)


if __name__ == "__main__":
    # getDefects4jLines()
    # getDefects4jFiles()
    test_all_patches()
