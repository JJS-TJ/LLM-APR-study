import json

d4c_bug_lists = '''
| Chart           | jfreechart                 |       26       | 1-26                | None                    |
| Cli             | commons-cli                |       39       | 1-5,7-40            | 6                       |
| Closure         | closure-compiler           |      174       | 1-62,64-92,94-176   | 63,93                   |
| Codec           | commons-codec              |       18       | 1-18                | None                    |
| Collections     | commons-collections        |        4       | 25-28               | 1-24                    |
| Compress        | commons-compress           |       47       | 1-47                | None                    |
| Csv             | commons-csv                |       16       | 1-16                | None                    |
| Gson            | gson                       |       18       | 1-18                | None                    |
| JacksonCore     | jackson-core               |       26       | 1-26                | None                    |
| JacksonDatabind | jackson-databind           |      112       | 1-112               | None                    |
| JacksonXml      | jackson-dataformat-xml     |        6       | 1-6                 | None                    |
| Jsoup           | jsoup                      |       93       | 1-93                | None                    |
| JxPath          | commons-jxpath             |       22       | 1-22                | None                    |
| Lang            | commons-lang               |       64       | 1,3-65              | 2                       |
| Math            | commons-math               |      106       | 1-106               | None                    |
| Mockito         | mockito                    |       38       | 1-38                | None                    |
| Time            | joda-time                  |       26       | 1-20,22-27          | 21                      |'''

def clean_parse_c(folder):
    with open(folder + "introbugc.json", "r") as f:
        result = json.load(f)
    cleaned_result = {}
    for k, v in result.items():
        lines = v['buggy'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k] = {"buggy": "\n".join([line[leading_white_space:] for line in lines])}
    return cleaned_result
def clean_parse_97c(folder):
    with open(folder + "condefectspy.json", "r") as f:
        result = json.load(f)
    cleaned_result = {}
    for k, v in result.items():
        lines = v['buggy'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k] = {"buggy": "\n".join([line[leading_white_space:] for line in lines])}
    return cleaned_result
def clean_parse_java(folder):
    with open(folder + "introbugjava.json", "r") as f:
        result = json.load(f)
    cleaned_result = {}
    for k, v in result.items():
        lines = v['buggy'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k] = {"buggy": "\n".join([line[leading_white_space:] for line in lines])}
    return cleaned_result
def clean_parse_97java(folder):
    with open(folder + "condefectsj.json", "r") as f:
        result = json.load(f)
    cleaned_result = {}
    for k, v in result.items():
        lines = v['buggy'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k] = {"buggy": "\n".join([line[leading_white_space:] for line in lines])}
    return cleaned_result

def _get_relevant_bugs(bugs, current_bug, only_same):
    potential_pairs = []
    project = current_bug.split("-")[0]
    for file_name, bug in bugs.items():
        if file_name == current_bug:
            continue
        if file_name.startswith(project + "-") and only_same:
            potential_pairs.append((len(bug['buggy']) + len(bug['fix']), file_name))
        elif not only_same:
            potential_pairs.append((len(bug['buggy']) + len(bug['fix']), file_name))
    # sort from smallest to largest
    potential_pairs.sort(key=lambda x: x[0])
    return potential_pairs


# picking an example fix pairs from a project
def choose_prompt2(bugs, current_bug, only_same=False):
    potential_pairs = _get_relevant_bugs(bugs, current_bug, only_same)
    return bugs[potential_pairs[0][1]]['buggy'], bugs[potential_pairs[0][1]]['fix']

def choose_prompt(bugs, current_bug):
    project = current_bug.split("-")[0]
    first_choose=['berry-3', 'coreutils-2','cpp_peglib-7', 'cppcheck-19','exiv2-16', 'jerryscript-11', 'libchewing-1', 'libtiff-2',  'libtiff_sanitizer-3', 'libucl-4', 'libxml2-7', 'ndpi-4', 'proj-7',  'wireshark-6', 'xbps-1', 'yara-2', 'zsh-5']
    second_choose=['berry-5', 'coreutils-1', 'cpp_peglib-6', 'cppcheck-10',  'exiv2-2','jerryscript-8', 'libchewing-5', 'libtiff-5', 'libtiff_sanitizer-1', 'libucl-2', 'libxml2-6', 'ndpi-3', 'proj-12', 'wireshark-1', 'xbps-5', 'yara-1', 'zsh-1',]
    if current_bug=="example-1" or current_bug=="dlt_daemon-1":
        return bugs[current_bug.split("-")[0]]['buggy'], bugs[current_bug.split("-")[0]]['fix']
    elif current_bug not in first_choose:
        for p in first_choose:
            if p.split('-')[0]==project:
                return bugs[p]['buggy'], bugs[p]['fix']
    else :
        for p in second_choose:
            if p.split('-')[0]==project:
                return bugs[p]['buggy'], bugs[p]['fix']


def clean_parse_d4j_single_hunk(folder):
    with open(folder + "/single_function_single_hunk_repair.json", "r") as f:
        result = json.load(f)
    cleaned_result = {}
    for k, v in result.items():
        lines = v['buggy'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k + ".java"] = {"buggy": "\n".join([line[leading_white_space:] for line in lines])}
        lines = v["prefix"].splitlines()
        cleaned_result[k + ".java"]["prefix"] = "\n".join([line[leading_white_space:] for line in lines])
        lines = v["suffix"].splitlines()
        cleaned_result[k + ".java"]["suffix"] = "\n".join([line[leading_white_space:] for line in lines])
        lines = v['fix'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k + ".java"]["fix"] = "\n".join([line[leading_white_space:] for line in lines])
    return cleaned_result


def clean_parse_d4c(folder):
    with open(folder + "single_function_bug.json", "r") as f:
        result = json.load(f)
    cleaned_result = {}
    bugscpp=["berry","coreutils","cpp_peglib","cppcheck","dlt_daemon","example","exiv2","jerryscript","libchewing","libtiff","libtiff_sanitizer","libucl","libxml2","ndpi","proj","wireshark","xbps","yara","zsh"]
    for k, v in result.items():
        if(k.split("-")[0] not in bugscpp):
            continue
        lines = v['buggy'].splitlines()
        #leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k] = {"buggy": "\n".join([line for line in lines])}
        #lines = v['fix'].splitlines()
        #leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        #cleaned_result[k + ".java"]["fix"] = "\n".join([line[leading_white_space:] for line in lines])
    with open(folder + "single_function_fixed.json", "r") as f2:
        result2 = json.load(f2)
    for k, v in result2.items():
        if(k.split("-")[0] not in bugscpp):
            continue
        lines = v.splitlines()
        cleaned_result[k]["fix"] = "\n".join([line for line in lines])  
    with open(folder + "other_bugfix.json", "r") as f3:
        result3 = json.load(f3)
    for k, v in result3.items():
        if(k.split("-")[0] not in bugscpp):
            continue
        lines = v['buggy'].splitlines()
        cleaned_result[k] = {"buggy": "\n".join([line for line in lines])}
        lines = v['fix'].splitlines()
        cleaned_result[k]["fix"] = "\n".join([line for line in lines])   
    return cleaned_result

def clean_parse_d4j_single_line(folder):
    with open(folder + "Defects4j/single_function_single_line_repair.json", "r") as f:
        result = json.load(f)
    cleaned_result = {}
    for k, v in result.items():
        lines = v['buggy'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k + ".java"] = {"buggy": "\n".join([line[leading_white_space:] for line in lines])}
        lines = v["prefix"].splitlines()
        cleaned_result[k + ".java"]["prefix"] = "\n".join([line[leading_white_space:] for line in lines])
        lines = v["suffix"].splitlines()
        cleaned_result[k + ".java"]["suffix"] = "\n".join([line[leading_white_space:] for line in lines])
        lines = v['fix'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k + ".java"]["fix"] = "\n".join([line[leading_white_space:] for line in lines])

        buggy_line = cleaned_result[k + ".java"]["buggy"] \
            .removeprefix(cleaned_result[k + ".java"]["prefix"]).removesuffix(
            cleaned_result[k + ".java"]["suffix"]).replace("\n", "")
        cleaned_result[k + ".java"]["buggy_line"] = buggy_line
    return cleaned_result
