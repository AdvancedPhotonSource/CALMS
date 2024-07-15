from langchain.chat_models import ChatOpenAI
import requests
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import Extra
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import pexpect
import subprocess
import os

MP_API_KEY = open('keys/MP_API_KEY').read().strip()

"""
===============================
Python Execution Tools
===============================
"""
def exec_cmd(py_str: str):
    """
    Placeholder for the function. While in testing, just keeping it as a print statement
    """
    print(py_str)
    
    return "Command Executed"



def lint_cmd(py_str: str, py_pfx = None, lint_fp = 'agent_scripts/tmp_lint.py'):
    """
    Helper function to enable linting. 
    Creates a file, prepends text to it, lints it, then removes the file.

        py_str: string to lint
        py_pfx: prefix to add to string. Used if appending py_str to an existing python file
    """
    with open(lint_fp, 'w') as lint_file:
        if py_pfx is not None:
            lint_file.write(py_pfx)
            lint_file.write("\n")
        lint_file.write(py_str)


    # Pylint's internal reporter API fails on so just use subprocess which seesm to be more reliable
    result = subprocess.run(["pylint", "agent_scripts/tmp_lint.py", "-d R,C,W"], stdout=subprocess.PIPE)
    result_str = result.stdout.decode('utf-8')

    os.remove(lint_fp)

    return result_str

def filter_pylint_lines(lint_output, start_ln):
    """
    Filter out the pylint lines that are not needed for the output
    """
    filtered_ouput = []
    for line in lint_output.split('\n'):
        if line.startswith("*********"):
            filtered_ouput.append(line)

        line_split = line.split(':') 
        if len(line_split) > 1:
            if line_split[1].isdigit():
                if int(line.split(':')[1]) > start_ln:
                    filtered_ouput.append(line)

    return '\n'.join(filtered_ouput)



"""
===============================
Wolfram Tools

NOTE: Requires a wolfram alpha appid. Uncomment and add to program to use.
===============================
"""

"""
import os

WF_API_KEY = open('keys/WF_API_KEY').read().strip()
os.environ["WOLFRAM_ALPHA_APPID"] = WF_API_KEY


from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper"\n"
wolfram = WolframAlphaAPIWrapper()

wolfram_tool = StructuredTool.from_function(wolfram.run,
                                            name="Calculator",
                                            description="When performing an arithmatic operation don't assume, run them through this tool as a seperate action. Examples may include addition, subtraction, multiplicaiton, and divison.")
"""


"""
===============================
S26 Tools
===============================
"""

with open('agent_scripts/S26_commandline.py', 'r') as s26_file:
    S26_FILE = ''.join(s26_file.readlines())

# Filters for langchain seems to be parsing the description as a fstring
S26_FILE_FILTER = S26_FILE.replace("{", "")
S26_FILE_FILTER = S26_FILE_FILTER.replace("}", "")

S26_FILE_LINES = len(S26_FILE.split('\n'))



exec_s26_cmd_tool = StructuredTool.from_function(exec_cmd,
                                            name="ExecPython",
                                            description="Takes in a python string and execs it in the envionment described by the script."
                                            + "Before executing the command think through all steps of the problem and use supporting tools when appropriate. "
                                            + "The script will contain objects and functions used to interact with the instrument. "
                                            + "Here are some rules to follow: \n"
                                            + "unlock_hybrid() and lock_hybrid() must be called before and after all motor movements"
                                            + " and scans."
                                            + " The script is described below \n\n" + S26_FILE_FILTER)



def s26_linter(py_str: str):
    """
    Linting tool for the S26 instrument. Prepends the s26 file.
    """
    lint_fp = 'agent_scripts/tmp_lint.py'
    lint_output = lint_cmd(py_str, py_pfx=S26_FILE, lint_fp=lint_fp)
    lint_output = filter_pylint_lines(lint_output, S26_FILE_LINES)
    return lint_output



exec_s26_lint_tool = StructuredTool.from_function(s26_linter,
                                            name="LintPython",
                                            description="Takes in a python string and lints it. The output will provide suggestions on how to improve the code."
                                            + " Use the linter before attempting to run any code")


"""
===============================
Diffrac Tools
===============================
"""

def get_lattice(material: str):
    try:
        lattice = mp_get_lattice(material, MP_API_KEY)
    except KeyError:
        return f"Unable to find material {material}"
    

    return f"{lattice['a']}, {lattice['b']}, {lattice['c']}, {lattice['alpha']}, {lattice['beta']}, {lattice['gamma']}"


def set_diffractometer(a: float, b: float, c: float,
                       alpha: float, beta: float, gamma: float, peak:list[int]):
    
    if len(peak) != 3:
        return "Peak parameters were incorrect. Instrument was NOT set"

    print(a, b, c, alpha, beta, gamma)
    print(peak[0], peak[1], peak[2])

    return "Diffractometer Set"


lattice_tool = StructuredTool.from_function(get_lattice,
                                            name="GetLattice",
                                            description="Gets the lattice parameters for the specified material")

diffractometer_tool = StructuredTool.from_function(set_diffractometer,
                                                   name="SetInstrument",
                                                    description="Sets the instrument to a material's lattice. Requires the 6 lattice parameters: a,b,c,alp,bet,gam."
                                                                + " Do not assume these parameters. Use the GetLattice tool to retrieve them."
                                                                + " The peak parameters are supplied by the user. They are 3 integers.")




class DiffractometerAIO(BaseTool, extra=Extra.allow):
    """
    Tool to query the lattice parameters from the materials project
    and set the diffractometer to the retrieved position. 

    To disable the connection to to spec, the init_spec_ext parameter can be set to false.
    """
    name = "setdetector"
    description = "tool to set the diffractometer based on the material being analyzed, the parameters are first the material then the peak sepearted by a space"

    def __init__(self, init_spec_ext):
        super().__init__()
        self.init_spec = init_spec_ext
        
        if self.init_spec:
            self.spec_session = pexpect.spawn("sixcsim", timeout=3)


    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        query_params = query.split(' ')
        # TODO: Make more rhobust

        material = query_params[0]
        peak = query_params[-3:]
        print(peak)

        try:
            print(query)
            lattice = mp_get_lattice(material, MP_API_KEY)
        except KeyError:
            return f"Unable to find material {material}"

        print(f'\nDEBUG: --- get lattice from Materials Project ---\n{lattice}')

        spec_lattice = [lattice['a'], lattice['b'], lattice['c'], 
                        lattice['alpha'], lattice['beta'], lattice['gamma']]
        print(f'DEBUG: Setting SPEC: {spec_lattice}')

        self.spec_session.sendline(f"ubr {peak[0]} {peak[1]} {peak[2]}")

        self.spec_session.sendline("wh")
        while(1):
            try:
                print(self.spec_session.readline())
            except:
                break 

        if self.init_spec:
            self.spec_session.sendline("setlat")
            self.spec_session.expect("real space lattice")
            self.spec_session.readline()
            # lats has 6 numbers, a,b,c,alf,bet,gam                                                                        
            for i in range(6):
                self.spec_session.sendline("{0}".format(spec_lattice[i]))
                self.spec_session.readline().decode()


            self.spec_session.sendline("p LAMBDA")
            while(1):
                try:
                    print(self.spec_session.readline().decode())
                except:
                    break 

        wh_output = []
        self.spec_session.sendline("wh")
        while(1):
            try:
                wh_output.append(self.spec_session.readline().decode())
            except:
                break 

        return ' '.join(wh_output)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")



def mp_get_lattice(material_formula: str, apikey: str) -> dict:
    """get lattice parameters from Materials Project"""

    url = "https://api.materialsproject.org/materials/summary/"
    kwargs = {
        'headers': {"X-API-KEY": apikey},
        'params': {
            # see https://api.materialsproject.org/docs#/Materials%20Summary
            'formula': material_formula,
            'deprecated': False,
            '_fields': ','.join(
                [
                    'material_id',
                    'formula_pretty',
                    'energy_above_hull',
                    'is_stable',
                    'theoretical',
                    'structure',
                    'symmetry',
                ]
            ),
        },
    }

    response = requests.get(url, **kwargs)
    results = response.json()['data']

    energy_sorted = sorted(
        [
            (
                # sorted by energy_above_hull
                mat['energy_above_hull'],
                # prefer stable and experimentally observed structures
                int(not mat['is_stable']) + int(mat['theoretical']),
                # original index in results
                ix,
            )
            for ix, mat in enumerate(results)
        ]
    )

    selected = results[energy_sorted[0][2]]

    symmetry = selected['symmetry']
    lattice = selected['structure']['lattice']
    info = {
        'id': selected['material_id'],
        'formula': selected['formula_pretty'],
        'symmetry': symmetry['symbol'],
        'crystal': symmetry['crystal_system'],
        'a': lattice['a'],
        'b': lattice['b'],
        'c': lattice['c'],
        'alpha': lattice['alpha'],
        'beta': lattice['beta'],
        'gamma': lattice['gamma'],
    }
    return info