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

MP_API_KEY = open('keys/MP_API_KEY').read().strip()


class DiffractometerAIO(BaseTool, extra=Extra.allow):
    """
    Tool to query the lattice parameters from the materials project
    and set the diffractometer to the retrieved position. 

    To disable the connection to to spec, the init_spec_ext parameter can be set to false.
    """
    name = "setdiffractometer"
    description = "tool to set the diffractometer based on the material being analyzed"

    def __init__(self, init_spec_ext):
        super().__init__()
        self.init_spec = init_spec_ext
        
        if self.init_spec:
            self.spec_session = pexpect.spawn("sixcsim", timeout=3)


    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        ENERGY = 10 # ASSUMED, could be parameterized later
        BRAGG_PEAK = 2 # ASSUMED, ...
        try:
            lattice = mp_get_lattice(query.upper(), MP_API_KEY)
        except KeyError:
            return f"Unable to find material {query.upper()}"

        print(f'\nDEBUG: --- get lattice from Materials Project ---\n{lattice}')

        spec_lattice = [lattice['a'], lattice['b'], lattice['c'], 
                        lattice['alpha'], lattice['beta'], lattice['gamma']]
        print(f'DEBUG: Setting SPEC: {spec_lattice}')

        if self.init_spec:
            self.spec_session.sendline("setlat")
            self.spec_session.expect("real space lattice")
            self.spec_session.readline()
            # lats has 6 numbers, a,b,c,alf,bet,gam                                                                        
            for i in range(6):
                self.spec_session.sendline("{0}".format(spec_lattice[i]))
                print(self.spec_session.readline())

        return 'Diffractometer Moved'

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