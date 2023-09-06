import ops_demo.func as func
from langchain.chat_models import ChatOpenAI
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
            lattice = func.mp_get_lattice(query.upper(), MP_API_KEY)
        except KeyError:
            return f"Unable to find material {query.upper()}"

        print(f'\nDEBUG: --- get lattice from Materials Project ---\n{lattice}')

        # TODO: add conversion to abc/abg conversion here and uncomment: 
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

