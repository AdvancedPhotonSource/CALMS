import ops_demo.func as func
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import pexpect

MP_API_KEY = open('keys/MP_API_KEY').read().strip()


class DiffractometerAIO(BaseTool):
    name = "movediffractometer"
    description = "tool to set the diffractometer based on the material being analyzed"

    def __init__(self, spec_session):
        super().__init__()
        self.spec_session = spec_session


    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        ENERGY = 10 # ASSUMED, could be parameterized later
        BRAGG_PEAK = 2 # ASSUMED, ...

        lattice = func.mp_get_lattice(query, MP_API_KEY)
        print(f'\nDEBUG: --- get lattice from Materials Project ---\n{lattice}')

        result = func.aps7id_calculate_angle(ENERGY, BRAGG_PEAK, lattice)

        print(f"\nDEBUG: angle = {result['angle']}")

        # TODO: add conversion to abc/abg conversion here and uncomment: 

        """
        self.spec_session.sendline("setlat")
        self.spec_session.expect("real space lattice")
        self.spec_session.readline()
        # lats has 6 numbers, a,b,c,alf,bet,gam                                                                        
        for i in range(6):
            self.spec_session.sendline("{0}".format(lats[i]))
            self.spec_session.readline()
        """

        return 'Diffractometer moved'

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class LatticeSearchTool(BaseTool):
    name = "getlattice"
    description = "tool to get the lattice parameters of a material when calculating a bragg peak"


    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return func.mp_get_lattice(query, MP_API_KEY)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class CalcAngleTool(BaseTool):
    name = "getlattice"
    description = "tool to get the lattice parameters of a material when calculating a bragg peak"


    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return func.mp_get_lattice(query, MP_API_KEY)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class SetSpecTool(BaseTool):
    name = "getlattice"
    description = "tool to get the lattice parameters of a material when calculating a bragg peak"


    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return func.mp_get_lattice(query, MP_API_KEY)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")