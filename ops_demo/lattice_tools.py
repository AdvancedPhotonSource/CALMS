import ops_demo.func as func
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

MP_API_KEY = open('keys/MP_API_KEY').read().strip()


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

