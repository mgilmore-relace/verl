from typing import Any

from .base_tool import BaseTool
from .schemas import ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

import demjson3
import json

class FailedTool(BaseTool):

    def __init__(self):
        pass

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        failed_call = parameters['input']

        print("failed tool call")

        try:
            demjson3.decode(failed_call, strict=True)
        except demjson3.JSONDecodeError as e:
            mes = e.message
        
        try:
            fix = json.dumps(demjson3.decode(failed_call))
        except:
            return ToolResponse(f"Error when executing tool: tool call was in invalid JSON\nError Message: {mes}"), 0.0, {}
        else:
            return ToolResponse(f"Error when executing tool: tool call was in invalid JSON\nError Message: {mes}\nPotential Fix: {fix}"), 0.0, {}