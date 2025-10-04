from .base_tool import BaseTool
from .schemas import ToolResponse

import demjson3

class FailedTool(BaseTool):

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        failed_call = parameters['input']

        try:
            demjson3.decode(failed_call, strict=True)
        except demjson3.JSONDecodeError as e:
            mes = e.message
        
        try:
            fix = demjson3.decode(failed_call)
        except:
            return ToolResponse(f"Error when executing tool: tool call was in invalid JSON\nError Message: {e.message}"), 0.0, {}
        finally:
            return ToolResponse(f"Error when executing tool: tool call was in invalid JSON\nError Message: {e.message}\nPotential Fix: {fix}"), 0.0, {}