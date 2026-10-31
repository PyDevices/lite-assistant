"""
Function tools the client defines and runs: their definitions as the chat
completions API takes them, which is what the LLM engines take, from that
API's request or from the flat form the Responses and Realtime APIs use.
"""
from typing import Any, Dict, List, Optional, Union


class ToolError(ValueError):
    """ A tool definition or tool_choice the server can't use. """


def chat_tools(tools: Optional[List[Dict[str, Any]]], tool_choice: Union[str, Dict[str, Any], None] = None,
               flat: bool = False) -> Optional[List[Dict[str, Any]]]:
    """
    The tools for the engine, in chat completions form, after tool_choice.
    flat marks the Responses/Realtime form, {"type": "function", "name", ...}.
    tool_choice "none" drops them all, and naming one function keeps just
    it. Gemma can't be made to call a tool, so "required" acts as "auto".
    """
    definitions = []
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            kind = tool.get("type") if isinstance(tool, dict) else type(tool).__name__
            raise ToolError(f"Only function tools are supported, not {kind!r}.")
        function = {k: v for k, v in tool.items() if k != "type"} if flat else tool.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            raise ToolError("Each function tool needs a name.")
        # strict asks for schema-exact arguments, which Gemma can't promise
        function = {"parameters": {"type": "object", "properties": {}},
                    **{k: v for k, v in function.items() if k != "strict"}}
        definitions.append({"type": "function", "function": function})
    if tool_choice == "none":
        return None
    if isinstance(tool_choice, dict):
        name = tool_choice.get("name") or (tool_choice.get("function") or {}).get("name")
        definitions = [d for d in definitions if d["function"]["name"] == name]
        if not definitions:
            raise ToolError(f"tool_choice names {name!r}, which isn't among the tools.")
    return definitions or None
