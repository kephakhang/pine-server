import asyncio

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing import Any, Union
from typing import Dict, List
from starlette.websockets import WebSocket
import asyncio


job_done = object() # signals the processing is done

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.message = {"event": "init", "payload": "init"}

    async def send(self):
        if self.websocket != None:
            await self.websocket.send_json(self.message)
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running. Clean the queue."""
        self.message = {"event": "start", "payload": "start"}
        asyncio.run(self.send())
        print('start : start')

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.message = {"event": "token", "payload": token}
        asyncio.run(self.send())
        # print('token : ', token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.message = {"event": "end", "payload": "end"}
        asyncio.run(self.send())
        print('end : end')

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        self.message = {"event": "error", "payload": error}
        asyncio.run(self.send())
        print('error : ', error)