from search import PineconeSearch
from fastapi import FastAPI, Body
from fastapi_utils.inferring_router import InferringRouter
from fastapi.middleware.cors import CORSMiddleware
from item import Item
from prompt import Prompt
import logging
from logging.handlers import RotatingFileHandler
from starlette.websockets import WebSocket, WebSocketDisconnect
from notifier import Notifier
# import json
# import socketio
# import sio


class RunModel:

    def __init__(self):
        """
        Creates a rotating log
        """
        self.logger = logging.getLogger('pine-server')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = RotatingFileHandler("./logs/run.log", maxBytes=30000000,
                                      encoding="utf-8", backupCount=7)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel("DEBUG")

        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allow all methods
            allow_headers=["*"],  # Allow all headers
        )
        # self.sio = socketio.AsyncServer(async_mode='asgi')
        # self.socket_app = socketio.ASGIApp(self.sio)
        # self.app.mount('/', self.socket_app)

        self.router = InferringRouter()
        self.router.get("/")(self.index)
        self.router.post("/search")(self.openai)
        self.router.post("/prompt")(self.prompt)
        self.app.include_router(self.router)
        self.notifier = Notifier()
        self.search = PineconeSearch(self.notifier, self.logger)

    # @sio.event
    # def connect(sid, environ, auth):
    #     """
    #     socket.io 클라이언트가 연결되면 아래 코드를 실행한다, connect는 미리 정의된 이벤트이며
    #     파라미터인 sid, environ, auth 들도 python-socketio 패키지에서 미리 정해놓은 것들이다.
    #     - environ: http 헤더를 포함한 http 리퀘스트 데이터를 담는 WSGI 표준 딕셔너리
    #     - auth: 클라이언트에서 넘겨준 인증 데이터, 데이터가 없으면 None이 된다
    #     - 클라이언트가 보낸 auth의 유저id값으로 해당 유저가 존재하지 않거나
    #     - 인증이 불가한 유저는 return False해서 연결이 안되게 막는다
    #     - return False 하는 대신 raise ConnectionRefusedError으로 연결 거부 메시지를 전송할 수도 있다
    #     """
    #     if not auth:
    #         return False

    def index(self):
        return {"message": "Hello World"}

    def prompt(self, prompt: Prompt):
        self.search.update_prompt(prompt.prompt)
        return prompt

    def openai(self, item: Item):
        try:
            self.logger.debug("input query")
            result = self.search.send_query(item)
            self.logger.debug('result : %s', result)
            return result
        except Exception as e:
            self.logger.error('예외가 발생했습니다.', str(e))
            return {
                "error": str(e)
            }


run = RunModel()
app = run.app


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await run.notifier.connect(websocket)
    print("websocket ", websocket)
    try:
        while True:
            message = await websocket.receive_json()
            if message['event'] == 'ping':
                await websocket.send_json({'event': 'pong', 'payload': 'pong'})
                print("receive ping : ",message)
            else:
                print("receive : ",message)
    except WebSocketDisconnect:
        run.notifier.remove(websocket)






