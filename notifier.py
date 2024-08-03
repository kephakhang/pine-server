from starlette.websockets import WebSocket, WebSocketDisconnect
import uuid
import json
import asyncio


class Notifier:
    def __init__(self):
        self.connections: dict = {}
        self.uids: dict = {}
        self.qa_dict: dict = {}
        self.qa_hotplc_dict: dict = {}
        self.generator = self.get_notification_generator()

    async def get_notification_generator(self):
        while True:
            message = yield
            await self._notify(message)

    async def push(self, msg: str):
        await self.generator.asend(msg)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        uid = str(uuid.uuid4())
        self.connections[uid] = websocket
        self.uids[websocket] = uid
        await websocket.send_json({"event": "connected", "payload": uid})

    async def send_json(self, uid: str, json: json):
        self.connections[uid].send_json(json)

    async def send_text(self, uid: str, text: str):
        self.connections[uid].send_text(text)

    def remove(self, websocket: WebSocket):
        uid = self.uids[websocket]
        del self.connections[uid]
        del self.uids[websocket]
        if uid in self.qa_dict:
            del self.qa_dict[uid]
        if uid in self.qa_hotplc_dict:
            del self.qa_hotplc_dict[uid]

    async def _notify(self, message: str):
        living_connections: dict
        living_uids: dict
        for uid, websocket in self.connections.items():
            # Looping like this is necessary in case a disconnection is handled
            # during await websocket.send_text(message)
            await websocket.send_text(message)
            living_connections[uid] = websocket
            living_uids[websocket] = uid
        self.connections = living_connections
        self.uids = living_uids
