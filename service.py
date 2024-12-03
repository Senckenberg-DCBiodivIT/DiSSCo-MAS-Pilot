import os
import uvicorn
import asyncio

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState  
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict 

class ImageRequest(BaseModel):
    image_url: str

class ProcessedMessageRequest(BaseModel):
    image_url: str
    output: list
    image_height: int
    image_width: int

message_queue: List[str] = []
processed_messages: Dict[str, Dict] = {}
active_connections: List[WebSocket] = []


app = FastAPI()


@app.post("/plant_organ_segmentation")
async def plant_organ_segmentation(request: ImageRequest):
    try:
        message_queue.append(request.image_url)
        timeout = 30
        start_time = asyncio.get_event_loop().time()

        while request.image_url not in processed_messages and (asyncio.get_event_loop().time() - start_time) <=timeout:
            await asyncio.sleep(1)

        if request.image_url not in processed_messages:
            return JSONResponse(content={"error": "Processing took too long."}, status_code=504)
            
        response_data = processed_messages[request.image_url]
        del processed_messages[request.image_url]
        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {str(e)}")

@app.websocket("/ws/new_message")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while websocket.application_state == WebSocketState.CONNECTED:
            if message_queue:   
                message = message_queue.pop(0)   
                await websocket.send_json({"message": message})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"Error: {str(e)}")
        await websocket.close(code=1000)


@app.post("/processed_message")
async def processed_message(request: ProcessedMessageRequest):
    try:
        processed_data = {
                "image_url": request.image_url,
                "output": request.output,
                "image_height": request.image_height,
                "image_width": request.image_width
            }

        processed_messages[request.image_url] = processed_data
        return JSONResponse(content={"message": "Processed data stored successfully."})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the mesage: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

