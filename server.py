import asyncio
import argparse
import ssl
import os 
import json
import typing as tp
import io
from PIL import Image
import base64

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from virtual_fitting.engine.async_preproc_engine import RequestTracker
from virtual_fitting.outputs import RequestOutput
from virtual_fitting.utils import random_uuid


TIMEOUT_KEEP_ALIVE = 5

app = FastAPI()
request_tracker = RequestTracker()


async def get_body(request: Request):
    content_type = request.headers.get('Content-Type')
    if content_type is None:
        raise HTTPException(status_code=400, detail='No Content-Type provided!')
    elif (content_type == 'application/x-www-form-urlencoded' or
          content_type.startswith('multipart/form-data')):
        try:
            return await request.form()
        except Exception:
            raise HTTPException(status_code=400, detail='Invalid Form data')
    else:
        raise HTTPException(status_code=400, detail='Content-Type not supported!')


@app.post("/preproc_pair")
async def create_upload_files(
    request: Request,
    request_dict = Depends(get_body),
):
    files = request_dict.getlist('files')
    
    if len(files) != 2:
        raise HTTPException(status_code=400, detail="Incorrect number of files. Expected 2 files.")
    
    file1, file2 = files[0], files[1]
    contents1 = await file1.read()
    contents2 = await file2.read()
    
    save_path = "./saved_files"
    os.makedirs(save_path, exist_ok=True)  # Создаем папку для сохранения файлов, если ее нет
    file1_save_path = os.path.join(save_path, file1.filename)
    with open(file1_save_path, "wb") as f:
        f.write(contents1)
        
    file2_save_path = os.path.join(save_path, file2.filename)
    with open(file2_save_path, "wb") as f:
        f.write(contents2)
    
    request_id = random_uuid()
    stream = request_tracker.add_request(request_id, image=file1_save_path, cloth=file2_save_path)
    request_tracker._request_streams[stream.request_id] = stream

    # Streaming case
    async def stream_results() -> tp.AsyncGenerator[bytes, None]:
        async for request_output in stream:
            ret = [{"path": path, "image": image_to_base64(Image.open(path))} 
                   for path, image in request_output.outputs]
            
            yield (json.dumps(ret) + "\n").encode('utf-8')
            
    async def task():
        await asyncio.sleep(3)
        request_tracker.process_request_output(RequestOutput(request_id, [(file1_save_path, open(file1_save_path, 'rb'))], finished=False))
        await asyncio.sleep(3)
        request_tracker.process_request_output(RequestOutput(request_id, [(file2_save_path, open(file2_save_path, 'rb'))], finished=True))
    
    asyncio.create_task(task())
    
    st = json.loads(str(request_dict.get('stream', False)).lower())
    print(st, isinstance(st, bool))
    if st:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in stream:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            request_tracker.abort_request(request_id,
                                            verbose=True)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    ret = [{"path": path, "image": image_to_base64(Image.open(path))} 
                   for path, image in request_output.outputs]
    
    return JSONResponse(ret)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="debug")

    args = parser.parse_args()

    app.root_path = args.root_path
    uvicorn.run(app, host='0.0.0.0', port=8000)
