import argparse
import base64
import json
from typing import AsyncGenerator
import ssl

import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from virtual_fitting.engine.arg_utils import AsyncEngineArgs
from virtual_fitting.engine.async_preproc_engine import AsyncPreprocEngine
from virtual_fitting.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

@app.post("/preproc_pair")
async def create_upload_files(
    request: Request, 
    file1: UploadFile = File(...), 
    file2: UploadFile = File(...)
):
    request_dict = await request.json()
    stream = request_dict.pop("stream", False)
    contents1 = await file1.read()
    contents2 = await file2.read()
    
    #sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    
    assert engine is not None
    results_generator = engine.preprocess(image=contents1, cloth=contents2)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            ret = [{"path": path, "image": base64.b64encode(image).decode("utf-8")} 
                   for path, image in request_output.outputs]

            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    ret = [{"path": path, "image": base64.b64encode(image).decode("utf-8")} 
                   for path, image in request_output.outputs]
    return JSONResponse(ret)


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
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncPreprocEngine.from_engine_args(engine_args)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
