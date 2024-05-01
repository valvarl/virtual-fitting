# Udapted from
# https://github.com/vllm-project/vllm/blob/v0.3.0/benchmarks/benchmark_serving.py

import argparse
import asyncio
import json
import os
import os.path as osp
import random
import time
import typing as tp

import aiohttp
from tqdm.asyncio import tqdm
import numpy as np

REQUEST_LATENCY: tp.List[float] = []


def sample_requests(
    dataset_path: str,
    num_requests: int,
) -> tp.List[tp.Dict[str, str]]:
    # Load the dataset.
    with open(osp.join(dataset_path, 'test_pairs.txt')) as f:
        dataset = [{"image": image, "cloth": cloth} for line in f.readlines() for image, cloth in line]

    sampled_requests = random.sample(dataset, num_requests)
    return sampled_requests


async def get_request(
    input_requests: tp.List[tp.Dict[str, str]],
    request_rate: float,
) -> tp.AsyncGenerator[tp.Tuple[str, str], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)
        
        
async def send_request(api_url: str, image, cloth, pbar: tqdm) -> None:
    request_start_time = time.perf_counter()

    headers = {"User-Agent": "Benchmark Client"}

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            pload = {
                'stream': False
            }
            files = [('files', open(image, 'rb')), ('files', open(cloth, 'rb'))]
            async with session.post(api_url, headers=headers,
                                    json=pload, files=files) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append(request_latency)
    pbar.update(1)
    

async def benchmark(
    api_url: str,
    input_requests: tp.List[tp.Tuple[str, int, int]],
    request_rate: float,
) -> None:
    tasks: tp.List[asyncio.Task] = []
    pbar = tqdm(total=len(input_requests))
    async for request in get_request(input_requests, request_rate):
        image, cloth = request['image'], request['cloth']
        task = asyncio.create_task(
            send_request(api_url, image, cloth, pbar))
        tasks.append(task)
    await asyncio.gather(*tasks)
    pbar.close()


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"{args.protocol}://{args.host}:{args.port}{args.endpoint}"
    input_requests = sample_requests(args.dataset, args.num_prompts)

    benchmark_start_time = time.perf_counter()
    asyncio.run(
        benchmark(api_url, input_requests, args.request_rate))
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {args.num_prompts / benchmark_time:.2f} requests/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--protocol",
                        type=str,
                        default="http",
                        choices=["http", "https"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", type=str, default="/generate")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--request-rate",
                        type=float,
                        default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                        "then all the requests are sent at time 0. "
                        "Otherwise, we use Poisson process to synthesize "
                        "the request arrival times.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
