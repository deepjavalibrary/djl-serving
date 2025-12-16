#!/usr/bin/env python3
"""
Long document QA benchmark for LMCache testing, adapted from https://github.com/LMCache/LMCache/blob/dev/benchmarks/long_doc_qa/long_doc_qa.py
"""

import argparse
import asyncio
import json
import pandas as pd
import random
import sys
import time
from dataclasses import dataclass

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp is required for async HTTP requests")
    print("Install it with: pip install aiohttp")
    sys.exit(1)


@dataclass
class RequestStats:
    prompt_id: int
    request_start: float
    ttft: float
    request_end: float
    successful: bool


async def send_djl_request(session, semaphore, base_url, prompt, output_len,
                           prompt_index, total_prompts):
    """Send a single async request to DJL's /invocations endpoint with streaming."""
    async with semaphore:
        start_time = time.time()
        first_token_time = None

        payload = {
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "max_tokens": output_len,
            "temperature": 0.0,
            "stream": True
        }

        responses = []
        token_count = 0

        try:
            async with session.post(
                    f"{base_url}/invocations",
                    json=payload,
                    headers={"Content-Type": "application/json"}) as response:
                async for line in response.content:
                    if not line:
                        continue

                    line_str = line.decode('utf-8').strip()

                    if not line_str or line_str.startswith('data:'):
                        continue

                    try:
                        chunk_data = json.loads(line_str)

                        if isinstance(chunk_data,
                                      dict) and 'choices' in chunk_data:
                            choice = chunk_data['choices'][0]
                            content = None

                            if 'delta' in choice and 'content' in choice[
                                    'delta']:
                                content = choice['delta']['content']
                            elif 'message' in choice and 'content' in choice[
                                    'message']:
                                content = choice['message']['content']

                            if content:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                responses.append(content)
                                token_count += 1
                    except json.JSONDecodeError:
                        continue

            end_time = time.time()
            final_response = "".join(responses)

            # Print complete request info
            print(f"\n[Request {prompt_index + 1}/{total_prompts}] "
                  f"Completed in {end_time - start_time:.2f}s")
            print(f"  Response: {final_response[:100]}...")
            print(f"  Tokens: {token_count}/{output_len}")

            ttft = (first_token_time -
                    start_time) if first_token_time is not None else -1
            return RequestStats(
                prompt_id=prompt_index,
                request_start=start_time,
                ttft=ttft,
                request_end=end_time,
                successful=ttft > 0,
            )
        except Exception as e:
            end_time = time.time()
            print(f"\n[Request {prompt_index + 1}/{total_prompts}] "
                  f"FAILED: {e}")
            return RequestStats(
                prompt_id=prompt_index,
                request_start=start_time,
                ttft=-1,
                request_end=end_time,
                successful=False,
            )


async def run_benchmark(base_url, model, prompts, output_len,
                        max_inflight_requests):
    """Run benchmark with given prompts using asyncio."""
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_inflight_requests)

    # Create aiohttp session with no timeout
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Create all tasks
        tasks = [
            send_djl_request(session,
                             semaphore, base_url, prompt, output_len, i,
                             len(prompts)) for i, prompt in enumerate(prompts)
        ]

        # Execute all tasks concurrently
        request_stats = await asyncio.gather(*tasks)

    # Sort by prompt_id to maintain order
    request_stats = list(request_stats)
    request_stats.sort(key=lambda x: x.prompt_id)
    return request_stats


def repeat_prompts(prompts, repeat_count, mode):
    """Repeat prompts according to the specified mode."""
    if mode == "tile":
        return prompts * repeat_count
    elif mode == "interleave":
        repeated = []
        for prompt in prompts:
            repeated.extend([prompt] * repeat_count)
        return repeated
    elif mode == "random":
        repeated = prompts * repeat_count
        random.shuffle(repeated)
        return repeated
    else:
        raise ValueError(f"Invalid repeat mode: {mode}")


def relative_time(df, start_time):
    """Convert absolute times to relative times."""
    df["request_start"] = df["request_start"] - start_time
    df["request_end"] = df["request_end"] - start_time
    df["ttft_time"] = df["request_start"] + df["ttft"]


async def main(args):
    random.seed(args.shuffle_seed)

    base_url = f"http://{args.host}:{args.port}"
    print(f"Using DJL endpoint: {base_url}")
    print(f"Model: {args.model}")

    # Pre-warmup
    print("\n=== Pre-warmup (5 requests) ===")
    pre_warmup_prompts = [
        str(i) + "xx" + " ".join(["hi"] * 1000) for i in range(5)
    ]
    await run_benchmark(base_url, args.model, pre_warmup_prompts,
                        args.output_len, args.max_inflight_requests)

    # Prepare main prompts
    warmup_prompts = [
        str(i) + " " + " ".join(["hi"] * args.document_length)
        for i in range(args.num_documents)
    ]

    # Warmup round
    print("\n=== Warmup round ===")
    warmup_start_time = time.time()
    warmup_request_stats = await run_benchmark(base_url, args.model,
                                               warmup_prompts, args.output_len,
                                               args.max_inflight_requests)
    warmup_end_time = time.time()

    # Query round
    print("\n=== Query round ===")
    print(f"Repeat mode: {args.repeat_mode}")
    query_prompts = repeat_prompts(warmup_prompts, args.repeat_count,
                                   args.repeat_mode)

    benchmark_start_time = time.time()
    benchmark_request_stats = await run_benchmark(base_url, args.model,
                                                  query_prompts,
                                                  args.output_len,
                                                  args.max_inflight_requests)
    benchmark_end_time = time.time()

    # Process results
    warmup_df = pd.DataFrame(
        [stats.__dict__ for stats in warmup_request_stats])
    relative_time(warmup_df, warmup_start_time)
    warmup_df["is_miss"] = True

    benchmark_df = pd.DataFrame(
        [stats.__dict__ for stats in benchmark_request_stats])
    relative_time(benchmark_df, benchmark_start_time)
    benchmark_df["is_miss"] = False

    # Print CSV data
    print("\n" + "=" * 80)
    print("WARMUP ROUND CSV DATA:")
    print("=" * 80)
    print(warmup_df.to_csv(index=False))

    print("\n" + "=" * 80)
    print("QUERY ROUND CSV DATA:")
    print("=" * 80)
    print(benchmark_df.to_csv(index=False))

    # Print summary
    warmup_mean_ttft = warmup_df.query("successful == True")["ttft"].mean()
    query_mean_ttft = benchmark_df.query("successful == True")["ttft"].mean()
    warmup_success_count = warmup_df.query("successful == True").shape[0]
    query_success_count = benchmark_df.query("successful == True").shape[0]

    print(f"Warmup round mean TTFT: {warmup_mean_ttft:.3f}s")
    print(f"Warmup round time: {warmup_end_time - warmup_start_time:.3f}s")
    print(f"Warmup round prompt count: {len(warmup_df)}")
    print(f"Warmup round successful prompt count: {warmup_success_count}")
    print(f"\n=== BENCHMARK RESULTS ===")
    print(f"Query round mean TTFT: {query_mean_ttft:.3f}s")
    print(
        f"Query round time: {benchmark_end_time - benchmark_start_time:.3f}s")
    print(f"Query round prompt count: {len(benchmark_df)}")
    print(f"Query round successful prompt count: {query_success_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DJL LMCache benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--host",
                        type=str,
                        default="localhost",
                        help="DJL server host")
    parser.add_argument("--port",
                        type=int,
                        default=8080,
                        help="DJL server port")
    parser.add_argument("--num-documents",
                        type=int,
                        default=8,
                        help="Number of documents")
    parser.add_argument("--document-length",
                        type=int,
                        default=20000,
                        help="Document length in tokens")
    parser.add_argument("--output-len",
                        type=int,
                        default=100,
                        help="Output length in tokens")
    parser.add_argument("--repeat-count",
                        type=int,
                        default=2,
                        help="Number of repeats")
    parser.add_argument("--repeat-mode",
                        type=str,
                        default="tile",
                        choices=["tile", "random", "interleave"],
                        help="Repeat mode")
    parser.add_argument("--max-inflight-requests",
                        type=int,
                        default=2,
                        help="Max concurrent requests")
    parser.add_argument("--shuffle-seed",
                        type=int,
                        default=0,
                        help="Random seed")

    args = parser.parse_args()
    asyncio.run(main(args))
