"""
Benchmark inference performance for Qwen2.5-7B-Instruct using ART.

This script sends 5 concurrent requests for a long story (max_tokens=200)
and repeats this 10 times, measuring iteration times and summarizing statistics.
"""
import os
import time
import asyncio
import statistics
from dotenv import load_dotenv
import art
from art.local import LocalBackend

load_dotenv()

async def main():
    # Define prompt and model
    prompt = (
        "Write a long story about a brave astronaut exploring a mysterious planet."
    )
    model = art.TrainableModel(
        name="benchmark-qwen2.5-7b-instruct",
        project="benchmark-qwen2.5-7b-instruct",
        base_model="Qwen/Qwen2.5-7B-Instruct",
    )
    backend = LocalBackend()
    await model.register(backend)

    # Prepare for inference
    client = model.openai_client()
    iterations = 10
    concurrency = 5
    durations = []

    for i in range(1, iterations + 1):
        print(f"Iteration {i}/{iterations}: sending {concurrency} concurrent requests...")
        start = time.perf_counter()
        # launch concurrent requests
        tasks = [
            client.chat.completions.create(
                model=model.name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=1.0,
            )
            for _ in range(concurrency)
        ]
        # Wait for all responses
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        print(f"  Time: {elapsed:.2f} seconds")

    # Compute statistics
    total_time = sum(durations)
    min_time = min(durations)
    max_time = max(durations)
    avg_time = statistics.mean(durations)
    std_time = statistics.stdev(durations) if len(durations) > 1 else 0.0
    total_requests = iterations * concurrency
    avg_per_request = total_time / total_requests

    # Report results
    print("\nInference benchmark results:")
    print(f"  Iterations:    {iterations}")
    print(f"  Concurrency:   {concurrency}")
    print(f"  Total time:    {total_time:.2f} s")
    print(f"  Min iteration: {min_time:.2f} s")
    print(f"  Max iteration: {max_time:.2f} s")
    print(f"  Avg iteration: {avg_time:.2f} s")
    print(f"  Std dev iter:  {std_time:.2f} s")
    print(f"  Avg per req:   {avg_per_request:.2f} s/request")

if __name__ == "__main__":
    asyncio.run(main())