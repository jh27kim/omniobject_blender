import glob
import json
import multiprocessing
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import boto3
import tyro
import wandb


@dataclass
class Args:
    workers_per_gpu: int
    """number of workers per gpu"""

    input_models_path: str
    """Path to a json file containing a list of 3D object files"""

    upload_to_s3: bool = False
    """Whether to upload the rendered images to S3"""

    log_to_wandb: bool = False
    """Whether to log the progress to wandb"""

    num_gpus: int = -1
    """number of gpus to use. -1 means all available gpus"""


def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    s3: Optional[boto3.client],
) -> None:
    while True:
        item, is_train = queue.get()
        if item is None:
            break

        print(f"ISTRAIN ISTRAIN ISTRAIN {is_train} ISTRAIN ISTRAIN ISTRAIN ISTRAIN")

        # Perform some operation on the item
        print(item, gpu)
        command = (
            f"export DISPLAY=:0.{gpu} &&"
            # f" blender-3.2.2-linux-x64/blender -b -P scripts/blender_script.py --"
            f" /Applications/Blender.app/Contents/MacOS/Blender -b -P omniobject_script/blender_script_zero123.py --"
            f" --object_path {item}"
            f" --is_train {is_train}"
            f" --category clock"
        )
        print("command: ", command)
        subprocess.run(command, shell=True)

        if Args.upload_to_s3:
            if item.startswith("http"):
                uid = item.split("/")[-1].split(".")[0]
                for f in glob.glob(f"views/{uid}/*"):
                    s3.upload_file(
                        f, "objaverse-images", f"{uid}/{f.split('/')[-1]}"
                    )
            # remove the views/uid directory
            shutil.rmtree(f"views/{uid}")

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    args = tyro.cli(Args)

    s3 = boto3.client("s3") if args.upload_to_s3 else None
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    if args.log_to_wandb:
        wandb.init(project="objaverse-rendering", entity="prior-ai2")

    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i, s3)
            )
            process.daemon = True
            process.start()

    # Add items to the queue
    with open(args.input_models_path, "r") as f:
        model_paths = json.load(f)
    
    test_index = int(len(model_paths) * 0.8)
    for i, item in enumerate(model_paths):
        is_train = True
        if i > test_index:
            is_train = False
        queue.put([item, is_train])

    # update the wandb count
    if args.log_to_wandb:
        while True:
            time.sleep(5)
            wandb.log(
                {
                    "count": count.value,
                    "total": len(model_paths),
                    "progress": count.value / len(model_paths),
                }
            )
            if count.value == len(model_paths):
                break

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)
