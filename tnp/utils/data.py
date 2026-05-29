import random
from datetime import datetime, timedelta
from typing import Optional

import torch


def adjust_num_batches(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()

    num_batches = worker_info.dataset.num_batches
    adjusted_num_batches = num_batches // worker_info.num_workers
    print(
        f"Adjusting worker {worker_id} num_batches from {num_batches} to {adjusted_num_batches}."
    )
    worker_info.dataset.num_batches = adjusted_num_batches


def era5_set_date_range(worker_id: int, num_days: Optional[int] = None):
    worker_info = torch.utils.data.get_worker_info()

    # Split total date range uniformly between workers.
    total_range = worker_info.dataset.time_range
    start = datetime.strptime(total_range[0], "%Y-%m-%d")
    end = datetime.strptime(total_range[1], "%Y-%m-%d")

    if num_days is None:
        diff = (end - start) / worker_info.num_workers
        start = start + diff * worker_id
        end = start + diff

    else:
        # Just randomly pick interval.
        total_days = (end - start).days
        day_offset = random.randrange(total_days - num_days)
        start = start + timedelta(days=day_offset)
        end = start + timedelta(days=num_days)

    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    date_range = (start_str, end_str)
    print(f"Adjusting worker {worker_id}'s date range to {date_range}.")
    worker_info.dataset.load_data((start_str, end_str))

    adjust_num_batches(worker_id)
