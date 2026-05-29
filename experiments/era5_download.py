import argparse
import os
from multiprocessing import Pool
from typing import Tuple

import cdsapi


def download_data(args: Tuple[int, int, str]):
    year, month, output_dir = args
    c = cdsapi.Client()

    output_file = f"{output_dir}/{month:02d}-{year}.nc"
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping.")
        return

    print(f"Downloading data for {year}-{month:02d}")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "skin_temperature",
            ],
            "year": f"{year}",
            "month": [
                f"{month:02d}",
            ],
            "day": [f"{day:02d}" for day in range(1, 32)],
            "time": [f"{hour:02d}:00" for hour in range(24)],
            "format": "netcdf",
        },
        output_file,
    )
    print(f"Finished downloading {output_file}")


def run(start_year: int, end_year: int, num_processes: int, output_dir: str):
    # Generate all year-month combinations
    tasks = [
        (year, month, output_dir)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a pool of worker processes
    with Pool(num_processes) as pool:
        # Map the download_data function to the tasks
        pool.map(download_data, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_year", type=int, required=True, help="Start year for data download"
    )
    parser.add_argument(
        "--end_year", type=int, required=True, help="End year for data download"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of parallel processes to use",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/era5",
        help="Directory to save downloaded files",
    )
    args = parser.parse_args()

    run(args.start_year, args.end_year, args.num_processes, args.output_dir)
