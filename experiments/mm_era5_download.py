import argparse
import multiprocessing
import os
from typing import List

import cdsapi


def download_era5_pl_data(
    year: int,
    month: int,
    variables: List[str],
    pressure_levels: List[str],
    output_dir: str,
    lat_range: List[float],
    lon_range: List[float],
) -> None:
    c = cdsapi.Client()

    # Format the date
    date = f"{year}-{month:02d}"

    # Prepare the request
    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": variables,
        "pressure_level": pressure_levels,
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": [lat_range[0], lon_range[0], lat_range[1], lon_range[1]],
    }

    # Prepare the output filename
    output_file = (
        f"{date}_lat{lat_range[0]}-{lat_range[1]}_lon{lon_range[0]}-{lon_range[1]}.nc"
    )
    output_path = os.path.join(output_dir, output_file)

    # Check if the file already exists
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping download.")
        return

    print(
        f"Downloading ERA5 pressure level data for {date} (lat: {lat_range}, lon: {lon_range})..."
    )
    c.retrieve("reanalysis-era5-pressure-levels", request, output_path)
    print(f"Download complete. File saved as {output_path}")


def download_worker(args):
    year, month, variables, pressure_levels, output_dir, lat_range, lon_range = args
    download_era5_pl_data(
        year, month, variables, pressure_levels, output_dir, lat_range, lon_range
    )


def download_era5_pl_data_parallel(
    start_year: int,
    end_year: int,
    variables: List[str],
    pressure_levels: List[str],
    output_dir: str,
    lat_range: List[float],
    lon_range: List[float],
    num_processes: int = 4,
) -> None:
    # Generate all year-month combinations
    date_ranges = [
        (year, month)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Prepare arguments for each worker
    args_list = [
        (year, month, variables, pressure_levels, output_dir, lat_range, lon_range)
        for year, month in date_ranges
    ]

    # Map the download function to the pool of workers
    pool.map(download_worker, args_list)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 pressure level data")
    parser.add_argument(
        "--start_year", type=int, help="Start year to download data for"
    )
    parser.add_argument("--end_year", type=int, help="End year to download data for")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["u_component_of_wind", "v_component_of_wind"],
        help="Variables to download",
    )
    parser.add_argument(
        "--pressure_levels",
        nargs="+",
        default=["1000", "850", "700"],
        help="Pressure levels to download",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/mm-era5",
        help="Directory to save downloaded files",
    )
    parser.add_argument(
        "--lat_range",
        nargs=2,
        type=float,
        default=[49, 25],
        help="Latitude range [North, South]",
    )
    parser.add_argument(
        "--lon_range",
        nargs=2,
        type=float,
        default=[-125, -66],
        help="Longitude range [West, East]",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of processes to use for parallel download",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    download_era5_pl_data_parallel(
        args.start_year,
        args.end_year,
        args.variables,
        args.pressure_levels,
        args.output_dir,
        args.lat_range,
        args.lon_range,
        args.num_processes,
    )
