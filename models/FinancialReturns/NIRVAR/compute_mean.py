#!/usr/bin/env python3
"""
Compute mean, max and min of a single-column CSV.

Usage:
    python col_stats.py data.csv
"""
import csv
import sys
import numpy as np

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    values = []

    # Read the first (only) column, skipping blank lines
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:                      # skip empty rows
                try:
                    values.append(float(row[0]))
                except ValueError:
                    # Ignore rows whose first entry isn’t numeric
                    continue

    if not values:
        print("No numeric values found in the file.")
        sys.exit(1)

    print(f"Mean: {np.mean(values)}")
    print(f"Max : {max(values)}")
    print(f"Min : {min(values)}")

if __name__ == "__main__":
    main()

