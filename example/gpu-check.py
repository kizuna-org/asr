import sys
from typing import List
import torch


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python gpu-check.py <input_file> <output_file>")
        sys.exit(1)
    input_file: str = sys.argv[1]
    output_file: str = sys.argv[2]

    if not torch.cuda.is_available():
        print("CUDA GPU is not available. Exiting.")
        sys.exit(1)
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    numbers: List[float] = []
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        numbers.append(float(line))
                    except ValueError:
                        print(f"Warning: Invalid number on line {line_num}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    if not numbers:
        print("Error: No valid numbers found in input file")
        sys.exit(1)

    tensor = torch.tensor(numbers, device="cuda")
    squared = torch.pow(tensor, 2)
    squared_cpu = squared.cpu().numpy()

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for num in squared_cpu:
                f.write(f"{num}\n")
    except IOError as e:
        print(f"Error writing to output file '{output_file}': {e}")
        sys.exit(1)

    print(f"Processed {len(numbers)} numbers.")
    print(f"Results written to {output_file}.")


if __name__ == "__main__":
    main()
