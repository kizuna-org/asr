import torch
import sys

def main():
  if len(sys.argv) != 3:
    print("Usage: python gpu-check.py <input_file> <output_file>")
    sys.exit(1)
  input_file = sys.argv[1]
  output_file = sys.argv[2]

  if not torch.cuda.is_available():
    print("CUDA GPU is not available. Exiting.")
    sys.exit(1)
  else:
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

  with open(input_file, "r") as f:
    numbers = [float(line.strip()) for line in f if line.strip()]

  tensor = torch.tensor(numbers, device="cuda")
  squared = tensor ** 2
  squared_cpu = squared.cpu().numpy()

  with open(output_file, "w") as f:
    for num in squared_cpu:
      f.write(f"{num}\n")

  print(f"Processed {len(numbers)} numbers. Results written to {output_file}.")

if __name__ == "__main__":
  main()
