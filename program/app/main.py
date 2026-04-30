import argparse

from processor.processor import Processor

def main():
  parser = argparse.ArgumentParser(description="LSTM configuration")
  parser.add_argument("-c", "--config", required=True, help="Path to config Yaml file")
  args = parser.parse_args()

  processor = Processor(args.config)

  processor.execute()

if __name__ == "__main__":
  main()