import sys, os, pandas as pd

def main():
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 2:
        print("\nUsage: python clean_csv.py <input_csv>\n")
        return

    # Get the input file path from command line arguments
    input_csv = sys.argv[1]

    try:
        # Open and read csv
        df = pd.read_csv(input_csv, keep_default_na=False, na_values=['NaN'])
    except FileNotFoundError:
        print(f"Error: File '{input_csv}' not found.")
    except Exception as e:
        print(f"Error: {e}")
    
    # Save dataframe as csv
    base_name, _ = os.path.splitext(input_csv)
    df.to_csv((base_name + '_clean.csv'), index=False)

if __name__ == "__main__":
    main()

