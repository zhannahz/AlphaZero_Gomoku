import os
import csv

abs_dir = os.path.dirname(os.path.abspath(__file__)) + "/Data/"
print (abs_dir)
def main():
    data = open_data_all()
    print(data)

def open_data_all():
    """Open all data files."""
    files = os.listdir(abs_dir)
    if files is None:
        print("No data files found.")
        return None
    d = []
    for file in files:
        if file.endswith(".txt"):
            d.append(open_data(os.path.join(abs_dir, file)))
    return d

def open_data(file):
    """Open a data file."""
    with open(file, 'r') as f:
        return f.readlines()

if __name__ == "__main__":
    main()
