# Open the file for reading
with open('data.txt', 'r') as f:
    # Read lines into a list, one line per item
    data = f.readlines()

# Optionally, you can remove any trailing whitespace (including newlines) from each item
data = [line.strip() for line in data]

# Now, data is a list where each item corresponds to a line from the file
print(data)
