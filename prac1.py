# Writing to a file
with open("notes.txt", "w") as f:
    f.write("Hello, this is a test.\n")
    f.write("Second line.\n")

# Reading it back
with open("notes.txt", "r") as f:
    content = f.read()
    print(content)

# Reading line by line
with open("notes.txt", "r") as f:
    for line in f:
        print("Line:", line.strip())
