'''
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
'''

'''
# List — an ordered collection
medications = ["Aspirin", "Paracetamol", "Ibuprofen"]

for med in medications:
    print(med)

# Dict — key-value pairs
prescription = {
    "patient": "John Doe",
    "medication": "Aspirin",
    "dosage": "500mg",
    "frequency": "twice daily"
}

for key, value in prescription.items():
    print(f"{key}: {value}")

# List of dicts — this is likely how your parsed prescriptions will look
prescriptions = [
    {"medication": "Aspirin", "dosage": "500mg"},
    {"medication": "Paracetamol", "dosage": "650mg"},
]

for p in prescriptions:
    print(f"{p['medication']} — {p['dosage']}")

'''

'''
def extract_dosage(text):
    # dummy logic for now
    if "500mg" in text:
        return "500mg"
    return "unknown"

def parse_prescription(line):
    parts = line.split(",")
    return {
        "medication": parts[0].strip(),
        "dosage": parts[1].strip() if len(parts) > 1 else "unknown"
    }

result = parse_prescription("Aspirin, 500mg")
print(result)  # {'medication': 'Aspirin', 'dosage': '500mg'}

'''
