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
