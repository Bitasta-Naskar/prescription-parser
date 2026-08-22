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
