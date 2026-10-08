from description_parser import extract_features


description = """
Female with an oval face,
eye7 eyes,
nose14 nose,
and high arch eyebrows
"""

features = extract_features(description)

print("\nEXTRACTED FEATURES")

print("Face:", features["face"])
print("Eyes:", features["eyes"])
print("Nose:", features["nose"])
print("Eyebrows:", features["eyebrows"])