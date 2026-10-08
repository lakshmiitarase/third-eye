from feature_selector import select_feature


face = select_feature(
    "face shapes",
    "oval"
)

eyebrows = select_feature(
    "eyebrows",
    "high arch"
)

print("\nFINAL SELECTION")

print("Face:", face)

print("Eyebrows:", eyebrows)