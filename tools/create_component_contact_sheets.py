"""Create labelled contact sheets for Third Eye facial components."""

from pathlib import Path
from math import ceil

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPONENT_ROOT = PROJECT_ROOT / "static" / "assets" / "face_components"
OUTPUT_ROOT = COMPONENT_ROOT / "contact_sheets"

CATEGORIES = [
    "heads",
    "hair",
    "eyes",
    "eyebrows",
    "noses",
    "lips",
    "moustaches",
    "beards",
    "ears",
    "necks",
    "glasses",
    "marks",
]

COLUMNS = 4
CELL_WIDTH = 260
CELL_HEIGHT = 220
IMAGE_WIDTH = 220
IMAGE_HEIGHT = 165
HEADER_HEIGHT = 55
BACKGROUND = (245, 245, 245)
CELL_BACKGROUND = (255, 255, 255)
TEXT_COLOUR = (25, 25, 25)
BORDER_COLOUR = (190, 190, 190)


def load_font(size):
    """Load an available Windows font, falling back to PIL's default."""
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]

    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)

    return ImageFont.load_default()


def place_on_white(image, width, height):
    """Thumbnail an image and place it on a white RGBA canvas."""
    image = image.convert("RGBA")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    x = (width - image.width) // 2
    y = (height - image.height) // 2
    canvas.alpha_composite(image, (x, y))
    return canvas.convert("RGB")


def create_contact_sheet(category):
    """Create one contact sheet and return its output path, or None."""
    category_dir = COMPONENT_ROOT / category
    files = sorted(category_dir.glob("*.png"))

    if not files:
        print(f"Skipping {category}: no PNG files")
        return None

    rows = ceil(len(files) / COLUMNS)
    sheet_width = COLUMNS * CELL_WIDTH
    sheet_height = HEADER_HEIGHT + rows * CELL_HEIGHT

    sheet = Image.new("RGB", (sheet_width, sheet_height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(28)
    label_font = load_font(18)

    draw.text(
        (16, 12),
        f"{category.upper()} — {len(files)} COMPONENTS",
        fill=TEXT_COLOUR,
        font=title_font,
    )

    for index, path in enumerate(files):
        row = index // COLUMNS
        column = index % COLUMNS
        x = column * CELL_WIDTH
        y = HEADER_HEIGHT + row * CELL_HEIGHT

        draw.rectangle(
            (x + 6, y + 6, x + CELL_WIDTH - 6, y + CELL_HEIGHT - 6),
            fill=CELL_BACKGROUND,
            outline=BORDER_COLOUR,
            width=2,
        )

        with Image.open(path) as image:
            preview = place_on_white(image, IMAGE_WIDTH, IMAGE_HEIGHT)

        image_x = x + (CELL_WIDTH - IMAGE_WIDTH) // 2
        image_y = y + 14
        sheet.paste(preview, (image_x, image_y))

        label = path.name
        label_box = draw.textbbox((0, 0), label, font=label_font)
        label_width = label_box[2] - label_box[0]
        label_x = x + (CELL_WIDTH - label_width) // 2
        label_y = y + IMAGE_HEIGHT + 25
        draw.text((label_x, label_y), label, fill=TEXT_COLOUR, font=label_font)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{category}_contact_sheet.jpg"
    sheet.save(output_path, quality=94)
    return output_path


def main():
    if not COMPONENT_ROOT.exists():
        raise FileNotFoundError(f"Component directory does not exist: {COMPONENT_ROOT}")

    print(f"Component root: {COMPONENT_ROOT}")
    print(f"Output root:    {OUTPUT_ROOT}")
    print()

    created = []

    for category in CATEGORIES:
        output_path = create_contact_sheet(category)
        if output_path is not None:
            created.append(output_path)
            print(f"Created: {output_path}")

    print()
    print(f"Created {len(created)} contact sheets.")


if __name__ == "__main__":
    main()
