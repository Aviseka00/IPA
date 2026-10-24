"""Build the pictures for the operator guide from the practice slide."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SRC = Path(
    r"C:\Users\80100173\.cursor\projects\c-Users-80100173-Ani-v0-4-0\assets"
    r"\c__Users_80100173_AppData_Roaming_Cursor_User_workspaceStorage_e55705b2c573290bfebbcbfa2356f783_images_1-9c3b2f8c-b3d3-485e-bda1-35ff13d2d2e2.jpg"
)
OUT = ROOT / "figures"
INK = (28, 32, 38)
GREEN = (36, 140, 90)
RED = (176, 48, 42)
BLUE = (69, 96, 196)
GOLD = (196, 132, 32)
WHITE = (255, 255, 255)


def font(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def label(draw, xy, text, fill=INK, bg=WHITE, size=22, bold=True, anchor="lt"):
    face = font(size, bold)
    box = draw.textbbox(xy, text, font=face, anchor=anchor)
    pad = 6
    draw.rounded_rectangle(
        (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
        radius=8, fill=bg,
    )
    draw.text(xy, text, font=face, fill=fill, anchor=anchor)


def ring(draw, center, radius, color, width=4):
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)


def cross(draw, center, radius, color, width=4):
    x, y = center
    draw.line((x - radius, y - radius, x + radius, y + radius), fill=color, width=width)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill=color, width=width)


def save(image, name):
    path = OUT / name
    image.convert("RGB").save(path, quality=92)
    return path


def practice_field(image):
    frame = image.copy()
    draw = ImageDraw.Draw(frame)
    # Three real dark clumps on this slide, plus one clean area of cells.
    debris = [(86, 266, "lt"), (289, 410, "lt"), (921, 263, "rt")]
    for index, (x, y, side) in enumerate(debris, start=1):
        ring(draw, (x, y), 28, RED, 5)
        if side == "rt":
            label(draw, (x - 36, y), f"Do not mark {index}", fill=RED, size=20, anchor="rm")
        else:
            label(draw, (x + 36, y), f"Do not mark {index}", fill=RED, size=20, anchor="lm")
    draw.rounded_rectangle((430, 90, 750, 390), radius=12, outline=GREEN, width=5)
    label(draw, (440, 78), "Cells live in here", fill=GREEN, size=22, anchor="lb")
    return save(frame, "01-practice-field.jpg")


def mark_or_skip(image):
    crop = image.crop((20, 180, 340, 480))
    draw = ImageDraw.Draw(crop)
    # Crop origin is (20, 180). Debris at full-image (86, 266) and (289, 410).
    cross(draw, (66, 86), 16, RED, 5)
    label(draw, (86, 86), "Skip", fill=RED, size=18, anchor="lm")
    cross(draw, (269, 230), 16, RED, 5)
    label(draw, (248, 230), "Skip", fill=RED, size=18, anchor="rm")
    ring(draw, (150, 150), 18, GREEN, 4)
    label(draw, (174, 150), "Outline", fill=GREEN, size=18, anchor="lm")
    return save(crop, "02-mark-or-skip.jpg")


def panel():
    width, height = 420, 980
    card = Image.new("RGB", (width, height), (35, 36, 43))
    draw = ImageDraw.Draw(card)
    title = font(28, True)
    body = font(16, False)
    small = font(14, False)
    draw.text((24, 18), "Ani", font=title, fill=WHITE)
    draw.text((24, 54), "The panel on the right of the window", font=body, fill=(183, 192, 204))

    sections = [
        (1, "Image", "Open image", "Then Previous, Next, or Recent"),
        (2, "Class for new objects", "cell", "Add class if you need another name"),
        (3, "Segment", "Classic, then Segment", "Labels the whole picture"),
        (4, "Draw", "Glide   Points   Pan", "Brush   Erase   Click"),
        (5, "Fix", "Fill holes    Check", "Fit    Contrast"),
        (6, "Save", "Save annotations", "Pick the file type in the list"),
        (7, "Feature extractor", "Extract features", "Optional sizes and a training table"),
    ]
    y = 100
    for number, kicker, main, note in sections:
        draw.rounded_rectangle((16, y, width - 16, y + 108), radius=12, fill=(46, 51, 62))
        draw.ellipse((28, y + 16, 64, y + 52), fill=BLUE)
        draw.text((46, 34 + y), str(number), font=font(18, True), fill=WHITE, anchor="mm")
        draw.text((76, y + 14), kicker.upper(), font=small, fill=(183, 192, 204))
        draw.rounded_rectangle((76, y + 40, width - 32, y + 74), radius=8, fill=BLUE if number in (1, 6) else (35, 36, 43))
        ink = WHITE
        draw.text((88, y + 48), main, font=font(16, True), fill=ink)
        draw.text((76, y + 80), note, font=small, fill=(183, 192, 204))
        y += 120
    return save(card, "03-panel.png")


def _cell_polygon(center, scale=1.0):
    cx, cy = center
    pts = [(0, -16), (14, -8), (16, 8), (6, 16), (-10, 14), (-16, 2), (-8, -12)]
    return [(cx + int(x * scale), cy + int(y * scale)) for x, y in pts]


def glide(image):
    crop = image.crop((420, 80, 760, 400)).resize((680, 640), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(crop, "RGBA")
    outline = _cell_polygon((250, 250), 2.2)
    draw.line(outline + [outline[0]], fill=GREEN + (255,), width=6)
    draw.ellipse((outline[0][0] - 7, outline[0][1] - 7, outline[0][0] + 7, outline[0][1] + 7), fill=GREEN)
    label(draw, (24, 24), "Hold Shift and drag around the cell", fill=GREEN, size=24)
    label(draw, (24, 70), "Let go when you are back at the start", size=20, bold=False)
    return save(crop, "04-glide.jpg")


def points(image):
    crop = image.crop((420, 80, 760, 400)).resize((680, 640), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(crop, "RGBA")
    outline = _cell_polygon((420, 300), 2.4)
    draw.line(outline, fill=BLUE + (255,), width=5)
    for index, (x, y) in enumerate(outline, start=1):
        draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=BLUE)
        draw.text((x, y), str(index), font=font(16, True), fill=WHITE, anchor="mm")
    label(draw, (24, 24), "Ctrl + click each corner", fill=BLUE, size=24)
    label(draw, (24, 70), "Click corner 1 again, or double-click, to close", size=20, bold=False)
    return save(crop, "05-points.jpg")


def classes(image):
    crop = image.crop((420, 80, 760, 400)).resize((680, 640), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(crop, "RGBA")
    groups = [
        ((230, 220), GREEN, "healthy"),
        ((430, 250), RED, "dead"),
        ((300, 430), GOLD, "infected"),
    ]
    for center, color, name in groups:
        outline = _cell_polygon(center, 2.0)
        draw.polygon(outline, fill=color + (70,))
        draw.line(outline + [outline[0]], fill=color + (255,), width=5)
    menu_box = (460, 80, 650, 250)
    draw.rounded_rectangle(menu_box, radius=10, fill=(46, 51, 62, 245))
    draw.text((478, 96), "Right-click menu", font=font(16, True), fill=WHITE)
    for offset, name, color in ((140, "healthy", GREEN), (174, "dead", RED), (208, "infected", GOLD)):
        draw.ellipse((478, offset, 496, offset + 18), fill=color)
        draw.text((506, offset - 2), name, font=font(18, False), fill=WHITE)
    label(draw, (24, 24), "Right-click a finished outline", size=24)
    return save(crop, "06-class.jpg")


def _outline_overlay(image, labels, color, width=2):
    import numpy as np
    from skimage.segmentation import find_boundaries

    base = np.array(image.convert("RGB"))
    edges = find_boundaries(np.asarray(labels), mode="outer")
    if width > 1:
        from scipy import ndimage as ndi
        edges = ndi.binary_dilation(edges, iterations=width - 1)
    base[edges] = color
    return Image.fromarray(base)


def autosegment(image):
    import numpy as np
    from ani.segment import run_segment

    labels = run_segment(np.array(image), "classic")
    count = int(np.count_nonzero(np.unique(labels)))
    crop_box = (430, 90, 750, 390)
    crop_labels = labels[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    framed = _outline_overlay(image.crop(crop_box), crop_labels, (36, 140, 90), width=1)
    framed = framed.resize((640, 600), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(framed)
    label(draw, (16, 16), f"Classic draft: {count} objects on the whole slide", fill=GREEN, size=20)
    label(draw, (16, 56), "Then correct it with Brush and Erase.", size=18, bold=False)
    return save(framed, "08-autosegment.jpg")


def click_one(image):
    import numpy as np
    from ani.segment import run_click

    blank = np.zeros(image.size[::-1], np.uint16)
    point = (200, 560)
    mask, box = run_click(np.array(image), blank, point, "classic")
    shown = image.copy()
    if mask is not None:
        full = np.zeros(blank.shape, np.uint16)
        full[box][np.asarray(mask)] = 1
        shown = _outline_overlay(shown, full, (69, 96, 196), width=2)
        y = (box[0].start + box[0].stop) // 2
        x = (box[1].start + box[1].stop) // 2
        crop = shown.crop((max(0, x - 180), max(0, y - 160), min(image.width, x + 180), min(image.height, y + 160)))
    else:
        crop = shown.crop((430, 90, 750, 390))
    crop = crop.resize((680, 560), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(crop)
    label(draw, (16, 16), "Click tool: one object, nothing else changes", fill=BLUE, size=20)
    return save(crop, "09-click.jpg")


def brush_erase(image):
    import numpy as np
    from ani.segment import run_click

    blank = np.zeros(image.size[::-1], np.uint16)
    mask, box = run_click(np.array(image), blank, (200, 560), "classic")
    shown = image.copy()
    if mask is not None:
        full = np.zeros(blank.shape, np.uint16)
        full[box][np.asarray(mask)] = 1
        shown = _outline_overlay(shown, full, (36, 140, 90), width=2)
        y = (box[0].start + box[0].stop) // 2
        x = (box[1].start + box[1].stop) // 2
        crop = shown.crop((max(0, x - 180), max(0, y - 160), min(image.width, x + 180), min(image.height, y + 160)))
    else:
        x, y = 560, 200
        crop = shown.crop((x - 180, y - 160, x + 180, y + 160))
    crop = crop.resize((680, 560), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(crop)
    draw.ellipse((470, 230, 560, 320), outline=BLUE, width=8)
    label(draw, (470, 200), "Brush adds", fill=BLUE, size=20, anchor="lb")
    cross(draw, (180, 360), 22, RED, 6)
    label(draw, (210, 360), "Erase removes", fill=RED, size=20, anchor="lm")
    return save(crop, "10-brush.jpg")


def features_card():
    card = Image.new("RGB", (760, 520), (245, 246, 248))
    draw = ImageDraw.Draw(card)
    draw.text((28, 20), "Feature extractor", font=font(26, True), fill=INK)
    fields = ["Well", "Slide", "Magnification", "µm per pixel"]
    x = 28
    for name in fields:
        draw.rounded_rectangle((x, 70, x + 168, 112), radius=8, fill=WHITE, outline=(210, 214, 220))
        draw.text((x + 12, 82), name, font=font(16, False), fill=(90, 98, 110))
        x += 180
    draw.rounded_rectangle((28, 132, 280, 176), radius=8, fill=BLUE)
    draw.text((154, 154), "Extract features", font=font(18, True), fill=WHITE, anchor="mm")
    rows = [
        ("features.csv", "One row per cell. The class column is the training label."),
        ("patches/", "A small colour crop of each cell."),
        ("masks/", "The cell, the nucleus estimate, and the cytoplasm."),
        ("cnn/", "A 128×128 picture of each cell for a network."),
        ("qc/", "A check picture with the outline and the cell number."),
    ]
    y = 196
    for name, note in rows:
        draw.text((28, y), name, font=font(16, True), fill=BLUE)
        draw.text((170, y), note, font=font(16, False), fill=INK)
        y += 36
    draw.text((28, 470), "Leave µm per pixel blank when you do not know the scale.", font=font(15, False), fill=(90, 98, 110))
    return save(card, "11-features.png")


def save_types():
    card = Image.new("RGB", (760, 420), (245, 246, 248))
    draw = ImageDraw.Draw(card)
    draw.text((28, 24), "In Save annotations, open the file-type list", font=font(26, True), fill=INK)
    rows = [
        ("seg.npy", "Best choice inside Ani. Open it again later and keep working."),
        ("TIFF or PNG", "A picture of the outlines, plus a small class file."),
        ("COCO", "For many training programs that expect COCO JSON."),
        ("YOLO", "One text file of outline points for YOLO training."),
        ("All of them", "Writes seg.npy, the class file, COCO, and YOLO together."),
    ]
    y = 80
    for name, note in rows:
        draw.rounded_rectangle((28, y, 732, y + 58), radius=10, fill=WHITE, outline=(210, 214, 220))
        draw.text((44, y + 8), name, font=font(18, True), fill=BLUE)
        draw.text((44, y + 32), note, font=font(15, False), fill=(70, 78, 90))
        y += 66
    return save(card, "07-save.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    image = Image.open(SRC).convert("RGB")
    image.save(ROOT / "demo.jpg", quality=95)
    practice_field(image)
    mark_or_skip(image)
    panel()
    glide(image)
    points(image)
    classes(image)
    save_types()
    autosegment(image)
    click_one(image)
    brush_erase(image)
    features_card()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
