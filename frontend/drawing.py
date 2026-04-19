def draw_lines(uploaded_file, result):
    from PIL import Image, ImageDraw
    import io

    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    draw = ImageDraw.Draw(image)

    for line in result.get("lines", []):
        bbox = line.get("bbox", line)
        draw.rectangle(
            [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
            outline="red",
            width=2,
        )
    return image
