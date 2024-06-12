from PIL import Image, ImageFilter
import numpy as np

def sobel_operator(image):
    Kx = np.array([[ -1, 0, 1], 
                   [ -2, 0, 2], 
                   [ -1, 0, 1]])

    Ky = np.array([[  1,  2,  1], 
                   [  0,  0,  0], 
                   [ -1, -2, -1]])

    width, height = image.size
    gray_image = image.convert('L')
    gray_pixels = np.array(gray_image)

    Gx = np.zeros_like(gray_pixels)
    Gy = np.zeros_like(gray_pixels)
    edge_pixels = np.zeros_like(gray_pixels, dtype=np.float32)

    for y in range(1, height-1):
        for x in range(1, width-1):
            Gx[y, x] = np.sum(Kx * gray_pixels[y-1:y+2, x-1:x+2])
            Gy[y, x] = np.sum(Ky * gray_pixels[y-1:y+2, x-1:x+2])
            edge_pixels[y, x] = np.sqrt(Gx[y, x]**2 + Gy[y, x]**2)

    edge_pixels = (edge_pixels / np.max(edge_pixels) * 255).astype(np.uint8)

    # Non-maximum suppression
    nms_pixels = np.zeros_like(edge_pixels)
    angle = np.arctan2(Gy, Gx) * (180 / np.pi)
    angle[angle < 0] += 180

    for y in range(1, height-1):
        for x in range(1, width-1):
            try:
                q = 255
                r = 255
                
                # Angle 0
                if (0 <= angle[y, x] < 22.5) or (157.5 <= angle[y, x] <= 180):
                    q = edge_pixels[y, x + 1]
                    r = edge_pixels[y, x - 1]
                # Angle 45
                elif (22.5 <= angle[y, x] < 67.5):
                    q = edge_pixels[y + 1, x - 1]
                    r = edge_pixels[y - 1, x + 1]
                # Angle 90
                elif (67.5 <= angle[y, x] < 112.5):
                    q = edge_pixels[y + 1, x]
                    r = edge_pixels[y - 1, x]
                # Angle 135
                elif (112.5 <= angle[y, x] < 157.5):
                    q = edge_pixels[y - 1, x - 1]
                    r = edge_pixels[y + 1, x + 1]

                if (edge_pixels[y, x] >= q) and (edge_pixels[y, x] >= r):
                    nms_pixels[y, x] = edge_pixels[y, x]
                else:
                    nms_pixels[y, x] = 0

            except IndexError as e:
                pass

    edge_image = Image.fromarray(nms_pixels.astype('uint8'))

    return edge_image

def apply_blur(image, radius):
    filtered_image = image.filter(ImageFilter.BoxBlur(radius))
    return filtered_image

def process_image(image_path, output_path):
    with Image.open(image_path) as original_img:
        original_img = original_img.convert('RGB')
        edited_img = original_img.copy()
        
        original_pixels = original_img.load()
        edited_pixels = edited_img.load()

        width, height = original_img.size

        sobel_img = apply_blur(sobel_operator(original_img), 2)
        sobel_pixels = sobel_img.load()

        for y in range(1, height-1):
            for x in range(1, width-1):
                sobel_value = sobel_pixels[x, y]
                n = int(sobel_value * 200 / 255)  # Map to 0-100
                if n < 50:
                    n = 0
                new_y = min(height-1, y + n)  # Ensure we don't go out of bounds


                for edit_y in range(y, new_y):
                    edited_pixels[x, edit_y] = original_pixels[x, y]

        sobel_img.save("sobel_image.bmp")
        edited_img.save(output_path)

# Example usage
input_image_path = 'input_image.bmp'
output_image_path = 'output_image.bmp'
process_image(input_image_path, output_image_path)
