import pytesseract
from PIL import Image
image = Image.open('experiment/1.jpg')
code = pytesseract.image_to_string(image, lang="eng")
print(code)