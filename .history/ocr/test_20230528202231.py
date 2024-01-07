import pytesseract
from PIL import Image
image = Image.open('test.jpg')
code = pytesseract.image_to_string(image, lang="eng")
print(code)