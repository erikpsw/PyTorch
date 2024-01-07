import pytesseract
from PIL import Image
image = Image.open(r'test.jpg')
# code = pytesseract.image_to_string(image, lang="eng")
# print(code)