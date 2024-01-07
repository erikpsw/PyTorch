import pytesseract
from PIL import Image
image = Image.open('mine.bpm')
code = pytesseract.image_to_string(image, lang="eng")
print(code)