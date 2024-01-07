import pytesseract
from PIL import Image
image = Image.open('experiment/MVIMG_20230528_194348.jpg')
code = pytesseract.image_to_string(image, lang="eng")
print(code)