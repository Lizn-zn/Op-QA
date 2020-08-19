from PIL import Image

im = Image.open('p/2008_000133.jpg')
# im = Image.open('p/2009_002649.jpg')
# im = Image.open('c/224_0003.jpg')
# im = Image.open('c/246_0043.jpg')
im = im.resize((224, 224))
im.save('opt1.png')
