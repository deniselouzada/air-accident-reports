from pypdf import PdfReader
import io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Embedding

reader = PdfReader('hudson-accident.pdf')
num_pages = len(reader.pages)

page = reader.pages[16]
text = page.extract_text()
print(text)

