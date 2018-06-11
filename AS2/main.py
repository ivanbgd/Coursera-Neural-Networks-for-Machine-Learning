from utils import *
from train import train
import pickle

epochs = 10
file_name = 'model.pkl'     # Default model has default values of hyper-parameters: lr = 0.1, momentum = 0.9, etc. Also, epochs = 10.
model = None

try:
    file = open(file_name, 'rb')
    model = pickle.load(file)
except FileNotFoundError:
    file = open(file_name, 'wb')
    model = train(epochs=epochs)
    pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)
finally:
    file.close()

display_nearest_words('school', model, 10)
display_nearest_words('schooool', model, 10)
display_nearest_words('children', model, 10)
display_nearest_words('could', model, 10)
display_nearest_words('dr.', model, 10)
display_nearest_words('day', model, 10)

word1, word2 = 'school', 'university'
wd = word_distance(word1, word2, model)
print("Word distance for '{}' and '{}' is {:.2f}.\n".format(word1, word2, wd))

word1, word2 = 'school', 'court'
wd = word_distance(word1, word2, model)
print("Word distance for '{}' and '{}' is {:.2f}.\n".format(word1, word2, wd))

predict_next_word('john', 'might', 'be', model, 3)
predict_next_word('life', 'in', 'new', model, 3)

