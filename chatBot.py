import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle
import os


class ChatBot:
    def __init__(self):
        # инициализация класса и загрузка данных при создании обьекта
        self.stemmer = LancasterStemmer()
        if os.path.exists('data.pickle'):
            self.words, self.labels, self.training, self.output = self.load_data()
        else:
            self.words, self.labels, self.training, self.output = self.initialize_data()
        # Сброс графа TensorFlow и построение модели
        tensorflow.compat.v1.reset_default_graph()
        self.net = self.build_model()

        # Загрузка при обучении модели при создании обьекта
        self.model = self.load_or_train_model()

    def load_data(self):
        # Загрузка данных из файла, если файл существует
        with open('data.pickle', 'rb') as pickle_file:
            return pickle.load(pickle_file)
        
    def initialize_data(self):
        # инициализация данных, если файл отсутствует
        words = []
        labels = []
        docs_x = []
        docs_y = []

        for intent in data['intents']:
            for pattern in intent['patterns']:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent['tag'])

                if intent['tag'] not in labels:
                    labels.append(intent['tag'])

        words = [self.stemmer.stem(w.lower()) for w in words if w != '?']
        words = sorted(list(set(words)))

        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [self.stemmer.stem(w) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)
                
            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

            with open('data.pickle', 'wb') as pickle_file:
                pickle.dump((words, labels, training, output), pickle_file)

            return np.array(words), np.array(labels), np.array(training), np.array(output)
    
    def build_model(self):
        # Построение модели нейронной сети
        net = tflearn.input_data(shape=[None, len(self.training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(self.output[0]), activation='softmax')
        net = tflearn.regression(net)
        return net
    
    def load_or_train_model(self):
        # Загрузка сохраненной модели или обучение новой
        model = tflearn.DNN(self.net)
        # раскоментировать после окончания обучени - model.load('model.tflearn')
        # и указать try:
        # model.load('model.tflearn')
        # после обучения перевести на except
        model.fit(self.training, self.output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save('model.tflearn')
        return model
    
    def bags_of_words(self, s):
        # Преобразование входной строки в массив для предсказания модели
        bag = [0 for _ in range(len(self.words))]

        s_words = nltk.word_tokenize(s)
        s_words = [self.stemmer.stem(word.lower()) for word in s_words]

        for si in s_words:
            for i, w in enumerate(self.words):
                if w == si:
                    bag[i] = 1
        return np.array(bag)
    
    def chat(self):
        # Основной цикл общения с ботом
        print("Hello, I'm a bot. Start chatting with me (To terminate chatbot, type: 'quit')")
        while True:
            user_input = input('You: ')
            if user_input.lower() == 'quit':
                break

            results = self.model.predict([self.bags_of_words(user_input)])
            results_index = np.argmax(results)
            tag = self.labels[results_index]

            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))

if __name__ == '__main__':
    nltk.download('punkt')
    # Загрузка файла с намериниями (intents)
    with open('intents.json') as js_file:
        data = json.load(js_file)
    
    # Создание объекта чат-бота и запуск общения
    chat_bot = ChatBot()
    chat_bot.chat()