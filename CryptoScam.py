import tensorflow as tf
import numpy as np
import pandas as pd
import os

# Параметры
MAX_SEQ_LENGTH = 10
CHARACTER_SET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-=+_!@#$%^&*()|><❤️✨✅🔥🎉😂😊⭐😭،؛؟ءآأل∀∁∂∃∄∅∆∇∈∉∊∋∎∑∓∔√∛∜∞∫∬∭∮∯∰∱∲∳∴∵∸∾∿"
NUM_CLASSES = len(CHARACTER_SET)
EMBEDDING_DIM = 16
HIDDEN_UNITS = 64
LEARNING_RATE = 0.01
EPOCHS = 100

def preprocess_data(filename):
    df = pd.read_csv(filename, header=None)
    decrypt_texts = df[0].values
    encrypt_texts = df[1].values
    
    # Создание словарей для преобразования символов в индексы
    char_to_idx = {char: idx for idx, char in enumerate(CHARACTER_SET)}
    idx_to_char = {idx: char for idx, char in enumerate(CHARACTER_SET)}
    
    def text_to_sequence(text, max_length):
        seq = [char_to_idx.get(c, 0) for c in text]  # 0 для неизвестных символов
        return seq + [0] * (max_length - len(seq))  # Дополнение до max_length
    
    x_data = np.array([text_to_sequence(t, MAX_SEQ_LENGTH) for t in decrypt_texts])
    y_data = np.array([text_to_sequence(t, MAX_SEQ_LENGTH) for t in encrypt_texts])
    
    return x_data, y_data, idx_to_char

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,)),
        tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIM),
        tf.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    ])
    return model

def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

def learn(filename, epochs, learnOnlyDecoder):
    x_data, y_data, idx_to_char = preprocess_data(filename)
    
    if learnOnlyDecoder == 'false':
        # Создание и настройка модели шифратора
        encoder_model = create_model()
        compile_model(encoder_model)
        
        # Проверка наличия сохраненной модели и загрузка весов
        if os.path.exists('encoder_model.keras'):
            encoder_model = tf.keras.models.load_model('encoder_model.keras')
        
        # Обучение модели
        encoder_model.fit(x_data, np.expand_dims(y_data, -1), epochs=epochs, batch_size=64, initial_epoch=0 if not os.path.exists('encoder_model.keras') else 1)
        encoder_model.save('encoder_model.keras')
    
    # Создание и настройка модели дешифратора
    decoder_model = create_model()
    compile_model(decoder_model)
    
    # Проверка наличия сохраненной модели и загрузка весов
    if os.path.exists('decoder_model.keras'):
        decoder_model = tf.keras.models.load_model('decoder_model.keras')
    
    # Обучение модели
    decoder_model.fit(np.expand_dims(y_data, -1), x_data, epochs=epochs, batch_size=64, initial_epoch=0 if not os.path.exists('decoder_model.keras') else 1)
    decoder_model.save('decoder_model.keras')
    
    return idx_to_char

def encrypt(text, model, idx_to_char):
    char_to_idx = {char: idx for idx, char in enumerate(CHARACTER_SET)}
    seq = np.array([char_to_idx.get(c, 0) for c in text]).reshape(1, -1)
    pred = model.predict(seq)
    pred_seq = np.argmax(pred, axis=-1).flatten()
    encrypted_text = ''.join([idx_to_char.get(idx, '?') for idx in pred_seq])
    return encrypted_text.strip()

def decrypt(text, model, idx_to_char):
    char_to_idx = {char: idx for idx, char in enumerate(CHARACTER_SET)}
    seq = np.array([char_to_idx.get(c, 0) for c in text]).reshape(1, -1)
    pred = model.predict(seq)
    pred_seq = np.argmax(pred, axis=-1).flatten()
    decrypted_text = ''.join([idx_to_char.get(idx, '?') for idx in pred_seq])
    return decrypted_text.strip()

def load_dataset(filename):
    """Загружает датасет и возвращает словарь для быстрого поиска."""
    df = pd.read_csv(filename, header=None)
    dataset = {row[0]: row[1] for _, row in df.iterrows()}  # {'расшифрованный': 'зашифрованный'}
    return dataset

if __name__ == "__main__":
    # Загрузка датасета для проверки
    dataset = load_dataset('dataset.csv')
    
    # Обучение моделей
    idx_to_char = learn('dataset.csv', EPOCHS, 'fals')
    
    # Загрузка обученных моделей
    encoder_model = tf.keras.models.load_model('encoder_model.keras')
    decoder_model = tf.keras.models.load_model('decoder_model.keras')

    # Пример шифрования и расшифрования
    #test_texts = ['vsEmPRIVET','zdravstvui','NochDvoyem', 'nochkadvoy', 'zdravtiLyu', '!RobloxBAN', 'ShreksOuse', '8937273625', 'I_LOVE_YOU', 'I_HATE_YOU','creATE_MAC','zdraste','zdrasti','zdorova!', 'zdrast', 'zdravst', 'zaebal','ZavaliEbal']
    test_texts = ['Adventure','Basketball','Diligently','Expansions','Generation','Geography','Innovation','Journalism','Leadership','Optimistic','Perception','Publishing','Reflection','Speculator','Translatio','Watermelon']
    #test_texts = ['-=+_!@#$%^','&*()|><❤️✨','✅🔥🎉😂😊⭐😭،؛؟','∀∁∂∃∄∅∆∇∈∉','∊∋∎∑∓∔√∛∜∞','√∛∜∞∫∬∭∮∯∰','∰∱∲∳∴∵∸∾∿']
    
    for text in test_texts:
        encrypted = encrypt(text, encoder_model, idx_to_char)
        decrypted = decrypt(encrypted, decoder_model, idx_to_char)
        
        # Проверка, есть ли данные в датасете
        in_dataset = text in dataset
        if in_dataset:
            expected_encrypted = dataset[text]
            is_encrypted_correct = (encrypted == expected_encrypted)
            is_decrypted_correct = (decrypted == text)
            if (is_encrypted_correct == False or is_decrypted_correct == False):
                print(f'Original: {text}, Encrypted: {encrypted}, Decrypted: {decrypted}')
                print(f'In Dataset: Yes, Expected Encrypted: {expected_encrypted}')
                print(f'Encryption Correct: {is_encrypted_correct}, Decryption Correct: {is_decrypted_correct}')
        else:
            print(f'Original: {text}, Encrypted: {encrypted}, Decrypted: {decrypted}')
            print(f'In Dataset: No')