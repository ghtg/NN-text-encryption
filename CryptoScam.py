import tensorflow as tf
import numpy as np
import pandas as pd
import os

# Параметры
MAX_SEQ_LENGTH = 10
CHARACTER_SET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-=+_!@#$%^&*()|><❤️✨✅🔥🎉😂😊⭐😭،؛؟ءآأل∀∁∂∃∄∅∆∇∈∉∊∋∎∑∓∔√∛∜∞∫∬∭∮∯∰∱∲∳∴∵∸∾∿абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
NUM_CLASSES = len(CHARACTER_SET)
EMBEDDING_DIM = 16
HIDDEN_UNITS = 64
LEARNING_RATE = 0.00001
EPOCHS = 10000

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

def create_models():
    # Encoder model
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,)),
        tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIM),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=1, padding='same'),
        tf.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    ])
    
    # Decoder model
    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,)),
        tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIM),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=1, padding='same'),
        tf.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    ])
    
    return encoder, decoder

def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

def learn(filename, epochs):
    x_data, y_data, idx_to_char = preprocess_data(filename)
    
    # Создание моделей, если они не загружены
    encoder, decoder = create_models()
    
    # Проверка наличия сохраненной модели и загрузка весов
    if os.path.exists('encoder_model.keras'):
        encoder = tf.keras.models.load_model('encoder_model.keras', compile=False)
        # Пересоздаем оптимизатор и компилируем модель заново
        encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        compile_model(encoder)  # Компилируем новую модель, если не загружена
    
    if os.path.exists('decoder_model.keras'):
        decoder = tf.keras.models.load_model('decoder_model.keras', compile=False)
        # Пересоздаем оптимизатор и компилируем модель заново
        decoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        compile_model(decoder)  # Компилируем новую модель, если не загружена
    
    for epoch in range(epochs):
        # Обучение шифратора (encoder)
        encoder.fit(x_data, np.expand_dims(y_data, -1), epochs=1, batch_size=64)
        
        # Прогнозирование зашифрованного текста
        encoder_predictions = encoder.predict(x_data)
        
        # Обучение дешифратора (decoder)
        decoder.fit(np.expand_dims(y_data, -1), x_data, epochs=1, batch_size=64)
        
        # Сохранение моделей каждые 100 эпох
        if (epoch + 1) % 100 == 0:
            encoder.save('encoder_model.keras')
            decoder.save('decoder_model.keras')
            print(f'Epoch {epoch + 1}/{epochs} - Models saved.')
    
    # Финальное сохранение моделей
    encoder.save('encoder_model.keras')
    decoder.save('decoder_model.keras')
    
    return idx_to_char


def encrypt(text, encoder, idx_to_char):
    char_to_idx = {char: idx for idx, char in enumerate(CHARACTER_SET)}
    seq = np.array([char_to_idx.get(c, 0) for c in text]).reshape(1, -1)
    pred = encoder.predict(seq)
    pred_seq = np.argmax(pred, axis=-1).flatten()
    encrypted_text = ''.join([idx_to_char.get(idx, '?') for idx in pred_seq])
    return encrypted_text.strip()

def decrypt(text, decoder, idx_to_char):
    char_to_idx = {char: idx for idx, char in enumerate(CHARACTER_SET)}
    seq = np.array([char_to_idx.get(c, 0) for c in text]).reshape(1, -1)
    pred = decoder.predict(seq)
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
    idx_to_char = learn('dataset.csv', EPOCHS)
    
    # Загрузка обученных моделей
    encoder_model = tf.keras.models.load_model('encoder_model.keras')
    decoder_model = tf.keras.models.load_model('decoder_model.keras')

    # Пример шифрования и расшифрования
    test_texts = ['ghtg_crazy','TowerBattl','cryptoscam','abcdefghij','vsemprivet','Vsem_Prive','kOmfOrtIkI','LokoMarina','1234567890','0NoskiMoi0','crackedboo','CodeMaster','HelloWorld','funny_game','skyHigh202','alphaOmega','RoboKnight','DragonSlay','SuperHero+','crazy_logi','TowerFall1','StealthNin','CyberPunk_','SpaceInvad','NeonDreame','VoidWalker','LaserKnigh','NochDvoem','vsEmPRIVET','zdravstvui','NochDvoyem','nochkadvoy','segodnya!!','zdravtiLyu','!RobloxBAN','scri!pters','cryptissas','cryptcscam','1231231231','Sooooooooo','Rak1NaDiva','Encryption','ShreksOuse','FRIENDSHIP','I_HATE_YOU','I_LOVE_YOU','8937273625','creATE_MAC','ok','skies','zdorova!','zdrasti','zdraste','privetikvs','zdrast','zdravst','zaebal','ZavaliEbal','Adventure','Basketball','Curriculum','Democracy','Diligently','Electronic','Expansions','Friendship','Generation','Geography','Historical','Innovation','Journalism','Leadership','Literature','Navigation','Optimistic','Perception','Publishing','Revolution','Reflection','Sensations','Speculator','Superhuman','Television','Terminolog','Translatio','Vegetarian','Volleyball','Watermelon','-=+_!@#$%^','&*()|><❤️✨','∀∁∂∃∄∅∆∇∈∉','∊∋∎∑∓∔√∛∜∞','√∛∜∞∫∬∭∮∯∰','∰∱∲∳∴∵∸∾∿','happy_day','zvezdochka','ljubov_pri','naxodka!','zdraste','privetik','super_bob','ne_vernus!','красава123','привет❤️','жизнь_боль','счастливчи','учёба_хард','давай_пого','огонь!!!','не_сдамся','полный_фа','звезды_на_','влюблённый','научная_с','Любимыыыый','GENESIS182','Genesis211','52SquadGo!','мама']
    
    # Открываем файл в режиме добавления
    errors = 0

    # Открываем файл в режиме добавления с указанием кодировки utf-8
    with open('errors_log.txt', 'a', encoding='utf-8') as file:
        for text in test_texts:
            encrypted = encrypt(text, encoder_model, idx_to_char)
            decrypted = decrypt(encrypted, decoder_model, idx_to_char)
            
            # Проверка, есть ли данные в датасете
            in_dataset = text in dataset
            if in_dataset:
                expected_encrypted = dataset[text]
                is_encrypted_correct = (encrypted == expected_encrypted)
                is_decrypted_correct = (decrypted == text)
                
                if not is_encrypted_correct:
                    # Записываем данные в файл, если расшифровка неверна
                    file.write(f'{text},{encrypted},{decrypted}\n')
                    file.write(f'In Dataset: Yes, Expected Encrypted: {expected_encrypted}\n')
                    file.write(f'Encryption Correct: {is_encrypted_correct}, Decryption Correct: {is_decrypted_correct}\n')
                    errors += 1
            else:
                # Записываем данные в файл, если текст отсутствует в датасете
                file.write(f'{text},{encrypted}\n')
        
        # Записываем количество ошибок в файл
        file.write(f'Errors: {errors}\n')
