import numpy as np
import tensorflow as tf

TEXT_FILE = 'shakespeare_text.txt'
BATCH_SIZE = 64
EMBEDDING_DIM = 256
RNN_UNITS = 1024
SEQ_LENGTH = 150
WEIGHTS_FILE = 'lstm_weights.h5'
TEST_STRING = 'ROMEO: \n'


def prepare_text_vocabulary():
    text = open(TEXT_FILE, 'r').read()
    vocabulary = sorted(set(text))
    return text, vocabulary


def prepare_dataset(do_print: bool = False):
    text, vocabulary = prepare_text_vocabulary()
    if do_print:
        print(text[:1000])
    vocab_size = len(vocabulary)
    if do_print:
        print(f'# of unique characters: {vocab_size}')

    char2index = {c: idx for idx, c in enumerate(vocabulary)}
    int_text = np.array([char2index[c] for c in text])
    index2char = np.array(vocabulary)
    if do_print:
        print('Character to Index: \n')
        for c, _ in zip(char2index, range(vocab_size)):
            print(f'  {repr(c):4s}: {char2index[c]:3d}')
        print('\nInput text to Integer: \n')
        print(f'{repr(text[:20])} mapped to {int_text[:20]}')

    char_dataset = tf.data.Dataset.from_tensor_slices(int_text)
    sequences = char_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

    if do_print:
        print('Character Stream:\n')
        for idx in char_dataset.take(10):
            print(index2char[idx.numpy()])
        print('Sequence:\n')
        for idx in sequences.take(10):
            print(repr(''.join(index2char[idx.numpy()])))

    def create_input_target_pair(chunk):
        return chunk[:-1], chunk[1:]

    dataset = sequences.map(create_input_target_pair)

    if do_print:
        for input_example, target_example in dataset.take(1):
            print('Input data: ', repr(''.join(index2char[input_example.numpy()])))
            print('Target data:', repr(''.join(index2char[target_example.numpy()])))

    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    return dataset, vocabulary


def build_model_lstm(vocab_size, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(RNN_UNITS,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def train_model(dataset, vocabulary, weights_file=None):
    vocab_size = len(vocabulary)
    lstm_model = build_model_lstm(
        vocab_size=vocab_size,
        batch_size=BATCH_SIZE)
    if weights_file:
        lstm_model.load_weights(weights_file)
    lstm_model.summary()

    example_prediction = None
    target_example_batch = None
    # check shape
    for input_example_batch, target_example_batch in dataset.take(1):
        example_prediction = lstm_model(input_example_batch)
        assert (example_prediction.shape == (BATCH_SIZE, SEQ_LENGTH, vocab_size))

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    example_loss = loss(target_example_batch, example_prediction)
    print('Prediction shape: ', example_prediction.shape)
    print('Loss:      ', example_loss.numpy().mean())

    lstm_model.compile(optimizer='adam', loss=loss)

    # train!
    EPOCHS = 5
    lstm_model.fit(dataset, epochs=EPOCHS)
    lstm_model.save_weights(WEIGHTS_FILE)
    print('Model saved.')


def predict(vocabulary, input_string):
    print(input_string, end='')

    char2index = {c: idx for idx, c in enumerate(vocabulary)}
    index2char = np.array(vocabulary)

    # Setup the Model and Load the Weights
    lstm_model = build_model_lstm(len(vocabulary), batch_size=1)
    lstm_model.load_weights(WEIGHTS_FILE)
    lstm_model.build(tf.TensorShape([1, None]))
    lstm_model.summary()

    def generate_text(model, start_string):
        num_generate = 1000
        input_eval = [char2index[c] for c in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        temperature = 0.5  # explain

        model.reset_states()
        for _ in range(num_generate):
            predictions = tf.squeeze(model(input_eval), 0) / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)
            c = index2char[predicted_id]
            print(c, end='')
            text_generated.append(c)
        return start_string + ''.join(text_generated)
    return generate_text(lstm_model, start_string=input_string)


def first_run():
    ds, v = prepare_dataset(True)
    train_model(ds, v)
    predict(v, TEST_STRING)


def run():
    ds, v = prepare_dataset()
    train_model(ds, v, WEIGHTS_FILE)
    predict(v, TEST_STRING)


def predict_input():
    _, v = prepare_text_vocabulary()
    input_text = input("Enter your starting string: ")
    predict(v, input_text)


# Use `first_run` at the very beginning to generate a weights file
# first_run()
# Subsequently use `run` to train incrementally and try to predict and see the result
# Re-run `run` to improve the weights.
run()
# Use `predict_input` when you want to enter your own prompt
# predict_input()
