

data = pd.read_excel('/home/usuario/dados/all_stocks_5yr.xlsx', index_col = 'date')

# Selecionando as colunas necessárias
data = data[['open', 'high', 'low', 'close']]

# Adicionando a coluna de previsão (close) e a deslocando
data['output'] = data.close.shift(-1)

# Removendo NaN na amostra final
data = data.dropna()

# Reescalando
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
rescaled = scaler.fit_transform(data.values)

# Divisão de teste e treino
training_ratio = 0.8
training_testing_index = int(len(rescaled) * training_ratio)
training_data = rescaled[:training_testing_index]
testing_data = rescaled[training_testing_index:]
training_length = len(training_data)
testing_length = len(testing_data)

# Divisão do treino em entrada e saída
training_input_data = training_data[:, 0:-1]
training_output_data = training_data[:, -1]

# Divisão do teste em entrada e saída
testing_input_data = testing_data[:, 0:-1]
testing_output_data = testing_data[:, -1]

# Reshape dos dados
training_input_data = training_input_data.reshape(training_input_data.shape[0], 1, training_input_data.shape[1])
testing_input_data = testing_input_data.reshape(testing_input_data.shape[0], 1, testing_input_data.shape[1])

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, SimpleRNN, GRU

# Construção do modelo LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(100, input_shape = (training_input_data.shape[1], training_input_data.shape[2])))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer = 'adam', loss='mse')

# Histórico, treinamento do modelo e tempo
start_time_lstm = time.time()
history_lstm = model_lstm.fit(
    training_input_data,
    training_output_data,
    epochs = 30,
    batch_size = 32,
    validation_data=(testing_input_data, testing_output_data),
    shuffle=False
)
end_time_lstm = time.time()
print("Tempo de treinamento do LSTM: ", end_time_lstm - start_time_lstm)

# Construção do modelo Rnn simples
model_rnn = Sequential()
model_rnn.add(SimpleRNN(100, input_shape = (training_input_data.shape[1], training_input_data.shape[2])))
model_rnn.add(Dense(1))
model_rnn.compile(optimizer = 'adam', loss='mse')

#Histórico do modelo e treinamento
start_time_rnn = time.time()
history_rnn = model_rnn.fit(
    training_input_data,
    training_output_data,
    epochs = 30,
    batch_size = 32,
    validation_data=(testing_input_data, testing_output_data),
    shuffle=False
)
end_time_rnn = time.time()
print("Tempo de treinamento do RNN: ", end_time_rnn - start_time_rnn)

# Construçaõ do modelo Gru
model_gru = Sequential()
model_gru.add(GRU(100, input_shape = (training_input_data.shape[1], training_input_data.shape[2])))
model_gru.add(Dense(1))
model_gru.compile(optimizer = 'adam', loss='mse')

#Histórico do modelo e treinamento
start_time_gru = time.time()
history_gru = model_gru.fit(
    training_input_data,
    training_output_data,
    epochs = 30,
    batch_size = 32,
    validation_data=(testing_input_data, testing_output_data),
    shuffle=False
)
end_time_gru = time.time()
print("Tempo de treinamento do GRU: ", end_time_gru - start_time_gru)

from matplotlib import pyplot
# Plot do histórico do Lstm
pyplot.plot(history_lstm.history['loss'], label='Loss do Treino - LSTM')
pyplot.plot(history_lstm.history['val_loss'], label='Loss do Teste - LSTM')
pyplot.legend()
pyplot.show()

# Plot do histórico do Rnn
pyplot.plot(history_rnn.history['loss'], label='Loss do Treino - RNN')
pyplot.plot(history_rnn.history['val_loss'], label='Loss do Teste - RNN')
pyplot.legend()
pyplot.show()

# Plot do histórico do Gry
pyplot.plot(history_gru.history['loss'], label='Loss do Treino - Gru')
pyplot.plot(history_gru.history['val_loss'], label='Loss do Teste - Gru')
pyplot.legend()
pyplot.show()

# Código para gerar as predições
raw_predictions_lstm = model_lstm.predict(testing_input_data)
raw_predictions_rnn = model_rnn.predict(testing_input_data)
raw_predictions_gru = model_gru.predict(testing_input_data)

# Remodelando os dados de entrada do teste de volta para 2d
testing_input_data = testing_input_data.reshape((testing_input_data.shape[0], testing_input_data.shape[2]))
testing_output_data = testing_output_data.reshape((len(testing_output_data), 1))

from numpy import concatenate
# Invertendo a escala para dados de previsão de cada modelo
unscaled_predictions_lstm = concatenate((testing_input_data, raw_predictions_lstm), axis = 1)
unscaled_predictions_lstm = scaler.inverse_transform(unscaled_predictions_lstm)
unscaled_predictions_lstm = unscaled_predictions_lstm[:, -1]

unscaled_predictions_rnn = concatenate((testing_input_data, raw_predictions_rnn), axis = 1)
unscaled_predictions_rnn = scaler.inverse_transform(unscaled_predictions_rnn)
unscaled_predictions_rnn = unscaled_predictions_rnn[:, -1]

unscaled_predictions_gru = concatenate((testing_input_data, raw_predictions_gru), axis = 1)
unscaled_predictions_gru = scaler.inverse_transform(unscaled_predictions_gru)
unscaled_predictions_gru = unscaled_predictions_gru[:, -1]

# Invertendo a escala para o dado atual
unscaled_actual_data = concatenate((testing_input_data, testing_output_data), axis = 1)
unscaled_actual_data = scaler.inverse_transform(unscaled_actual_data)
unscaled_actual_data = unscaled_actual_data[:, -1]

# Plot do LSTM
pyplot.plot(unscaled_actual_data, label='Preço de Fechamento Atual Ajustado')
pyplot.plot(unscaled_predictions_lstm, label='Preço de Fechamento do LSTM')
pyplot.legend()
pyplot.show()

# Plot do Rnn Simples
pyplot.plot(unscaled_actual_data, label='Preço de Fechamento Atual Ajustado')
pyplot.plot(unscaled_predictions_rnn, label='Preço de Fechamento do Rnn Simples')
pyplot.legend()
pyplot.show()

# Plot do Gru
pyplot.plot(unscaled_actual_data, label='Preço de Fechamento Atual Ajustado')
pyplot.plot(unscaled_predictions_gru, label='Preço de Fechamento do Gru')
pyplot.legend()
pyplot.show()

# Plot comparativo dos três modelos de predição
pyplot.plot(unscaled_actual_data, label='Preço de Fechamento Atual Ajustado')
pyplot.plot(unscaled_predictions_lstm, label='Preço de Fechamento do LSTM')
pyplot.plot(unscaled_predictions_rnn, label='Preço de Fechamento do Rnn Simples')
pyplot.plot(unscaled_predictions_gru, label='Preço de Fechamento do Gru')
pyplot.legend()
pyplot.show()