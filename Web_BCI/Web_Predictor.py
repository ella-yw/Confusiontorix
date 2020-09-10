def pred(link, keras, pd, tts):
    
    keras.backend.clear_session()
    
    data = pd.read_csv("file_to_predict/" + link)
    
    X = data[['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']].values
    X = X.reshape(X.shape + (1,))
    
    y = data[['user-definedlabeln']].values
    
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.25, random_state=42)
    
    generalized_model = keras.models.load_model('../Models/Generalized_Model.hdf5')
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(100, 7, activation='relu', input_shape=(8, 1)))
    model.add(keras.layers.Conv1D(150, 1, activation='relu'))
    model.add(keras.layers.Conv1D(200, 1, activation='relu'))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(units = 64, activation = 'relu', kernel_initializer='normal'))
    model.add(keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer='normal'))
    model.add(keras.layers.Dense(units = 4, activation = 'relu', kernel_initializer='normal'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    for layer in generalized_model.layers: layer.trainable = False
    
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00001), loss='binary_crossentropy', metrics=['acc'])
    
    model.fit(X_train, y_train, batch_size=30, epochs=100, verbose=1, validation_data=(X_test, y_test), 
              callbacks=[keras.callbacks.ModelCheckpoint('home/static/home/Model.hdf5', monitor="val_acc", save_best_only=True, save_weights_only=False)])
    
    model = keras.models.load_model('home/static/home/Model.hdf5')
    
    y_pred = []; y_pred_raw = model.predict(X_test)
    for j in y_pred_raw: y_pred.append(round(j[0]))
    
    zero_count = y_pred.count(0); one_count = y_pred.count(1);
    
    if zero_count > one_count: output = "No Confusion"; percentage = str((zero_count / (zero_count + one_count)) * 100) + "%";
    else: output = "Confusion"; percentage = str((one_count / (zero_count + one_count)) * 100) + "%";
    
    return (output, percentage)
    