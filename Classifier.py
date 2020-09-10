from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
from Classification_Report import plot_classification_report; from Confusion_Matrix import plot_confusion_matrix
import pandas as pd, keras, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

################################################

data = pd.read_csv("EEG_data.csv")

data = data[['subject ID', 'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2', 'user-definedlabeln']]
data.loc[data['subject ID'] == 8].to_csv(path_or_buf="Validation/Subject_9.csv", index=False)
data.loc[data['subject ID'] == 9].to_csv(path_or_buf="Validation/Subject_10.csv", index=False)
data_maj = data.loc[data['subject ID'] < 8]

X = data_maj[['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']].values
X = X.reshape(X.shape + (1,))

y = data_maj[['user-definedlabeln']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

################################################

model = keras.models.Sequential()
model.add(keras.layers.Conv1D(100, 7, activation='relu', input_shape=(8, 1)))
model.add(keras.layers.Conv1D(150, 1, activation='relu'))
model.add(keras.layers.Conv1D(200, 1, activation='relu'))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(units = 64, activation = 'relu', kernel_initializer='normal'))
model.add(keras.layers.Dense(units = 16, activation = 'relu', kernel_initializer='normal'))
model.add(keras.layers.Dense(units = 4, activation = 'relu', kernel_initializer='normal'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(lr=0.00001), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, batch_size=30, epochs=250, verbose=1, validation_data=(X_test, y_test), 
                    callbacks=[keras.callbacks.ModelCheckpoint('Models/Generalized_Model.hdf5', monitor="val_acc", save_best_only=True, save_weights_only=False)])

################################################

generalized_model = keras.models.load_model('Models/Generalized_Model.hdf5')

y_pred = []; y_pred_raw = generalized_model.predict(X_test)
for j in y_pred_raw: y_pred.append(round(j[0]))

f = open("Development_Outputs/Generalized_Metrics.txt", "w+")
f.write("Generalized Model Metrics -->> "); 
f.write("\n\nAccuracy Score: "); f.write(str(accuracy_score(y_test, y_pred)))
f.write("\n\nArea Under the Receiver Operating Characteristics (AUROC): "); f.write(str(roc_auc_score(y_test, y_pred)))
f.write("\n\nCohen\'s Kappa Co-efficient (K): "); f.write(str(cohen_kappa_score(y_test, y_pred)))
f.close(); print(open("Development_Outputs/Generalized_Metrics.txt", "r").read())

cr = classification_report(y_test, y_pred, target_names=['Confused', 'Not-Confused']); cr = cr.split("\n")
plot_classification_report(cr[0] + '\n\n' + cr[2] + '\n' + cr[3] + '\n\n' + cr[5] + '\n',
                           title = 'Classification Report')
plt.savefig('Development_Outputs/Generalized_Classification_Report.png', dpi=200, format='png', bbox_inches='tight'); plt.close();

cnf_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cnf_matrix, classes=['Confused', 'Not-Confused'], title='Confusion Matrix')
plt.savefig('Development_Outputs/Generalized_Confusion_Matrix.png', dpi=200, format='png', bbox_inches='tight'); plt.close();

################################################

X_sub = data_maj[['subject ID', 'Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2', 'user-definedlabeln']]

for i in range(len(set(list(X_sub['subject ID'])))): 
    
    vars()['sub_' + str(i)] = X_sub.loc[X_sub['subject ID'] == i]
    
    vars()['sub_' + str(i) + '_X'] = vars()['sub_' + str(i)][['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']].values
    vars()['sub_' + str(i) + '_y'] = vars()['sub_' + str(i)][['user-definedlabeln']]
    vars()['sub_' + str(i) + '_X'] = vars()['sub_' + str(i) + '_X'].reshape(vars()['sub_' + str(i) + '_X'].shape + (1,))

    (vars()['X_train_' + str(i)], 
     vars()['X_test_' + str(i)], 
     vars()['y_train_' + str(i)], 
     vars()['y_test_' + str(i)]) = train_test_split(vars()['sub_' + str(i) + '_X'], 
                                                    vars()['sub_' + str(i) + '_y'], 
                                                    test_size=0.25, random_state=42)
    
    for layer in generalized_model.layers: layer.trainable = False
    
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00001), loss='binary_crossentropy', metrics=['acc'])
    
    history = model.fit(vars()['X_train_' + str(i)], vars()['y_train_' + str(i)], 
                        batch_size=30, epochs=250, verbose=1, 
                        validation_data=(vars()['X_test_' + str(i)], vars()['y_test_' + str(i)]), 
                        callbacks=[keras.callbacks.ModelCheckpoint('Models/Subject_' + str(i+1) + '_Model.hdf5', 
                                   monitor="val_acc", save_best_only=True, save_weights_only=False)])
    
    model = keras.models.load_model('Models/Subject_' + str(i+1) + '_Model.hdf5')
    
    vars()['y_pred_' + str(i)] = []; vars()['y_pred_raw_' + str(i)] = model.predict(vars()['X_test_' + str(i)])
    for j in vars()['y_pred_raw_' + str(i)]: vars()['y_pred_' + str(i)].append(round(j[0]))
    
    f = open("Development_Outputs/Subject_" + str(i+1) + "_Metrics.txt", "w+")
    f.write("Subject_" + str(i+1) + " Model Metrics -->> "); 
    f.write("\n\nAccuracy Score: "); f.write(str(accuracy_score(vars()['y_test_' + str(i)], vars()['y_pred_' + str(i)])))
    f.write("\n\nArea Under the Receiver Operating Characteristics (AUROC): "); f.write(str(roc_auc_score(vars()['y_test_' + str(i)], vars()['y_pred_' + str(i)])))
    f.write("\n\nCohen\'s Kappa Co-efficient (K): "); f.write(str(cohen_kappa_score(vars()['y_test_' + str(i)], vars()['y_pred_' + str(i)])))
    f.close(); print(open("Development_Outputs/Subject_" + str(i+1) + "_Metrics.txt", "r").read())
    
    cr = classification_report(vars()['y_test_' + str(i)], vars()['y_pred_' + str(i)], target_names=['Confused', 'Not-Confused']); cr = cr.split("\n")
    plot_classification_report(cr[0] + '\n\n' + cr[2] + '\n' + cr[3] + '\n\n' + cr[5] + '\n',
                               title = 'Classification Report')
    plt.savefig('Development_Outputs/Subject_' + str(i+1) + '_Classification_Report.png', dpi=200, format='png', bbox_inches='tight'); plt.close();
    
    cnf_matrix = confusion_matrix(vars()['y_test_' + str(i)], vars()['y_pred_' + str(i)])
    plot_confusion_matrix(cnf_matrix, classes=['Confused', 'Not-Confused'], title='Confusion Matrix')
    plt.savefig('Development_Outputs/Subject_' + str(i+1) + '_Confusion_Matrix.png', dpi=200, format='png', bbox_inches='tight'); plt.close();
    
################################################