import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pandas.read_csv("iris.csv",delimiter=',',header=None)
train, test = train_test_split(data, test_size=0.2)

x_train=train.iloc[:,0:4].values.astype(float)
y_train=train.iloc[:,4]

x_test=test.iloc[:,0:4].values.astype(float)
y_test = test.iloc[:,4]


# One hot encoding of classes
encoder= LabelEncoder()
encoder.fit(y_train)
encoded_y = encoder.transform(y_train)
encoded_y = keras.utils.to_categorical(encoded_y)

encoder2= LabelEncoder()
encoder2.fit(y_test)
encoded_y_test = encoder2.transform(y_test)
encoded_y_test = keras.utils.to_categorical(encoded_y_test)



# Build structure of model
model = Sequential()
model.add(Dense(6,input_dim=4, activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(3,activation='sigmoid'))

# Compile model
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="./weights.hdf5",monitor='accuracy', verbose=1, save_best_only=True)

# Train model
history = model.fit(x_train,encoded_y,batch_size=5,epochs=300,verbose=2,callbacks=[checkpointer],validation_data=(x_test,encoded_y_test))

# Plot history keys
print("Metrics are: "+ str(history.history.keys()))

# Plot Accuracy over Time
acc = max(history.history['acc'])
index_acc= np.argmax(history.history['acc'])
print("Highest accuracy achieved in epoch "+str(index_acc+1)+" with accuracy of "+str(acc)+".")
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.show()

# Plot Loss over Time
loss = min(history.history['loss'])
index_loss= np.argmin(history.history['loss'])
print("Lowest loss achieved in epoch "+str(index_loss+1)+" with loss of "+str(loss)+".")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

# Test on new data
predictions=model.predict(x_test)
classes = np.zeros_like(predictions)
classes[np.arange(len(predictions)), predictions.argmax(1)] = 1

print("Now testing on unknown data...\n")
count=0
score=0
for row in classes:
    if np.array_equal(row,encoded_y_test[count]):
        score=score+1
    count = count+1
print("Accuracy on unknown data is: "+str((score/classes.__len__())*100)+"%")
