# Performance Summary of Shallow Config (Run 2)

## Summary
98.658
98.833
98.750
99.075
99.100
Accuracy: mean=98.883 std=0.176 n=5

## Model Definitions
```py
def define_model():
    model = Sequential()
    # First Conv Block
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    # Second Conv Block
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    # Dense Layers
    model.add(Dense(60, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    
    opt = SGD(learning_rate=0.01, momentum=0.9)  # Updated for modern Keras
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```