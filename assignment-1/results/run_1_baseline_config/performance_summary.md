# Performance Summary of Baseline Config (Run 1)

## Summary
98.925
99.033
99.142
99.017
98.983
Accuracy: mean=99.020 std=0.071 n=5

## Model Definitions
```py
def define_model():
    model = Sequential()
    # First Conv Block
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    # Second Conv Block
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    # Third Conv Block (additional)
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    # Dense Layers
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```