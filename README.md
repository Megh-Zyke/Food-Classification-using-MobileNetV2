
# Food Classification using MobileNetV2

This project focuses on classifying food images into 11 different categories using the MobileNetV2 architecture. The model is trained on the Food-11 dataset, and data augmentation techniques are applied for improved generalization.

## Project Structure

- **/kaggle/input**: Directory containing the dataset and MobileNetV2 weights.
  - **/food11**: Subdirectory containing the Food-11 dataset.
    - **/train**: Training images.
    - **/test**: Testing images.
  - **/mobilenetv2**: Subdirectory containing MobileNetV2 weights (`mobilenet_v2_weights.h5`).

- **main.ipynb**: Jupyter notebook containing the code for data preprocessing, model building, and training.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib

## Usage

1. **Install Dependencies:**
   - Ensure the required packages are installed using the following command:
     ```bash
     pip install numpy pandas tensorflow matplotlib
     ```

2. **Run Jupyter Notebook:**
   - Open the Jupyter notebook `main.ipynb`.
   - Execute each cell sequentially. The notebook includes the following sections:
     - Importing necessary libraries.
     - Data preprocessing using ImageDataGenerator.
     - Building the MobileNetV2-based classification model.
     - Training the model on the Food-11 dataset.

3. **Adjust Configurations:**
   - Customize hyperparameters and configurations in the notebook, such as the target image size, batch size, and number of epochs.

4. **Training Overview:**
   - After training, the notebook displays the model's performance metrics, such as accuracy and loss.

## Data Augmentation for Training Data

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # Other augmentation parameters can be uncommented and adjusted as needed.
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',    
    shuffle=True,
    seed=42,
    class_mode='categorical'
)
```

- `ImageDataGenerator` is configured for training data with specified augmentation techniques (commented out in this example).
- `flow_from_directory` generates batches of augmented data from the specified directory.

## Sample Images Display

```python
images, labels = next(train_generator)

plt.figure(figsize=(10, 10))

for i in range(6):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(food_items[np.argmax(labels[i])])
```

- A batch of augmented images and their corresponding labels are displayed in a grid.

## MobileNetV2 Model Setup

```python
num_classes = 11
local_weights_path = "/kaggle/input/mobilenetv2/mobilenet_v2_weights.h5"

model = MobileNetV2(input_shape=(224, 224, 3), weights=None, include_top=False)
model.load_weights(local_weights_path)

x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x) 
x = Dense(64, activation='relu')(x) 
out = Dense(num_classes, activation='softmax')(x) 

mobileNet_model = Model(inputs=model.input, outputs=out)
model.trainable = False
mobileNet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

- MobileNetV2 is imported and initialized with pre-trained weights.
- A custom classification head is added on top of MobileNetV2.
- The model is compiled with categorical crossentropy loss and the Adam optimizer.

## Model Training

```python
num_epochs = 30
mobileNet_model.fit(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, epochs=num_epochs)
```

- The model is trained using the `fit` method on the training generator.

## Model Training Metrics

After training for 30 epochs, the model achieves the following performance:

- **Loss:** 0.0112
- **Accuracy:** 99.60%

These metrics indicate the final training performance of the MobileNetV2 model on the Food-11 dataset.

...
