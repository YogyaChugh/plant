import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the paths to your train and test directories
train_dir = 'C:/Users/mudit/PycharmProjects/Agro_lens/data/train'
test_dir = 'C:/Users/mudit/PycharmProjects/Agro_lens/data/test'

# ImageDataGenerators for training and testing data
train_datagen = ImageDataGenerator(rescale=1./255)  # Rescale images between 0 and 1
test_datagen = ImageDataGenerator(rescale=1./255)   # Rescale images between 0 and 1

# Load training data from the directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),  # Resize images to 256x256
    batch_size=32,           # Number of images per batch
    class_mode='categorical' # Categorical classification (35 classes)
)

# Load testing data from the directory
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),  # Resize images to 256x256
    batch_size=32,
    class_mode='categorical' # Categorical classification (35 classes)
)

# Define a simple CNN model for multiclass classification (35 categories)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer with number of classes (35 categories)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # You can adjust this based on your need
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)
# Print model summary to see details of each layer
model.summary()

# Save the trained model
model.save('C:/Users/mudit/PycharmProjects/Agro_lens/model/model_multiclass.h5')
