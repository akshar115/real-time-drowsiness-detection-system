import os
# Ensure Keras cache points to local directory inside workspace for sandbox compatibility
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.path.dirname(os.path.abspath(__file__))
if 'USERPROFILE' not in os.environ:
    os.environ['USERPROFILE'] = os.path.dirname(os.path.abspath(__file__))

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Small
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

def build_mobilenet_model(input_shape=(128, 128, 3)):
    """
    Builds a MobileNetV2 model for binary classification.
    Top 30 layers are unfrozen for fine-tuning.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False
        
    # Unfreeze top 30 layers
    for layer in base_model.layers[-30:]:
        layer.trainable = True
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def build_mobilenet_v3_model(input_shape=(128, 128, 3)):
    """
    Builds a MobileNetV3 (Next-Gen) model for binary classification.
    Features Squeeze-and-Excitation attention blocks and Hard-Swish activations.
    """
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False
        
    # Unfreeze top 30 layers for fine-tuning
    for layer in base_model.layers[-30:]:
        layer.trainable = True
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def build_custom_cnn(input_shape=(128, 128, 3)):
    """
    Builds an advanced custom CNN model from scratch for binary classification.
    Upgraded with GlobalAveragePooling to make it immune to data augmentation shifts!
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # CRITICAL FIX: Use GlobalAveragePooling2D instead of Flatten!
    # Because we shift the images by 20% during augmentation, Flatten() completely fails 
    # to learn position-invariant features. GlobalAveragePooling solves this.
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_binary_classifier(data_dir, classes, model_save_path, model_type='mobilenet', target_size=(128, 128), batch_size=32, epochs=20):
    print(f"--- Training model for classes: {classes} ---")
    
    # Preprocessing setup:
    # MobileNetV3 has internal Rescaling (expects [0, 255]), so mobilenet_v3.preprocess_input is an identity function.
    # MobileNetV2 and Custom CNN expect pixels scaled in [-1, 1].
    if model_type == 'mobilenet_v3':
        prep_func = tf.keras.applications.mobilenet_v3.preprocess_input
    else:
        prep_func = tf.keras.applications.mobilenet_v2.preprocess_input
        
    datagen = ImageDataGenerator(
        preprocessing_function=prep_func,
        validation_split=0.2,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=classes, # Explicit class mapping
        subset='training'
    )
    
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=classes,
        subset='validation'
    )
    
    # Handle class imbalance (especially for yawning)
    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights_arr))
    print(f"Class weights: {class_weights}")

    if model_type == 'custom_cnn':
        print("Using Custom CNN architecture trained from scratch.")
        model = build_custom_cnn(input_shape=(target_size[0], target_size[1], 3))
    elif model_type == 'mobilenet_v3':
        print("Using MobileNetV3 (Next-Gen) architecture with fine-tuning.")
        model = build_mobilenet_v3_model(input_shape=(target_size[0], target_size[1], 3))
    else:
        print("Using MobileNetV2 architecture with fine-tuning.")
        model = build_mobilenet_model(input_shape=(target_size[0], target_size[1], 3))
    
    checkpoint = ModelCheckpoint(
        model_save_path, 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eye_dir', type=str, required=True, help="Path to Eye dataset (containing 'Closed' and 'Open' folders)")
    parser.add_argument('--mouth_dir', type=str, required=True, help="Path to Mouth dataset (containing 'no_yawn' and 'Yawn' folders)")
    parser.add_argument('--model_type', type=str, default='mobilenet', choices=['mobilenet', 'custom_cnn', 'mobilenet_v3'], help="Architecture to train: 'mobilenet', 'custom_cnn', or 'mobilenet_v3'")
    args = parser.parse_args()
    
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Train Eye Model
    # Classes: Closed=0, Open=1
    train_binary_classifier(
        data_dir=args.eye_dir,
        classes=['Closed', 'Open'],
        model_save_path=os.path.join(model_dir, f'eye_model_{args.model_type}.h5'),
        model_type=args.model_type,
        epochs=40
    )
    
    # Train Mouth Model
    # Classes: no_yawn=0, Yawn=1
    train_binary_classifier(
        data_dir=args.mouth_dir,
        classes=['no_yawn', 'Yawn'],
        model_save_path=os.path.join(model_dir, f'mouth_model_{args.model_type}.h5'),
        model_type=args.model_type,
        epochs=40
    )
    
    print(f"Training complete. Models saved to '{model_dir}' directory.")
