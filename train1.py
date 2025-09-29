import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ======================
# CẤU HÌNH
# ======================
IMG_SIZE = 128       # kích thước ảnh đầu vào
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = "data"    # thư mục chứa train/val
OUTDIR = "outputs_cnn1"  # nơi lưu model và labels

# ======================
# LOAD DATASET
# ======================
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Chuẩn hóa dữ liệu
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ======================
# XÂY CNN MODEL
# ======================
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# ======================
# TRAINING
# ======================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ======================
# LƯU KẾT QUẢ
# ======================
os.makedirs(OUTDIR, exist_ok=True)

# lưu model h5
model.save(os.path.join(OUTDIR, "best_model.h5"))
print(f"✅ Đã lưu model: {os.path.join(OUTDIR, 'best_model.h5')}")

# lưu labels.txt
with open(os.path.join(OUTDIR, "labels.txt"), "w", encoding="utf-8") as f:
    for label in class_names:
        f.write(label + "\n")
print(f"✅ Đã lưu nhãn: {os.path.join(OUTDIR, 'labels.txt')}")
