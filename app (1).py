# app.py
# Streamlit CNN image-classification app using an uploaded ZIP dataset.
# Expected ZIP structure:
# dataset.zip/
#   class_1/
#     image1.jpg
#   class_2/
#     image2.jpg

import os
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ImageFile.LOAD_TRUNCATED_IMAGES = True

st.set_page_config(page_title="CNN Image Classification", page_icon="🧠", layout="wide")
st.title("🧠 CNN Image Classification")
st.write("Upload your image dataset as a ZIP file. The app will perform EDA, split the data, train a CNN, and show evaluation results.")

IMAGE_SIZE = (64, 64)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@st.cache_resource(show_spinner=False)
def build_and_train_model(X_train, y_train, X_test, y_test, num_classes, class_names):
    # Import TensorFlow only after the user uploads a valid dataset.
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=12,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )
    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, labels=list(range(num_classes)),
        target_names=class_names, zero_division=0
    )
    return model, history.history, predictions, accuracy, report

def find_class_folder(extract_dir):
    """Find a folder whose immediate subfolders contain image files."""
    candidates = []
    for folder in [Path(extract_dir)] + [p for p in Path(extract_dir).rglob("*") if p.is_dir()]:
        class_dirs = []
        for child in folder.iterdir():
            if child.is_dir():
                count = sum(1 for f in child.rglob("*")
                            if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS)
                if count:
                    class_dirs.append((child, count))
        if len(class_dirs) >= 2:
            candidates.append((sum(c for _, c in class_dirs), folder, class_dirs))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, root, class_dirs = candidates[0]
    return root, sorted(class_dirs, key=lambda item: item[0].name.lower())

def load_dataset(class_dirs):
    images, labels = [], []
    class_names = [folder.name for folder, _ in class_dirs]
    for label, (folder, _) in enumerate(class_dirs):
        files = [p for p in folder.rglob("*")
                 if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]
        for file_path in files:
            try:
                with Image.open(file_path) as img:
                    img = img.convert("RGB").resize(IMAGE_SIZE)
                    images.append(np.asarray(img, dtype=np.float32) / 255.0)
                    labels.append(label)
            except Exception:
                continue
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int64), class_names

uploaded_zip = st.file_uploader(
    "📁 Upload your image dataset (ZIP file)",
    type=["zip"],
    help="The ZIP must contain one folder per class, with images inside each folder."
)

if uploaded_zip is None:
    st.info("Upload your dataset ZIP file using the upload bar above to begin.")
    st.markdown("**Required ZIP structure:** `dataset/class_name/image.jpg`")
    st.stop()

try:
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = Path(temp_dir) / "uploaded_dataset.zip"
        zip_path.write_bytes(uploaded_zip.getvalue())

        with zipfile.ZipFile(zip_path, "r") as archive:
            # Reject unsafe paths before extraction.
            for member in archive.infolist():
                target = (Path(temp_dir) / "extracted" / member.filename).resolve()
                if not str(target).startswith(str((Path(temp_dir) / "extracted").resolve())):
                    raise ValueError("The ZIP contains an unsafe file path.")
            archive.extractall(Path(temp_dir) / "extracted")

        dataset_root, class_dirs = find_class_folder(Path(temp_dir) / "extracted")
        if not class_dirs:
            st.error("Could not find class folders. Your ZIP must contain at least two class folders, each with images.")
            st.stop()

        X, y, class_names = load_dataset(class_dirs)

        if len(X) == 0 or len(class_names) < 2:
            st.error("No usable images were found. Please check your ZIP file.")
            st.stop()

        counts = np.bincount(y, minlength=len(class_names))
        if np.any(counts < 2):
            st.error("Each class must contain at least 2 valid images for a train/test split.")
            st.stop()

        st.success(f"Dataset uploaded successfully: {len(X)} images across {len(class_names)} classes.")

        st.subheader("1. Exploratory Data Analysis (EDA)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total images", len(X))
        c2.metric("Number of classes", len(class_names))
        c3.metric("Image dimensions", f"{IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}")

        distribution = pd.DataFrame({
            "Class": class_names,
            "Images": counts
        })
        st.dataframe(distribution, use_container_width=True)

        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots()
            ax.pie(counts, labels=class_names, autopct="%1.1f%%", startangle=90)
            ax.set_title("Class Distribution")
            st.pyplot(fig)
            plt.close(fig)

        with right:
            fig, ax = plt.subplots()
            ax.bar(class_names, counts)
            ax.set_title("Images per Class")
            ax.set_xlabel("Class")
            ax.set_ylabel("Number of images")
            ax.tick_params(axis="x", rotation=35)
            st.pyplot(fig)
            plt.close(fig)

        st.write("Sample images")
        sample_cols = st.columns(min(len(class_names), 4))
        for i, class_name in enumerate(class_names):
            sample_index = int(np.where(y == i)[0][0])
            with sample_cols[i % len(sample_cols)]:
                st.image(X[sample_index], caption=class_name, use_container_width=True)

        st.subheader("2. Train/Test Split")
        test_size = st.slider("Test data percentage", min_value=10, max_value=40, value=20, step=5)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=42, stratify=y
        )
        a, b = st.columns(2)
        a.metric("Training images", len(X_train))
        b.metric("Testing images", len(X_test))

        st.subheader("3. CNN Model Training")
        if st.button("Train CNN Model", type="primary"):
            with st.spinner("Training CNN model. This may take a few minutes..."):
                model, history, predictions, accuracy, report = build_and_train_model(
                    X_train, y_train, X_test, y_test, len(class_names), class_names
                )
                model.save("cnn_image_classifier.keras")
                st.session_state["cnn_results"] = {
                    "accuracy": accuracy,
                    "report": report,
                    "history": history,
                    "predictions": predictions,
                    "y_test": y_test,
                    "class_names": class_names,
                }
                st.session_state["cnn_model_ready"] = True
            st.success("CNN training completed. Model saved as cnn_image_classifier.keras.")

        if st.session_state.get("cnn_model_ready"):
            result = st.session_state["cnn_results"]
            st.subheader("4. Model Accuracy")
            st.metric("Test Accuracy", f"{result['accuracy'] * 100:.2f}%")

            hist = result["history"]
            fig, ax = plt.subplots()
            ax.plot(hist.get("accuracy", []), label="Training accuracy")
            ax.plot(hist.get("val_accuracy", []), label="Validation accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("CNN Training History")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("5. Classification Report")
            st.code(result["report"])

            st.subheader("6. Confusion Matrix")
            cm = confusion_matrix(result["y_test"], result["predictions"],
                                  labels=list(range(len(class_names))))
            fig, ax = plt.subplots()
            ax.imshow(cm)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_xticks(range(len(class_names)), class_names, rotation=45)
            ax.set_yticks(range(len(class_names)), class_names)
            for row in range(cm.shape[0]):
                for col in range(cm.shape[1]):
                    ax.text(col, row, cm[row, col], ha="center", va="center")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

except zipfile.BadZipFile:
    st.error("The uploaded file is not a valid ZIP archive.")
except Exception as exc:
    st.exception(exc)
