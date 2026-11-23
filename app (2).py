# app.py
# Streamlit CNN classifier for individually uploaded Benign and Malignant images.
# Educational/research use only; not for clinical diagnosis.

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ImageFile.LOAD_TRUNCATED_IMAGES = True

st.set_page_config(page_title="Benign vs Malignant CNN", page_icon="🧠", layout="wide")
st.title("🧠 CNN Image Classification")
st.write(
    "Upload individual image files into the correct class. The app performs EDA, "
    "splits the dataset, trains a CNN, and reports test-set results."
)
st.warning(
    "Research/educational demonstration only. This model is not a medical device "
    "and must not be used to diagnose or guide treatment."
)

IMAGE_SIZE = (64, 64)
CLASSES = ["Benign", "Malignant"]
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]

@st.cache_resource(show_spinner=False)
def train_cnn(X_train, y_train, X_test, y_test):
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
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=32,
        callbacks=[callback],
        verbose=0,
    )
    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(
        y_test, preds, labels=[0, 1], target_names=CLASSES, zero_division=0
    )
    return model, history.history, preds, accuracy, report

def decode_uploads(uploaded_files, class_id):
    images, labels, previews = [], [], []
    for uploaded in uploaded_files or []:
        try:
            raw = uploaded.getvalue()
            with Image.open(io.BytesIO(raw)) as image:
                image = image.convert("RGB").resize(IMAGE_SIZE)
                arr = np.asarray(image, dtype=np.float32) / 255.0
            images.append(arr)
            labels.append(class_id)
            previews.append((arr, uploaded.name))
        except Exception:
            st.warning(f"Could not read image: {uploaded.name}")
    return images, labels, previews

st.subheader("1. Upload individual images")
st.caption("Select multiple image files for each class. Do not upload a ZIP file.")
col_b, col_m = st.columns(2)
with col_b:
    benign_files = st.file_uploader(
        "🟢 Upload BENIGN images",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        key="benign_images",
    )
with col_m:
    malignant_files = st.file_uploader(
        "🔴 Upload MALIGNANT images",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        key="malignant_images",
    )

benign_X, benign_y, benign_preview = decode_uploads(benign_files, 0)
malignant_X, malignant_y, malignant_preview = decode_uploads(malignant_files, 1)

X_list = benign_X + malignant_X
y_list = benign_y + malignant_y

if not X_list:
    st.info("Upload images using the two class-specific upload boxes above.")
    st.markdown("**Class labels:** Benign = 0 · Malignant = 1")
    st.stop()

X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.int64)
counts = np.bincount(y, minlength=2)

st.success(f"Loaded {len(X)} images: {counts[0]} Benign and {counts[1]} Malignant.")

st.subheader("2. Exploratory Data Analysis (EDA)")
m1, m2, m3 = st.columns(3)
m1.metric("Total images", len(X))
m2.metric("Benign images", int(counts[0]))
m3.metric("Malignant images", int(counts[1]))

distribution = pd.DataFrame({"Class": CLASSES, "Image count": counts})
st.dataframe(distribution, use_container_width=True)

chart1, chart2 = st.columns(2)
with chart1:
    fig, ax = plt.subplots()
    ax.pie(counts, labels=CLASSES, autopct="%1.1f%%", startangle=90)
    ax.set_title("Class Distribution")
    st.pyplot(fig)
    plt.close(fig)
with chart2:
    fig, ax = plt.subplots()
    ax.bar(CLASSES, counts)
    ax.set_title("Images per Class")
    ax.set_ylabel("Number of images")
    st.pyplot(fig)
    plt.close(fig)

st.write("Uploaded image previews")
preview_cols = st.columns(4)
all_previews = benign_preview[:2] + malignant_preview[:2]
for idx, (image, filename) in enumerate(all_previews):
    with preview_cols[idx]:
        st.image(image, caption=filename, use_container_width=True)

st.subheader("3. Train/Test Split")
test_percent = st.slider("Testing data percentage", 10, 40, 20, 5)
if min(counts) < 2:
    st.error("Please upload at least 2 valid images in BOTH classes before splitting.")
    st.stop()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_percent / 100,
        random_state=42,
        stratify=y,
    )
except ValueError as err:
    st.error(f"Unable to split this dataset: {err}. Upload more images in each class.")
    st.stop()

s1, s2 = st.columns(2)
s1.metric("Training images", len(X_train))
s2.metric("Testing images", len(X_test))

st.subheader("4. CNN Model Training")
if st.button("Train CNN Model", type="primary"):
    with st.spinner("Training CNN model..."):
        model, history, predictions, accuracy, report = train_cnn(
            X_train, y_train, X_test, y_test
        )
        model.save("benign_malignant_cnn.keras")
        st.session_state["bm_results"] = {
            "accuracy": accuracy,
            "report": report,
            "history": history,
            "predictions": predictions,
            "y_test": y_test,
        }
    st.success("Training completed. Model saved as benign_malignant_cnn.keras.")

if "bm_results" in st.session_state:
    result = st.session_state["bm_results"]
    st.subheader("5. Test Accuracy")
    st.metric("CNN Test Accuracy", f"{result['accuracy'] * 100:.2f}%")

    history = result["history"]
    fig, ax = plt.subplots()
    ax.plot(history.get("accuracy", []), label="Training accuracy")
    ax.plot(history.get("val_accuracy", []), label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("6. Classification Report")
    st.code(result["report"])

    st.subheader("7. Confusion Matrix")
    cm = confusion_matrix(result["y_test"], result["predictions"], labels=[0, 1])
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Confusion Matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
