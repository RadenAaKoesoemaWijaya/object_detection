import streamlit as st
import cv2
import numpy as np
import os
import time
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf  # Tambahkan import ini
from utils.detection import load_yolo_model, detect_objects
from utils.classification import load_classification_model, classify_objects
from utils.visualization import visualize_results, plot_statistics
from utils.training import train_classification_model, prepare_yolo_dataset, train_yolo_model, plot_training_history
import shutil

# Set page config
st.set_page_config(page_title="Deteksi Objek Preparat Mikroskopis (DETOKS)", layout="wide")

# Title and description
st.title("Aplikasi Deteksi Objek Preparat Mikroskopis (DETOKS)")
st.markdown("""
Aplikasi ini membantu menganalisis gambar preparat mikroskopis untuk mendeteksi dan menghitung:
- Jenis bakteri
- Jamur
- Parasit
- Sel-sel darah
""")

# Sidebar for navigation
st.sidebar.title("Fitur Pilihan")
page = st.sidebar.radio("Pilih Halaman", ["Deteksi Objek", "Analisis Statistik", "Pelatihan Model", "Tentang Aplikasi"])

# Load models
@st.cache_resource
def load_models():
    yolo_model = load_yolo_model()
    classification_model = load_classification_model()
    return yolo_model, classification_model

# Try to load models
try:
    with st.spinner("Memuat model..."):
        yolo_model, classification_model = load_models()
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.info("Pastikan model telah diunduh atau dilatih terlebih dahulu.")
    yolo_model, classification_model = None, None

# Deteksi Objek page
if page == "Deteksi Objek":
    st.header("Deteksi Objek pada Preparat Mikroskopis")
    
    # Upload image
    uploaded_file = st.file_uploader("Unggah gambar preparat mikroskopis", type=["jpg", "jpeg", "png"])
    
    # Image processing options
    st.subheader("Pengaturan Deteksi")
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider("Ambang Batas Kepercayaan", 0.0, 1.0, 0.25, 0.05)
    with col2:
        overlap_threshold = st.slider("Ambang Batas Tumpang Tindih (IoU)", 0.0, 1.0, 0.45, 0.05)
    
    # Process image button
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Asli", use_column_width=True)
        
        # Process button
        if st.button("Proses Gambar"):
            if yolo_model is not None and classification_model is not None:
                with st.spinner("Memproses gambar..."):
                    # Convert PIL Image to numpy array
                    img_array = np.array(image)
                    
                    # Detect objects
                    start_time = time.time()
                    detections = detect_objects(yolo_model, img_array, confidence_threshold, overlap_threshold)
                    
                    # Classify detected objects
                    if len(detections) > 0:
                        classifications = classify_objects(classification_model, img_array, detections)
                        
                        # Visualize results
                        result_img, counts = visualize_results(img_array, detections, classifications)
                        
                        # Display results
                        st.subheader("Hasil Deteksi")
                        st.image(result_img, caption="Hasil Deteksi", use_column_width=True)
                        
                        # Display counts
                        st.subheader("Jumlah Objek Terdeteksi")
                        count_df = pd.DataFrame(list(counts.items()), columns=["Jenis", "Jumlah"])
                        st.dataframe(count_df)
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.barplot(x="Jenis", y="Jumlah", data=count_df, ax=ax)
                        ax.set_title("Distribusi Objek Terdeteksi")
                        st.pyplot(fig)
                        
                        # Processing time
                        processing_time = time.time() - start_time
                        st.info(f"Waktu pemrosesan: {processing_time:.2f} detik")
                        
                        # Save results option
                        if st.button("Simpan Hasil"):
                            # Create results directory if it doesn't exist
                            os.makedirs("results", exist_ok=True)
                            
                            # Save image
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            result_img_pil = Image.fromarray(result_img)
                            result_img_pil.save(f"results/result_{timestamp}.png")
                            
                            # Save counts to CSV
                            count_df.to_csv(f"results/counts_{timestamp}.csv", index=False)
                            
                            st.success(f"Hasil berhasil disimpan di folder 'results'")
                    else:
                        st.warning("Tidak ada objek yang terdeteksi. Coba sesuaikan ambang batas kepercayaan.")
            else:
                st.error("Model belum dimuat. Silakan periksa kesalahan di atas.")

# Analisis Statistik page
elif page == "Analisis Statistik":
    st.header("Analisis Statistik")
    
    # Check if results directory exists and has files
    if os.path.exists("results") and any(f.endswith(".csv") for f in os.listdir("results")):
        # List all CSV files in results directory
        csv_files = [f for f in os.listdir("results") if f.endswith(".csv")]
        
        # Let user select a file
        selected_file = st.selectbox("Pilih file hasil untuk dianalisis", csv_files)
        
        if selected_file:
            # Load the CSV file
            df = pd.read_csv(os.path.join("results", selected_file))
            
            # Display the data
            st.subheader("Data")
            st.dataframe(df)
            
            # Plot statistics
            st.subheader("Visualisasi")
            fig = plot_statistics(df)
            st.pyplot(fig)
    else:
        st.info("Belum ada hasil yang disimpan. Silakan proses beberapa gambar terlebih dahulu.")

# Pelatihan Model page
elif page == "Pelatihan Model":
    st.header("Pelatihan Model")
    
    # Create tabs for different training options
    train_tab, model_tab = st.tabs(["Pelatihan Model", "Model yang Tersedia"])
    
    with train_tab:
        st.subheader("Pelatihan Model Baru")
        
        # Select model type to train
        model_type = st.radio("Pilih Jenis Model untuk Dilatih", ["Model Deteksi Objek (YOLO)", "Model Klasifikasi"])
        
        if model_type == "Model Klasifikasi":
            st.write("Unggah dataset klasifikasi (folder dengan subfolder untuk setiap kelas)")
            
            # Upload dataset
            uploaded_dataset = st.file_uploader("Unggah dataset klasifikasi (ZIP)", type=["zip"])
            
            # Model name input
            model_name = st.text_input("Nama File Model", "classification_model.h5")
            if not model_name.endswith(".h5"):
                model_name += ".h5"
            
            # Training parameters
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.slider("Jumlah Epoch", 5, 100, 20)
                batch_size = st.slider("Batch Size", 8, 64, 32)
            with col2:
                img_size = st.slider("Ukuran Gambar", 32, 224, 64)
                validation_split = st.slider("Rasio Validasi", 0.1, 0.5, 0.2, 0.05)
            
            # Train button
            if uploaded_dataset is not None and st.button("Latih Model Klasifikasi"):
                with st.spinner("Memproses dataset..."):
                    # Save uploaded zip file
                    dataset_zip_path = "temp_dataset.zip"
                    with open(dataset_zip_path, "wb") as f:
                        f.write(uploaded_dataset.getbuffer())
                    
                    # Extract dataset
                    import zipfile
                    dataset_path = "temp_dataset"
                    os.makedirs(dataset_path, exist_ok=True)
                    
                    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(dataset_path)
                    
                    # Find the actual dataset directory (first directory in extracted content)
                    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
                    if len(subdirs) > 0:
                        dataset_path = os.path.join(dataset_path, subdirs[0])
                    
                    # Check if dataset has proper structure
                    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
                    
                    if len(class_dirs) < 2:
                        st.error("Dataset harus memiliki minimal 2 subfolder kelas!")
                    else:
                        st.info(f"Melatih model klasifikasi dengan {len(class_dirs)} kelas: {', '.join(class_dirs)}")
                        
                        # Create progress bar and metrics
                        progress_bar = st.progress(0)
                        metrics_container = st.container()
                        plot_container = st.container()
                        
                        # Create callback to update progress
                        class StreamlitCallback(tf.keras.callbacks.Callback):
                            def __init__(self, total_epochs):
                                self.total_epochs = total_epochs
                                self.current_epoch = 0
                                self.metrics_history = {}
                            
                            def on_epoch_begin(self, epoch, logs=None):
                                self.current_epoch = epoch
                                progress_bar.progress(epoch / self.total_epochs)
                            
                            def on_epoch_end(self, epoch, logs=None):
                                # Update metrics
                                for metric, value in logs.items():
                                    if metric not in self.metrics_history:
                                        self.metrics_history[metric] = []
                                    self.metrics_history[metric].append(value)
                                
                                # Display current metrics
                                with metrics_container:
                                    cols = st.columns(4)
                                    cols[0].metric("Epoch", f"{epoch+1}/{self.total_epochs}")
                                    cols[1].metric("Loss", f"{logs.get('loss', 0):.4f}")
                                    cols[2].metric("Accuracy", f"{logs.get('accuracy', 0):.4f}")
                                    cols[3].metric("Val Accuracy", f"{logs.get('val_accuracy', 0):.4f}")
                                
                                # Update plot
                                if epoch > 0 and epoch % 2 == 0:  # Update plot every 2 epochs
                                    with plot_container:
                                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                                        
                                        # Plot accuracy
                                        ax1.plot(self.metrics_history.get('accuracy', []), label='Training')
                                        ax1.plot(self.metrics_history.get('val_accuracy', []), label='Validation')
                                        ax1.set_title('Model Accuracy')
                                        ax1.set_ylabel('Accuracy')
                                        ax1.set_xlabel('Epoch')
                                        ax1.legend()
                                        
                                        # Plot loss
                                        ax2.plot(self.metrics_history.get('loss', []), label='Training')
                                        ax2.plot(self.metrics_history.get('val_loss', []), label='Validation')
                                        ax2.set_title('Model Loss')
                                        ax2.set_ylabel('Loss')
                                        ax2.set_xlabel('Epoch')
                                        ax2.legend()
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                        
                        # Train model with custom callback
                        try:
                            # Create custom callback
                            streamlit_callback = StreamlitCallback(epochs)
                            
                            # Train model with custom path
                            model_path = os.path.join("models", model_name)
                            model, history, class_names = train_classification_model(
                                dataset_path, 
                                epochs=epochs, 
                                batch_size=batch_size, 
                                img_size=img_size, 
                                validation_split=validation_split,
                                custom_callbacks=[streamlit_callback],
                                model_path=model_path
                            )
                            
                            # Update progress to 100%
                            progress_bar.progress(1.0)
                            
                            # Plot final training history
                            st.subheader("Hasil Pelatihan Final")
                            fig = plot_training_history(history)
                            st.pyplot(fig)
                            
                            # Show class mapping
                            st.subheader("Kelas yang Dilatih")
                            for i, name in enumerate(class_names):
                                st.write(f"{i}: {name}")
                            
                            st.success(f"Model klasifikasi berhasil dilatih dan disimpan di 'models/{model_name}'")
                            
                            # Reload the model
                            st.info("Memuat ulang model...")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Gagal melatih model: {e}")
                
                # Clean up
                if os.path.exists(dataset_zip_path):
                    os.remove(dataset_zip_path)
                if os.path.exists("temp_dataset"):
                    shutil.rmtree("temp_dataset")
        
        elif model_type == "Model Deteksi Objek (YOLO)":
            st.write("Unggah dataset deteksi objek (folder dengan subfolder Images dan XML Files)")
            
            # Upload dataset
            uploaded_dataset = st.file_uploader("Unggah dataset deteksi objek (ZIP)", type=["zip"])
            
            # Model name input
            model_name = st.text_input("Nama File Model", "yolov8n_custom.pt")
            if not model_name.endswith(".pt"):
                model_name += ".pt"
            
            # Training parameters
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.slider("Jumlah Epoch", 5, 100, 50)
                img_size = st.slider("Ukuran Gambar", 320, 1280, 640, 32)
            with col2:
                batch_size = st.slider("Batch Size", 4, 32, 16)
            
            # Train button
            if uploaded_dataset is not None and st.button("Latih Model Deteksi Objek"):
                with st.spinner("Memproses dataset..."):
                    # Save uploaded zip file
                    dataset_zip_path = "temp_dataset.zip"
                    with open(dataset_zip_path, "wb") as f:
                        f.write(uploaded_dataset.getbuffer())
                    
                    # Extract dataset
                    import zipfile
                    dataset_path = "temp_dataset"
                    os.makedirs(dataset_path, exist_ok=True)
                    
                    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(dataset_path)
                    
                    # Find the actual dataset directory (first directory in extracted content)
                    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
                    if len(subdirs) > 0:
                        dataset_path = os.path.join(dataset_path, subdirs[0])
                    
                    # Check if dataset has proper structure
                    if not os.path.exists(os.path.join(dataset_path, "Images")) or \
                       not os.path.exists(os.path.join(dataset_path, "XML Files")):
                        st.error("Dataset harus memiliki subfolder 'Images' dan 'XML Files'!")
                    else:
                        # Prepare YOLO dataset
                        with st.spinner("Menyiapkan dataset YOLO..."):
                            try:
                                yolo_dataset_path, class_mapping = prepare_yolo_dataset(dataset_path)
                                
                                # Show class mapping
                                st.subheader("Kelas yang Terdeteksi")
                                for class_name, class_id in class_mapping.items():
                                    st.write(f"{class_id}: {class_name}")
                                
                                # Create containers for live updates
                                progress_container = st.container()
                                metrics_container = st.container()
                                plot_container = st.container()
                                
                                # Create a placeholder for the training plot
                                plot_placeholder = st.empty()
                                
                                # Train YOLO model with live updates
                                with st.spinner(f"Melatih model YOLO dengan {epochs} epoch..."):
                                    # Create a placeholder for the progress bar
                                    progress_bar = st.progress(0)
                                    
                                    # Create a placeholder for metrics
                                    metrics_cols = st.columns(4)
                                    epoch_metric = metrics_cols[0].empty()
                                    loss_metric = metrics_cols[1].empty()
                                    precision_metric = metrics_cols[2].empty()
                                    recall_metric = metrics_cols[3].empty()
                                    
                                    # Setup callback for live updates
                                    def update_progress(progress=None, epoch=None, metrics=None):
                                        if progress is not None:
                                            progress_bar.progress(progress)
                                        if epoch is not None and metrics is not None:
                                            epoch_metric.metric("Epoch", f"{epoch}/{epochs}")
                                            loss_metric.metric("Loss", f"{metrics.get('loss', 0):.4f}")
                                            precision_metric.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                                            recall_metric.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                                    
                                    # Train model with custom name and callback
                                    model_path = train_yolo_model(
                                        yolo_dataset_path, 
                                        epochs=epochs, 
                                        img_size=img_size, 
                                        batch_size=batch_size,
                                        model_name=model_name,
                                        progress_callback=update_progress
                                    )
                                    
                                    # Update progress to 100%
                                    progress_bar.progress(1.0)
                                    
                                    st.success(f"Model YOLO berhasil dilatih dan disimpan di '{model_path}'")
                                    
                                    # Show training results
                                    results_path = os.path.join("runs", "detect", "yolo_microscopic_objects")
                                    if os.path.exists(os.path.join(results_path, "results.png")):
                                        st.subheader("Hasil Pelatihan")
                                        st.image(os.path.join(results_path, "results.png"), caption="Metrik Pelatihan YOLO")
                                    
                                    # Show example predictions if available
                                    if os.path.exists(os.path.join(results_path, "val_batch0_pred.jpg")):
                                        st.subheader("Contoh Prediksi")
                                        st.image(os.path.join(results_path, "val_batch0_pred.jpg"), caption="Contoh Prediksi pada Data Validasi")
                                    
                                    # Reload the model
                                    st.info("Memuat ulang model...")
                                    st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Gagal melatih model: {e}")
                
                # Clean up
                if os.path.exists(dataset_zip_path):
                    os.remove(dataset_zip_path)
                if os.path.exists("temp_dataset"):
                    shutil.rmtree("temp_dataset")
                if os.path.exists("yolo_dataset"):
                    shutil.rmtree("yolo_dataset")
    
    with model_tab:
        st.subheader("Model yang Tersedia")
        
        # Check for existing models
        if os.path.exists("models"):
            model_files = os.listdir("models")
            
            if len(model_files) > 0:
                st.write("Model yang tersedia:")
                for model_file in model_files:
                    st.write(f"- {model_file}")
                    
                # Add option to delete models
                model_to_delete = st.selectbox("Pilih model untuk dihapus", ["None"] + model_files)
                if model_to_delete != "None" and st.button("Hapus Model"):
                    try:
                        os.remove(os.path.join("models", model_to_delete))
                        st.success(f"Model {model_to_delete} berhasil dihapus")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Gagal menghapus model: {e}")
            else:
                st.info("Belum ada model yang tersedia. Silakan latih model terlebih dahulu.")
        else:
            st.info("Belum ada model yang tersedia. Silakan latih model terlebih dahulu.")

# About page
elif page == "Tentang Aplikasi":
    st.header("Tentang Aplikasi")
    st.markdown("""
    ## Aplikasi Deteksi Objek Preparat Mikroskopis
    
    Aplikasi ini dikembangkan untuk membantu analisis gambar preparat mikroskopis di laboratorium. 
    Aplikasi menggunakan:
    
    - **YOLO (You Only Look Once)**: Algoritma state-of-the-art untuk deteksi objek
    - **Deep Learning dengan Keras**: Untuk klasifikasi objek yang terdeteksi
    - **Streamlit**: Untuk antarmuka web yang interaktif
    
    ### Fitur Utama
    
    - Deteksi dan penghitungan otomatis berbagai objek mikroskopis
    - Klasifikasi objek berdasarkan jenisnya
    - Visualisasi hasil dengan anotasi gambar
    - Analisis statistik dari hasil deteksi
    - Penyimpanan hasil untuk analisis lebih lanjut
    - Pelatihan model kustom dengan dataset pengguna
    
    ### Cara Penggunaan
    
    1. Unggah gambar preparat mikroskopis
    2. Sesuaikan parameter deteksi jika diperlukan
    3. Klik tombol "Proses Gambar"
    4. Lihat hasil deteksi dan statistik
    5. Simpan hasil jika diperlukan
    6. Latih model kustom dengan dataset Anda sendiri
    
    ### Pengembangan Lebih Lanjut
    
    Aplikasi ini dapat dikembangkan lebih lanjut dengan:
    - Menambahkan fitur pelacakan perubahan dari waktu ke waktu
    - Mengintegrasikan dengan sistem laboratorium yang ada
    - Melatih model pada dataset khusus untuk meningkatkan akurasi
    """)