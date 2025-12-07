import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class FaceRecognitionSystem:
    def __init__(self, input_shape=(128, 128, 3), num_classes=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def create_cnn_model(self):
        """إنشاء نموذج CNN للتعرف على الوجه"""
        model = models.Sequential([
            # الطبقة الأولى
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # الطبقة الثانية
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # الطبقة الثالثة
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # الطبقة الرابعة
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # طبقات Fully Connected
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            # طبقة الإخراج
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def load_and_preprocess_data(self, data_dir):
        """تحميل وتجهيز بيانات التدريب"""
        images = []
        labels = []
        
        print(f"جاري تحميل البيانات من {data_dir}...")
        
        for person_name in os.listdir(data_dir):
            person_dir = os.path.join(data_dir, person_name)
            if os.path.isdir(person_dir):
                for img_file in os.listdir(person_dir):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(person_dir, img_file)
                        
                        # قراءة الصورة
                        img = cv2.imread(img_path)
                        if img is not None:
                            # تغيير الحجم
                            img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
                            # تحويل BGR إلى RGB
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            # Normalize
                            img = img.astype('float32') / 255.0
                            
                            images.append(img)
                            labels.append(person_name)
        
        images = np.array(images)
        labels = np.array(labels)
        
        return images, labels
    
    def prepare_labels(self, labels):
        """تحضير التسميات للتدريب"""
        # ترميز التسميات
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        encoded_labels = keras.utils.to_categorical(encoded_labels, self.num_classes)
        
        return encoded_labels
    
    def build_model(self):
        """بناء النموذج مع المترجم والدالة المفقودة"""
        self.model = self.create_cnn_model()
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """تدريب النموذج"""
        print("بدء التدريب...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            ),
            keras.callbacks.ModelCheckpoint(
                'best_face_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """تقييم النموذج"""
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"الدالة المفقودة: {loss:.4f}")
        print(f"الدقة: {accuracy:.4f}")
        return loss, accuracy
    
    def predict(self, image):
        """التنبؤ بهوية الوجه"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        predictions = self.model.predict(image)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        if confidence > 0.7:  # عتبة الثقة
            person_name = self.label_encoder.inverse_transform([predicted_class])[0]
            return person_name, confidence
        else:
            return "Unknown", confidence
    
    def save_model(self, model_path='face_recognition_model.h5'):
        """حفظ النموذج"""
        self.model.save(model_path)
        print(f"تم حفظ النموذج في {model_path}")
    
    def load_model(self, model_path='face_recognition_model.h5'):
        """تحميل النموذج"""
        self.model = keras.models.load_model(model_path)
        print(f"تم تحميل النموذج من {model_path}")

class FaceDetector:
    """كشف الوجوه في الوقت الحقيقي"""
    def __init__(self):
        # استخدام Haar Cascade لاكتشاف الوجه
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_faces(self, image):
        """كشف الوجوه في الصورة"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def extract_face(self, image, face_coords):
        """استخراج الوجه من الإحداثيات"""
        x, y, w, h = face_coords
        face = image[y:y+h, x:x+w]
        return face

def real_time_recognition(model_path='face_recognition_model.h5'):
    """التعرف على الوجه في الوقت الحقيقي باستخدام الكاميرا"""
    # تحميل النموذج
    face_system = FaceRecognitionSystem()
    face_system.load_model(model_path)
    
    face_detector = FaceDetector()
    
    # فتح الكاميرا
    cap = cv2.VideoCapture(0)
    
    print("بدء التعرف على الوجه في الوقت الحقيقي...")
    print("اضغط على 'q' للخروج")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # كشف الوجوه
        faces = face_detector.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # رسم مربع حول الوجه
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # استخراج الوجه
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size > 0:
                # معالجة الوجه
                face_img = cv2.resize(face_img, (128, 128))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = face_img.astype('float32') / 255.0
                
                # التنبؤ
                name, confidence = face_system.predict(face_img)
                
                # عرض النتيجة
                label = f"{name}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # عرض الإطار
        cv2.imshow('Face Recognition', frame)
        
        # الخروج عند الضغط على 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """الدالة الرئيسية"""
    print("=" * 50)
    print("نظام التعرف على الوجه باستخدام TensorFlow")
    print("=" * 50)
    
    # إنشاء نظام التعرف على الوجه
    face_system = FaceRecognitionSystem(input_shape=(128, 128, 3))
    
    # تحميل البيانات (افترض أن البيانات في مجلد 'dataset')
    data_dir = 'dataset'
    
    if os.path.exists(data_dir):
        # تحميل وتجهيز البيانات
        images, labels = face_system.load_and_preprocess_data(data_dir)
        
        if len(images) == 0:
            print("لم يتم العثور على صور في المجلد dataset!")
            print("يرجى إنشاء هيكل المجلد كما يلي:")
            print("dataset/")
            print("├── person1/")
            print("│   ├── img1.jpg")
            print("│   └── img2.jpg")
            print("└── person2/")
            print("    ├── img1.jpg")
            print("    └── img2.jpg")
            return
        
        # تحضير التسميات
        encoded_labels = face_system.prepare_labels(labels)
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            images, encoded_labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print(f"عدد صور التدريب: {len(X_train)}")
        print(f"عدد صور التحقق: {len(X_val)}")
        print(f"عدد صور الاختبار: {len(X_test)}")
        print(f"عدد الأفراد: {face_system.num_classes}")
        
        # بناء النموذج
        face_system.build_model()
        
        # تدريب النموذج
        history = face_system.train(X_train, y_train, X_val, y_val, epochs=30)
        
        # تقييم النموذج
        print("\nتقييم النموذج على بيانات الاختبار:")
        face_system.evaluate(X_test, y_test)
        
        # حفظ النموذج
        face_system.save_model()
        
        # عرض رسوم بيانية للتدريب
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
        print("\nهل تريد تشغيل التعرف في الوقت الحقيقي؟ (y/n)")
        choice = input().strip().lower()
        
        if choice == 'y':
            real_time_recognition()
    else:
        print(f"المجلد {data_dir} غير موجود!")
        print("يرجى إنشاء مجلد dataset ووضع صور الوجوه فيه")

if __name__ == "__main__":
    main()