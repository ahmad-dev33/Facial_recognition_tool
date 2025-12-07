
import os
import numpy as np
import cv2

def create_dummy_dataset():
    # إنشاء هيكل المجلدات
    persons = ['ahmad', 'mohammed', 'sara', 'fatima', 'ali']
    
    # إنشاء مجلد dataset إذا لم يكن موجوداً
    os.makedirs('dataset', exist_ok=True)
    
    for person in persons:
        person_dir = f'dataset/{person}'
        os.makedirs(person_dir, exist_ok=True)
        
        print(f"إنشاء صور لـ {person}...")
        
        # إنشاء 5 صور وهمية لكل شخص
        for i in range(5):
            # إنشاء صورة عشوائية (128x128)
            img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
            
            # إضافة بعض الملامح الوهمية
            if i % 2 == 0:
                # إضافة دائرة للوجه
                cv2.circle(img, (64, 64), 40, (100, 100, 200), -1)
            
            # حفظ الصورة
            img_path = f'{person_dir}/image_{i+1}.jpg'
            cv2.imwrite(img_path, img)
            print(f"  - تم إنشاء {img_path}")
    
    # إنشاء ملف README للـ dataset
    with open('dataset/README.md', 'w', encoding='utf-8') as f:
        f.write("# Face Recognition Dataset\n\n")
        f.write("## هيكل المجلدات\n")
        f.write("- كل مجلد يمثل شخصاً مختلفاً\n")
        f.write("- كل صورة داخل المجلد تمثل عينة للشخص\n")
        f.write("- حجم الصورة: 128x128 بكسل\n\n")
        f.write("## الأفراد الموجودين\n")
        for person in persons:
            f.write(f"- {person}\n")
    
    print(f"\nتم إنشاء {len(persons)} أشخاص بـ {len(persons)*5} صورة")

if __name__ == "__main__":
    create_dummy_dataset()