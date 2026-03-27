import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import pathlib
#Extraction of zip file
def extract_zip_file(p):
  import zipfile
  zip_ref = zipfile.ZipFile(p, 'r')
  zip_ref.extractall('./')
  zip_ref.close()
def run(l):
        import tensorflow as tf
        from tensorflow import keras
        from keras import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        import matplotlib.pyplot as plt
        import tensorflow as tf
        from tensorflow import keras
        from keras import Sequential
        from keras.layers import Dense, Dropout, BatchNormalization
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, classification_report
        import joblib
        import matplotlib.pyplot as plt
        #Extraction of training and testing data
        train_ds = keras.utils.image_dataset_from_directory(
        directory=l[0],
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(128, 128),
        seed=42
        )
        validation_ds = keras.utils.image_dataset_from_directory(
            directory=l[1],
            labels='inferred',
            label_mode='int',
            batch_size=32,
            image_size=(128, 128),
            seed=42
        )
        print("✅ Datasets loaded successfully!")
        def process(image, label):
            image = tf.cast(image/255., tf.float32)
            return image, label
        class_names = train_ds.class_names
        NUM_CLASSES = len(class_names)

        train_ds = train_ds.map(process)
        validation_ds = validation_ds.map(process)
        print("Creating optimized CNN model...")
        #Defining the CNN Model
        model = Sequential([
          Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128,128,3)),
          BatchNormalization(),
          MaxPooling2D(2,2),
          Dropout(0.25),

          Conv2D(64, (3,3), activation='relu', padding='same'),
          BatchNormalization(),
          MaxPooling2D(2,2),
          Dropout(0.25),

          Conv2D(128, (3,3), activation='relu', padding='same'),
          BatchNormalization(),
          MaxPooling2D(2,2),
          Dropout(0.25),

          Conv2D(256, (3,3), activation='relu', padding='same'),
          BatchNormalization(),
          GlobalAveragePooling2D(),
          Dropout(0.5),

          Dense(256, activation='relu'),
          BatchNormalization(),
          Dropout(0.5),
          Dense(128, activation='relu', name='feature_layer'),
          Dense(NUM_CLASSES, activation='softmax')
      ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Model Summary:")
        model.summary()
        history = model.fit(train_ds, epochs=10, validation_data=validation_ds,verbose=1,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
        )
        feature_extractor = keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer('feature_layer').output
        )
        #Feature Extraction
        print("Extracting features efficiently...")
        def extract_features(dataset):
            features, labels = [], []
            for images, lbls in tqdm(dataset, desc="Extracting"):
                feats = feature_extractor(images, training=False)
                features.append(feats.numpy())
                labels.append(lbls.numpy())
            return np.concatenate(features), np.concatenate(labels)

        train_features, train_labels = extract_features(train_ds)
        val_features, val_labels = extract_features(validation_ds)

        print(f"Training features: {train_features.shape}")
        print(f"Validation features: {val_features.shape}")
        omega = 0.97
        pop_size = 15
        max_iter = 18
        w, c1, c2 = 0.8, 1.8, 1.8
        infection_prob = 0.1
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
            train_features, train_labels, test_size=0.2, stratify=train_labels, random_state=42  # Smaller validation set
        )
        n_features = train_features.shape[1]
        print(f"Total features: {n_features}")
        def fast_fitness_function(mask):
            if np.sum(mask) == 0:
                return 1.0
            if np.sum(mask) < 5:
                return 1.0
            Xtr = X_train_opt[:, mask == 1]
            Xv = X_val_opt[:, mask == 1]
            clf = LogisticRegression(
                max_iter=500,
                solver='liblinear',
                random_state=42,
                n_jobs=-1
            )
            clf.fit(Xtr, y_train_opt)
            acc = accuracy_score(y_val_opt, clf.predict(Xv))
            feat_ratio = np.sum(mask) / len(mask)
            if feat_ratio < 0.05 or feat_ratio > 0.7:
                penalty = 0.2
            else:
                penalty = 0.0

            return omega * (1 - acc) + (1 - omega) * feat_ratio + penalty
        #Using soft computing
        print("Initializing PSO + BEOSA...")
        positions = np.random.randint(0, 2, (pop_size, n_features))
        velocities = np.random.uniform(-2, 2, (pop_size, n_features))
        pbest = positions.copy()
        pbest_fitness = np.array([fast_fitness_function(p) for p in positions])
        gbest = pbest[np.argmin(pbest_fitness)].copy()
        gbest_fitness = np.min(pbest_fitness)

        print(f"Initial fitness: {gbest_fitness:.4f}, Features: {np.sum(gbest)}")
        best_fitness_history = []
        best_accuracy_history = []

        print("Starting optimization loop...")
        for iteration in range(max_iter):
            for i in range(pop_size):
                r1, r2 = np.random.rand(), np.random.rand()

                velocities[i] = (
                    w * velocities[i] +
                    c1 * r1 * (pbest[i] - positions[i]) +
                    c2 * r2 * (gbest - positions[i])
                )
                prob = 1 / (1 + np.exp(-velocities[i]))
                new_position = (np.random.rand(n_features) < prob).astype(int)
                if np.random.rand() < infection_prob:
                    mutation_count = max(1, n_features // 20)
                    indices = np.random.choice(n_features, mutation_count, replace=False)
                    new_position[indices] = 1 - new_position[indices]

                positions[i] = new_position
                current_fitness = fast_fitness_function(new_position)
                if current_fitness < pbest_fitness[i]:
                    pbest_fitness[i] = current_fitness
                    pbest[i] = new_position.copy()

                    if current_fitness < gbest_fitness:
                        gbest_fitness = current_fitness
                        gbest = new_position.copy()
            if np.sum(gbest) > 0:
                Xtr_best = X_train_opt[:, gbest == 1]
                Xv_best = X_val_opt[:, gbest == 1]
                clf_temp = LogisticRegression(max_iter=500, random_state=42)
                clf_temp.fit(Xtr_best, y_train_opt)
                current_accuracy = accuracy_score(y_val_opt, clf_temp.predict(Xv_best))
            else:
                current_accuracy = 0.0

            best_fitness_history.append(gbest_fitness)
            best_accuracy_history.append(current_accuracy)

            if (iteration + 1) % 5 == 0:
                print(f"Iter {iteration+1}/{max_iter} | Fit: {gbest_fitness:.4f} | "
                    f"Feat: {np.sum(gbest)} | Acc: {current_accuracy:.4f}")

        print("\n✅ Optimization Completed!")
        selected_mask = gbest
        X_train_selected = train_features[:, selected_mask == 1]
        X_val_selected = val_features[:, selected_mask == 1]
        # Using Ensemble Learning
        print(f"Selected features: {X_train_selected.shape[1]} out of {n_features}")
        from sklearn.ensemble import VotingClassifier
        from sklearn.svm import SVC
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        svm_clf = SVC(kernel='linear', probability=True, random_state=42)
        lr_clf = LogisticRegression(max_iter=1000, random_state=42)
        final_classifier = VotingClassifier(
            estimators=[
                ('rf', rf_clf),
                ('svm', svm_clf),
                ('lr', lr_clf)
            ],
            voting='soft'
        )

        print("Training final ensemble classifier...")
        final_classifier.fit(X_train_selected, train_labels)
        val_predictions = final_classifier.predict(X_val_selected)
        final_accuracy = accuracy_score(val_labels, val_predictions)

        print(f"🎯 FINAL VALIDATION ACCURACY: {final_accuracy * 100:.2f}%")
        feature_extractor.save("./fast_pso_beosa_feature_extractor.h5")
        joblib.dump(final_classifier, "./fast_pso_beosa_classifier.pkl")
        joblib.dump(selected_mask, "./fast_pso_beosa_mask.pkl")
        import numpy as np
        import tensorflow as tf
        from tensorflow import keras
        from keras import Sequential
        from keras.layers import Dense, Dropout, BatchNormalization
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, classification_report
        import joblib
        import matplotlib.pyplot as plt
        #N-cross validation
        def cnn_n_fold_cross_validation(n_splits=5):

            print(f"🚀 Starting {n_splits}-Fold Cross Validation with CNN...")

            # Load the saved models and data
            try:
                feature_extractor = keras.models.load_model("./fast_pso_beosa_feature_extractor.h5", compile=False)
                selected_mask = joblib.load("./fast_pso_beosa_mask.pkl")
                print("✅ Models loaded successfully!")
            except:
                print("❌ Models not found. Please run the training first.")
                return
            print("Loading datasets for cross-validation...")
            try:
                full_ds = keras.utils.image_dataset_from_directory(
                    directory=l[2],
                    labels='inferred',
                    label_mode='int',
                    batch_size=64,
                    image_size=(128, 128),
                    seed=42,
                    shuffle=True
                )
            except Exception as e:
                    print(f"Error loading dataset: {e}")
                    return

            def process(image, label):
                image = tf.cast(image/255., tf.float32)
                return image, label

            full_ds = full_ds.map(process)
            print("Extracting features for cross-validation...")
            all_features, all_labels = [], []
            for images, labels in full_ds:
                feats = feature_extractor(images, training=False).numpy()
                all_features.append(feats)
                all_labels.append(labels.numpy())

            all_features = np.concatenate(all_features)
            all_labels = np.concatenate(all_labels)
            selected_features = all_features[:, selected_mask == 1]

            print(f"Selected features shape: {selected_features.shape}")
            print(f"Labels shape: {all_labels.shape}")
            print(f"Number of classes: {len(np.unique(all_labels))}")
            def create_cnn_classifier(input_dim, num_classes=5):
                model = Sequential([
                  Dense(512, activation='relu', input_shape=(input_dim,)),
                  BatchNormalization(),
                  Dropout(0.5),
                  Dense(256, activation='relu'),
                  BatchNormalization(),
                  Dropout(0.4),
                  Dense(128, activation='relu'),
                  BatchNormalization(),
                  Dropout(0.3),
                  Dense(num_classes, activation='softmax')
              ])
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )

                return model

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            fold_accuracies = []
            fold_losses = []
            fold_histories = []
            precisions = []
            recalls = []
            f1_scores = []
            print(f"\n📊 Performing {n_splits}-Fold Cross Validation with CNN...")

            for fold, (train_idx, val_idx) in enumerate(skf.split(selected_features, all_labels)):
                print(f"\n🔄 Fold {fold + 1}/{n_splits}")

                X_train, X_val = selected_features[train_idx], selected_features[val_idx]
                y_train, y_val = all_labels[train_idx], all_labels[val_idx]

                cnn_classifier = create_cnn_classifier(input_dim=X_train.shape[1], num_classes=len(np.unique(all_labels))) # Pass number of classes
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=0
                )

                history = cnn_classifier.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping],
                    verbose=0
                )

                val_loss, val_accuracy = cnn_classifier.evaluate(X_val, y_val, verbose=0)

                fold_accuracies.append(val_accuracy)
                fold_losses.append(val_loss)
                fold_histories.append(history.history)


                print(f"   Fold {fold + 1} Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
                print(f"   Fold {fold + 1} Loss: {val_loss:.4f}")
                from sklearn.metrics import precision_score, recall_score, f1_score
                y_pred = np.argmax(cnn_classifier.predict(X_val), axis=1)
                fold_precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                fold_recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                fold_f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                print(f"   Fold {fold + 1} Precision: {fold_precision:.4f}")
                print(f"   Fold {fold + 1} Recall: {fold_recall:.4f}")
                print(f"   Fold {fold + 1} F1-Score: {fold_f1:.4f}")
            print("\n" + "="*60)
            #Printing Final Metrics
            print("🎯 CNN CROSS-VALIDATION RESULTS SUMMARY")
            print("="*60)

            mean_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            mean_loss = np.mean(fold_losses)

            print(f"\n📊 Overall Results:")
            print(f"Mean Accuracy: {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")
            print(f"Standard Deviation: {std_accuracy:.4f} ({std_accuracy*100:.2f}%)")
            print(f"Mean Loss: {mean_loss:.4f}")
            print(f"\n📈 Fold-wise Accuracies:")
            for i, acc in enumerate(fold_accuracies):
                print(f"  Fold {i+1}: {acc:.4f} ({acc*100:.2f}%)")
            print(f"\nFinal Accuracy: {mean_accuracy*100:.2f}% +/- {std_accuracy*100:.2f}%")
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 2, 1)
            for i, history in enumerate(fold_histories):
                plt.plot(history['accuracy'], label=f'Fold {i+1} Train', alpha=0.7)
                plt.plot(history['val_accuracy'], label=f'Fold {i+1} Val', linestyle='--', alpha=0.7)
            plt.title('Model Accuracy Across Folds')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            for i, history in enumerate(fold_histories):
                plt.plot(history['loss'], label=f'Fold {i+1} Train', alpha=0.7)
                plt.plot(history['val_loss'], label=f'Fold {i+1} Val', linestyle='--', alpha=0.7)
            plt.title('Model Loss Across Folds')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

            print("\n🏋️ Training Final CNN Model on Full Dataset...")
            final_cnn_classifier = create_cnn_classifier(input_dim=selected_features.shape[1], num_classes=len(np.unique(all_labels))) # Pass number of classes

            final_history = final_cnn_classifier.fit(
                selected_features, all_labels,
                epochs=30,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )

            final_cnn_classifier.save("./final_cnn_feature_classifier.h5")
            print("✅ Final CNN classifier saved!")
            return {
                'fold_accuracies': fold_accuracies,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'final_model': final_cnn_classifier
            }

        cnn_results = cnn_n_fold_cross_validation(n_splits=5)
import os
import sys
#Calling all the functions for code execution
k1 = os.listdir('./')
extract_zip_file(f"./{sys.argv[1]}")
k2 = os.listdir('./')
base_dir = [k for k in k2 if k not in k1]
train_dir = ""
test_dir = ""
val_dir = ""
for i in os.listdir(os.path.join('./' , base_dir[0])):
  if("train" in str(i).lower()):
    train_dir = os.path.join('./' , base_dir[0] , i)
    break
val_dir = train_dir
for i in os.listdir(os.path.join('./' , base_dir[0])):
  if("val" in str(i).lower()):
    val_dir = os.path.join('./' , base_dir[0] , i)
    break
for i in os.listdir(os.path.join('./' , base_dir[0])):
  if("test" in str(i).lower()):
    test_dir = os.path.join('./' , base_dir[0] , i)
    break
run([train_dir , val_dir , test_dir])