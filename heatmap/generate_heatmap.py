import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil

def determine_final_category(row):
    if row['confidence'] <= 0.4:
        return 'nondiagnostic'
    elif row['predicted_class'] == 'normal' and row['confidence'] > 0.4:
        return 'normal'
    else:
        return 'tumor'

def generate_heatmap_from_df(df, matrix_shape=(54, 54), pixel_size=100):
    # Initialize three matrices, which are used to store the information of normal, tumor and nondiagnostic respectively
    heatmap_matrix_normal = np.zeros(matrix_shape)
    heatmap_matrix_tumor = np.zeros(matrix_shape)
    heatmap_matrix_nondiagnostic = np.zeros(matrix_shape)

    # Iterate over each row in the DataFrame and update the corresponding matrix based on the Final Category
    for index, row in df.iterrows():
        filename = row['filename']
        final_category = row['Final Category']

        # Extract coordinates (x, y) from filename
        file_name_without_extension = filename.split('.')[0]  
        x, y = file_name_without_extension.split('_')[1:3]  
        x, y = int(x), int(y)

        # The coordinates are converted to the lattice of the adaptation matrix
        x //= 100
        y //= 100

        if final_category == "normal":
            for i in range(3):
                for j in range(3):
                    heatmap_matrix_normal[y+i, x+j] += 1
        elif final_category == "tumor":
            for i in range(3):
                for j in range(3):
                    heatmap_matrix_tumor[y+i, x+j] += 1
        elif final_category == "nondiagnostic":
            for i in range(3):
                for j in range(3):
                    heatmap_matrix_nondiagnostic[y+i, x+j] += 1


    # You can choose any matrix you want
    # return heatmap_matrix_combined
    return heatmap_matrix_tumor

df = pd.read_csv('./predictions.csv')

df['Final Category'] = df.apply(determine_final_category, axis=1)

df = df.sort_values(by='filename')

matrix = generate_heatmap_from_df(df)

pixel_size = 100

total_pixels = matrix.shape[0] * pixel_size

plt.figure(figsize=(10, 10))

heatmap = plt.imshow(matrix, cmap='RdYlBu_r', interpolation='nearest', extent=[0, total_pixels, 0, total_pixels], aspect='auto')

plt.axis('off')
plt.tight_layout()
plt.savefig('./heatmap.png')

# Delete temporarily stored folders
shutil.rmtree('./t')