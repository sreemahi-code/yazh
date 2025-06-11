import numpy as np
from sklearn.preprocessing import LabelEncoder

# Replace this with your actual list of class labels
labels = ["Likhe Jo Khat Tuje", "Back Up Friend","Breathing In","Could I Be Enough","Golden NeonNiteClub","keshi - less of you"
,"Let You Go","Life","Never Let Go","The Hush","The King"]



# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save the classes to a .npy file
np.save("classes.npy", label_encoder.classes_)

print("✅ Saved classes.npy with classes:", label_encoder.classes_)
