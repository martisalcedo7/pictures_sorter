import cv2
import matplotlib.pyplot as plt
import os


def get_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades +
                                        "haarcascade_frontalface_default.xml")

source_path = 'pictures/source/'
selected_path = 'pictures/selected/'
unselected_path = 'pictures/unselected/'

selected_counter = 0
total_counter = 0
for image_file_name in get_files(source_path):

    print("Processing image",total_counter,"-", image_file_name)
    image_path = os.path.join(source_path, image_file_name)

    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    face = face_classifier.detectMultiScale(gray_image,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(5, 5))

    if len(face) > 0:
        # for (x, y, w, h) in face:
        #     cv2.rectangle(gray_image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        os.system("cp " + image_path + " " + os.path.join(selected_path, image_file_name))
        selected_counter += 1
    else:
        os.system("cp " + image_path + " " + os.path.join(unselected_path, image_file_name))
    # plt.imshow(gray_image)
    # plt.show()
    total_counter += 1

print(selected_counter, "images selected of", total_counter, "given images.")