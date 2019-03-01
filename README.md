Notice: Config and weights files were too big to upload to GitHub, so all needed files to run are not present.

Object detection is one of my interests in machine learning. I used this pneumonia detection challenge as an opportunity to learn object detection.

I found a pneumonia detection challenge hosted on Kaggle.com. (https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) A stat they was listed on the competition’s page that motivated me to attempt the challenge was that internationally 15% of all deaths of children under the age of 5, is caused by pneumonia.

I knew nothing about object detection at the start and so I learned a lot from it. A month into working on the project, my priorities changed, and this project was moved to the back burner. 

A few months later, I reattempted this challenge with additional knowledge and a new approach. This approach utilized the Darknet and YOLOv3 object detection. (https://pjreddie.com/darknet/yolo/) After 5 days of approximately 8 hours of training a day, the model achieved an F1 score of 0.74 and an average IoU of 64% on the validation set.

The model was still improving at the end of the first round of training, so further improvements can be made. For this first round of training, the training and validation sets only contained images that had detection boxes. This was so the model would get to a state where it could detect pneumonia the fastest. Toward the end of the training, I started to notice that the model began to send false positives. So, I believe that the model would benefit from some further training was some images that do not contain pneumonia. 
