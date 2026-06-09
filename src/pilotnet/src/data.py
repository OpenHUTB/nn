from utils.screen import message
import os, json, cv2

class PilotData(object):

    # given the path to an image file as argument, return a PilotData object
    def __init__(self, path_to: 'Path to the image file', image_file: 'An image file with driving data as filename' = '', isTraining = True):
        if isTraining:
            self.speed, self.steering, self.throttle, self.brake, self.image = self.parse_train(path_to, image_file)
        else:
            self.speed, self.steering, self.throttle, self.brake, self.image = self.parse_test(path_to)

    # this method overrides str() for PilotData objects
    def __str__(self):
        return f'PilotData(speed={self.speed:.1f}, steering={self.steering:.4f}, throttle={self.throttle:.4f}, brake={self.brake:.4f})'

    def __repr__(self):
        return f"PilotData(speed={self.speed}, steering={self.steering}, throttle={self.throttle}, brake={self.brake}, image={self.image})"

    # create data for training
    def parse_train(self, path_to, image_file):
        # remove the last 4 characters from filename which would be the file extension
        # then parse the resulting string into a list
        data = json.loads(image_file[:-4])
        # read and resize the image
        image = cv2.imread(f'{path_to}{image_file}')
        if image is None:
            raise ValueError(f'Failed to read image: {path_to}{image_file}')
        image = cv2.resize(image, (160, 120))
        # data format: [speed, steering, throttle, brake]
        return (data[0], data[1], data[2], data[3], image)
    
    # create data for prediction
    def parse_test(self, file_path):
        # read and resize the image
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f'Failed to read image: {file_path}')
        image = cv2.resize(image, (160, 120))
        # reshape image to make it consumable for the Input neuron
        image = image.reshape(1, 120, 160, 3)
        # return format: (speed=0, steering=0, throttle=0, brake=0, image)
        return (0, 0, 0, 0, image)

class Data():
    def __init__(self, isTraining: 'whether to prepare data for training or prediction' = True):
        self.data = self.generate_data()

    def generate_data(self):
        data = []
        skipped = 0
        # get context & loop though each directory
        with os.scandir('recordings/') as recordings:
            for recording in recordings:
                # get context & loop through each image
                message(f'Extracting from {recording.name}')
                with os.scandir(recording) as images:
                    for image in images:
                        try:
                            # add a new PilotData instance to array
                            data.append(PilotData(f'recordings/{recording.name}/', image.name))
                        except (ValueError, Exception) as e:
                            skipped += 1
                            continue
        if skipped > 0:
            message(f'Skipped {skipped} corrupted images')
        return data

    # return the first 3/4 of total data
    def training_data(self):
        return self.data[:int((len(self.data)*3)/4)]
    
    # return last 1/4 of total data
    def testing_data(self):
        return self.data[int((len(self.data)*3)/4)+1:]