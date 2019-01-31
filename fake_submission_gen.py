from submission import SubmissionWriter
import json
import os

""" Valid submission generation example. """

# load test image list
dataset_root = '/datasets/speed'
with open(os.path.join(dataset_root, 'test.json'), 'r') as f:
    test_images = json.load(f)
with open(os.path.join(dataset_root, 'real_test.json'), 'r') as f:
    real_test_images = json.load(f)

submission = SubmissionWriter()

# iterating over all test and real test images, appending submission
for image in test_images[::-1]:

    filename = image['filename']

    # arbitrary prediction, just to store something.
    q = [1.0, 0.0, 0.0, 0.0]
    r = [10.0, 0.0, 0.0]

    submission.append_test(filename, q, r)

for real_image in real_test_images:
    filename = real_image['filename']
    q = [.71, .71, 0.0, 0.0]
    r = [9.0, .1, .1]
    submission.append_real_test(filename, q, r)

submission.export(suffix='debug')
print('Submission exported.')
