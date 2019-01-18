import json
import os
from datetime import datetime
import csv


class SubmissionWriter:

    """ Class for collecting results and exporting submission. """

    def __init__(self):
        self.test_results = []
        self.tron_results = []
        return

    def _append(self, filename, q, r, tron):
        if tron:
            self.tron_results.append({'filename': filename, 'q': q, 'r': r})
        else:
            self.test_results.append({'filename': filename, 'q': q, 'r': r})
        return

    def append_test(self, filename, q, r):

        """ Append pose estimation for test image to submission. """

        self._append(filename, q, r, tron=False)
        return

    def append_tron(self, filename, q, r):

        """ Append pose estimation for tron image to submission. """

        self._append(filename, q, r, tron=True)
        return

    def checks(self):
        return True

    def export(self, out_dir='', suffix=None):

        """ Exporting submission json file containing the collected pose estimates. """

        sorted_test = sorted(self.test_results, key=lambda k: k['filename'])
        sorted_tron = sorted(self.tron_results, key=lambda k: k['filename'])
        results = {'test': sorted_test,
                   'tron': sorted_tron}
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        if suffix is None:
            suffix = timestamp
        submission_path = os.path.join(out_dir, 'submission_{}.csv'.format(suffix))
        with open(submission_path, 'w') as f:
            csv_writer = csv.writer(f, lineterminator='\n')
            for result in (sorted_test + sorted_tron):
                csv_writer.writerow([result['filename'], *(result['q'] + result['r'])])

        print('Submission saved to {}.'.format(submission_path))
        return
