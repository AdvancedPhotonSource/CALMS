import sdl_agents
import unittest
import glob
import os
import re

"""
Implement s26 questions as a test suite. Allows for easy running and evaluation if the answers are correct.
"""


class S26CodeGenQuestions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ag_fw= sdl_agents.AutoGenSystem('ArgoLLMs',
                                             's26_workdir',
                                             's26_commands/S26_commandline_full.py',)
        cls.ag_fw.register_local_agents()
        cls.output_dir = 's26_workdir'
        return super().setUpClass()


    @classmethod
    def tearDownClass(cls):
        cls.cleanup_output_directory()
        return super().tearDownClass()
    

    @classmethod
    def cleanup_output_directory(cls):
        files = glob.glob(os.path.join(cls.output_dir, '*'))
        for file in files:
            if os.path.isfile(file):
                try:
                    os.remove(file)
                    print(f"Removed: {file}")
                except Exception as e:
                    print(f"Error removing {file}: {e}")   
    

    def check_output_file(self, expected_pattern):
        files = glob.glob(os.path.join(self.output_dir, '*'))
        if not files:
            print("No files found in output directory")
            return False
            
        filepath = max(files, key=os.path.getmtime)
        print(f"Checking most recent file: {os.path.basename(filepath)}")
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            print(content)
            
            if expected_pattern in content:
                return True
            return False
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return False

    def test_scan_move_up(self):
        request= "Move the sample up by 100"
        response = self.ag_fw.initiate_chat(request)
        self.assertTrue(self.check_output_file('movr(samy, 100)'))

    def test_scan_move_down(self):
        request = "Move the sample down by 100 um"
        response = self.ag_fw.initiate_chat(request)
        self.assertTrue(self.check_output_file('movr(samy, -100)'))

    def test_beam_move_up(self):
        request= "Move the beam up by 10 um"
        response = self.ag_fw.initiate_chat(request)
        self.assertTrue(self.check_output_file('movr(hybridy, 10)'))

    def beam_move_down(self):
        request= "Move the beam down by 10 um"
        response = self.ag_fw.initiate_chat(request)
        self.assertTrue(self.check_output_file('movr(hybridy, -10)'))

    def scan2d_q1(self):
        request= "scan the sample in the vertical direction, from absolute position -100 um to 100 um, in 50 steps, with an exposure time of 1 sec"
        response = self.ag_fw.initiate_chat(request)
        self.assertTrue(self.check_output_file('unlock_hybrid(); scan1d(samy, -100, 100, 50, 1, Absolute=True); lock_hybrid();'))

    def scan2d_q2(self):
        request= "scan the beam in the vertical direction, from absolute position -10 um to 10 um, in 50 steps, with an exposure time of 1 sec"
        response = self.ag_fw.initiate_chat(request)
        self.assertTrue(self.check_output_file('scan1d(hybridy, -10, 10, 50, 1, Absolute=True) '))

    def scan2d_q3(self):
        request = "scan the sample in the vertical direction, from -100 um to 100 um with regard to its current position, in 50 steps, with an exposure time of 1 sec"
        response = self.ag_fw.initiate_chat(request)
        self.assertTrue(self.check_output_file('movr(hybridy, 10)'))

    def scan2d_q4(self):
        request = "do a 2D scan with an exposure time of 0.1 sec. the two motors are respectively hybridx and hybridy, range is 10 um from the current position, with a step size of 1 um"
        response = self.ag_fw.initiate_chat(request)
        self.assertTrue(self.check_output_file('unlock_hybrid(); scan1d(samy, -100, 100, 50, 1, Absolute=False); lock_hybrid();'))

    def scan2d_q5(self):
        request = "do a 2D scan with an exposure time of 0.1 sec. the two motors are respectively hybridx and hybridy, range is 10 um from the current position, taking 20 steps"
        response = self.ag_fw.initiate_chat(request)
        self.assertTrue(self.check_output_file('unlock_hybrid(); scan1d(samy, -100, 100, 50, 1, Absolute=False); lock_hybrid();'))

    def scan2d_q6(self):
        request = " do a 2D scan of an area of 20 um by 20 um using the x-ray beam, centered around the current position. The resolution if 1 um. The exposure time is 0.1 sec. Use hybridy as the outer loop motor."
        response = self.ag_fw.initiate_chat(request)
        self.assertTrue(self.check_output_file('scan2d(hybridx, -10, 10, 20, hybridy, -10, 10, 20, 0.1, Absolute=False);'))

    def scan2d_q7(self):
        request = "do a 2D scan of an area of 20 um by 20 um using the x-ray beam, centered around the current position. The resolution if 1 um. Use hybridy as the outer loop motor."
        response = self.ag_fw.initiate_chat(request)
        self.assertTrue(self.check_output_file('scan2d(hybridx, -10, 10, 20, hybridy, -10, 10, 20, 0.1, Absolute=False);'))

    def scan2d_alg_1(self):
        request = "do a 2D scan of an area of 20 um by 20 um using the x-ray beam, centered around the current position. The resolution if 1 um. The exposure time is 0.1 sec. Use hybridy as the outer loop motor."
        response = self.ag_fw.initiate_chat(request)
        self.assertTrue(self.check_output_file('scan2d(hybridy, -10, 10, 20, hybridx, -10, 10, 20, 0.1, Absolute=False);'))

    def scan2d_alg_2(self):
        request = "do a 2D scan of an area of 10 um by 10 um using the x-ray beam, centered around the current position. The resolution is 100 nm, and I need it done in an hour, set the exposure time accordingly. Use hybridy as the outer loop motor."
        response = self.ag_fw.initiate_chat(request)
        self.assertTrue(self.check_output_file('scan2d(hybridy, 3, 5, 100, hybridx, 3, 5, 100, 1, Absolute=False)'))

    def scan2d_alg_3(self):
        request = "do a 2D scan of an area of 10 um by 10 um using the x-ray beam, centered around the current position. I need it done in an hour. Use hybridy as the outer loop motor. should prompt as either the resolution or the exposure time needs to be known"
        response = self.ag_fw.initiate_chat(request)
        self.assertTrue(self.check_output_file('scan2d(hybridy, -5, 5, 100, hybridx, -5, 5, 100, 0.036, Absolute=False);'))

    

if __name__ == '__main__':
    unittest.main()


