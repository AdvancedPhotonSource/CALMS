import sdl_agents
import unittest

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
        return super().setUpClass()

        

    def test_scan_move_up(self):
        request= "Move the sample up by 100"
        response = self.ag_fw.initiate_chat(request)
        print(response)
        self.assertEqual('foo'.upper(), 'FOO')

    def test_scan_move_down(self):
        request = "Move the sample down by 100 um"
        self.assertEqual('foo'.upper(), 'FOO')

    def beam_move_up(self):
        request= "Move the beam up by 10 um"
    
    def beam_move_down(self):
        request= "Move the beam down by 10 um"

    def scan2d_q1(self):
        request= "scan the sample in the vertical direction, from absolute position -100 um to 100 um, in 50 steps, with an exposure time of 1 sec"

    def scan2d_q2(self):
        request= "scan the beam in the vertical direction, from absolute position -10 um to 10 um, in 50 steps, with an exposure time of 1 sec"

    def scan2d_q3(self):
        request = "scan the sample in the vertical direction, from -100 um to 100 um with regard to its current position, in 50 steps, with an exposure time of 1 sec"

    def scan2d_q4(self):
        request = "do a 2D scan with an exposure time of 0.1 sec. the two motors are respectively hybridx and hybridy, range is 10 um from the current position, with a step size of 1 um"

    def scan2d_q5(self):
        request = "do a 2D scan with an exposure time of 0.1 sec. the two motors are respectively hybridx and hybridy, range is 10 um from the current position, taking 20 steps"

    def scan2d_q6(self):
        request = " do a 2D scan of an area of 20 um by 20 um using the x-ray beam, centered around the current position. The resolution if 1 um. The exposure time is 0.1 sec. Use hybridy as the outer loop motor."

    def scan2d_q7(self):
        request = "do a 2D scan of an area of 20 um by 20 um using the x-ray beam, centered around the current position. The resolution if 1 um. Use hybridy as the outer loop motor."

    def scan2d_alg_1(self):
        request = "do a 2D scan of an area of 20 um by 20 um using the x-ray beam, centered around the current position. The resolution if 1 um. The exposure time is 0.1 sec. Use hybridy as the outer loop motor."

    def scan2d_alg_2(self):
        request = "do a 2D scan of an area of 10 um by 10 um using the x-ray beam, centered around the current position. The resolution is 100 nm, and I need it done in an hour, set the exposure time accordingly. Use hybridy as the outer loop motor."

    def scan2d_alg_3(self):
        request = "do a 2D scan of an area of 10 um by 10 um using the x-ray beam, centered around the current position. I need it done in an hour. Use hybridy as the outer loop motor. should prompt as either the resolution or the exposure time needs to be known"

    

if __name__ == '__main__':
    unittest.main()


