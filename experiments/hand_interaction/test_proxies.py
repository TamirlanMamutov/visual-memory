import unittest
import numpy as np
from experiments.hand_interaction.extract import samples
from experiments.hand_interaction.proxies import proxy_features

class HandContracts(unittest.TestCase):
    def test_sampling_is_prequery_and_bounded(self):
        track=[[f,0,0,1,1] for f in range(100)]
        result=samples(track,100)
        self.assertLessEqual(len(result),18);self.assertTrue(all(0<=f<100 for f in result))
        with self.assertRaises(ValueError):samples(track,99)
    def test_missing_hand_not_release(self):
        track=[[0,10,10,20,20],[1,10,10,20,20]]
        hand=dict(box=[5,5,25,25],center=[15,15],side='Left')
        d={0:dict(hands=[hand]),1:dict(hands=[])};images={f:np.zeros((64,64,3),np.uint8) for f in [0,1]}
        vector,coverage=proxy_features(track,[0,1],d,images)
        self.assertEqual(vector[9:11],[0.,0.]);self.assertEqual(coverage['observed_hand_pairs'],0)
    def test_blank_background_motion_is_missing(self):
        track=[[0,10,10,20,20],[1,10,10,20,20]]
        hand=dict(box=[5,5,25,25],center=[15,15],side='Left')
        d={f:dict(hands=[hand]) for f in [0,1]};images={f:np.zeros((64,64,3),np.uint8) for f in [0,1]}
        vector,coverage=proxy_features(track,[0,1],d,images)
        self.assertEqual(coverage['camera_valid_pairs'],0);self.assertEqual(vector[6:8],[0.,0.])

if __name__=='__main__':unittest.main()
