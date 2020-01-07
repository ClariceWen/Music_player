from make_some_noise import *
import helpers as helper
import numpy as np
import random
import unittest


def count_wave(temp):
    num_swa = 0
    num_rset = 0
    for wave in temp:
        if isinstance(wave, SawtoothWave):
            num_swa += 1
        elif isinstance(wave, Rest):
            num_rset += 1
    return(num_swa, num_rset)

class test_simplewave(unittest.TestCase):
    def setUp(self):
        self.simple = SimpleWave(200, 0.5, 0.5)
        self.simple2 = SimpleWave(300, 1, 1)
        self.simple3 = SimpleWave(9000, 1, 1)
        self.complex = ComplexWave([SimpleWave(200, 0.5, 0.5), SimpleWave(300, 1, 1)])
        self.rest = Rest(10)

    def tearDown(self):
        self.simple = SimpleWave(200, 0.5, 0.5)
        self.simple2 = SimpleWave(300, 1, 1)
        self.complex = ComplexWave(
            [SimpleWave(200, 0.5, 0.5), SimpleWave(300, 0.5, 1)])

    def test_add(self):
        temp = self.simple + self.simple2
        self.assertEqual(True, len(self.complex.get_waves()) == 2)
        temp_waves = temp.get_waves()
        sorted(temp_waves, key=lambda x:x._duration)
        exp = self.complex.get_waves()
        sorted(exp, key = lambda x:x._duration)
        res = exp == temp_waves
        self.assertTrue(res, "You wave produced under addition should have exact same waves as its operands")
        self.assertEqual(True, isinstance(self.simple + self.complex, ComplexWave), "You should have a complex wave when you add a simple with a complex")
        res2 = self.simple2 + self.rest
        #self.assertTrue(res2.get_duration() - 10 < 0.0001, "The duration of adding a simple wave with a rest should be the one which has the longer time")
        self.assertTrue(isinstance(res2, ComplexWave), "I dont want to explain this")
        self.assertTrue(len(res2.get_waves()) > 1, "You should combine waves")


    def test_get_duration(self):
        self.assertEqual(True, self.simple.get_duration() <= 1 and self.simple2.get_duration() <= 1, "The duration for simple wave should always bound by 1")
        self.assertTrue((self.simple2 + self.simple3).get_duration() == 1)
        self.assertAlmostEqual(0.5, self.simple.get_duration())
        self.assertAlmostEqual(1.0, self.simple2.get_duration())


class test_complexwave(unittest.TestCase):
    def setUp(self):
        self.simple = SimpleWave(200, 0.5, 0.5)
        self.simple2 = SimpleWave(300, 1, 1)
        self.complex = ComplexWave(
            [SimpleWave(200, 0.5, 0.5), SimpleWave(300, 1, 1)])
        self.complex2 = self.complex + self.complex
        self.complex3 = ComplexWave([SimpleWave(200, 1, 0.5), SimpleWave(300, 1, 1)])
        self.complex4 = self.complex3 + self.complex3
        self.rest = Rest(10)

    def tearDown(self):
        self.simple = SimpleWave(200, 0.5, 0.5)
        self.simple2 = SimpleWave(300, 1, 1)
        self.complex = ComplexWave(
            [SimpleWave(200, 0.5, 0.5), SimpleWave(300, 1, 1)])
        self.complex2 = self.complex + self.complex

    def test_add(self):
        self.assertTrue((self.simple + self.simple2).get_waves() == self.complex.get_waves())
        self.assertTrue((self.complex + self.complex).get_waves() == self.complex2.get_waves())
        self.assertTrue((self.complex + self.rest).get_waves() == (self.complex.get_waves() + self.rest.get_waves()))
        self.assertTrue((len((self.complex + self.rest).get_waves()) == len(self.complex.get_waves()) + len(self.rest.get_waves())))
        self.assertTrue(len(self.complex2.get_waves()) == 4 and len(self.complex.get_waves()
                                                                          ) == self.complex.complexity())
        self.assertTrue(self.complex3.get_duration() == 1)
        self.assertTrue(len((self.complex + self.simple).get_waves()) == 3)
        self.assertTrue(self.complex4.get_duration() == 1)
        self.assertTrue(self.complex2.get_waves()[:2] == self.complex.get_waves())
        waves = (self.complex + self.simple).get_waves()
        sorted(waves, key=lambda x:x._frequency)
        temp = self.complex.get_waves() + [self.simple]
        sorted(temp, key=lambda x:x._frequency)
        self.assertTrue(temp == waves)

    def test_get_duration(self):
        wave_list = [SimpleWave(100, 999, 1) for i in range(10)]
        temp = SimpleWave(100, 1, 1)
        for wave in wave_list:
            temp += wave
        self.assertTrue(temp.get_duration() == 999)

    def test_get_duration_2(self):
        durations = [random.randint(1,9999) for i in range(10)]
        wave_list = [SimpleWave(100, durations[i], 1) for i in range(10)]
        temp = SimpleWave(100, 1, 1)
        for wave in wave_list:
            temp += wave
        self.assertTrue(temp.get_duration() == max(durations))

    def test_play(self):
        exp = self.complex2.play()
        act = (self.simple + self.simple2).play()
        self.assertTrue(len(exp) == helper._SAMPLE_RATE)
        self.assertTrue(len(exp) == len(act))
        np.testing.assert_allclose(exp,act,atol=0.001)
        temp1 = self.complex.play()
        temp1.__add__(temp1)
        np.testing.assert_allclose(temp1, self.complex2.play())



class test_Note(unittest.TestCase):
    def setUp(self):
        self.note = Note([SimpleWave(200, 1, 0.5),SimpleWave(300, 1, 0.5),
                          ComplexWave([SimpleWave(200, 1, 0.5),
                                       SimpleWave(300, 1, 1)])])
        self.wavelist = [SimpleWave(200, 1, 0.5),SimpleWave(300, 1, 0.5),
                          SimpleWave(200, 1, 0.5),
                                       SimpleWave(300, 1, 1)]
        self.note2 = self.note + self.note

    def tearDown(self):
        self.note.amplitude = 1

    def test_add(self):
        self.assertTrue(len(self.note2.get_waves()) == 6)
        self.assertTrue(self.note2.get_duration() == 6)
        self.note.amplitude = 0.5
        self.note3 = Note([])
        self.note3.amplitude = 0.3
        self.note3 = self.note3 + self.note
        self.assertTrue(self.note3.amplitude == 0.5)

    def test_getwave(self):
        temp = []
        for wave in self.note.get_waves():
            if isinstance(wave, ComplexWave):
                for swave in wave.get_waves():
                    temp.append(swave)
            else:
                temp.append(wave)
        self.assertTrue(self.wavelist == temp)

    def test_getwave_2(self):
        temp = []
        for wave in self.note2.get_waves():
            if isinstance(wave, ComplexWave):
                for swave in wave.get_waves():
                    temp.append(swave)
            else:
                temp.append(wave)
        self.assertTrue((self.wavelist + self.wavelist) == temp)

    def test_play(self):
        temp = self.note2.play()
        temp = list(temp)
        half_temp = temp[:len(temp) // 2]
        half_temp_2 = temp[len(temp) // 2:]
        self.assertCountEqual(half_temp, half_temp_2,
                              "complex2 is built by two idetincal complex waves, so the first half elements should be identical to the other half elements")
        self.assertCountEqual(half_temp, list(self.note.play()),
                              "The first half elements of complex2 should be exact same as its component")
        self.assertTrue(max(half_temp) <= 1)
        self.assertTrue(min(half_temp) >= -1)

    def test_play2(self):
        temp = self.note.play()
        new_note = Note([SimpleWave(200, 1, 0.5),SimpleWave(300, 1, 0.5),
                          ComplexWave([SimpleWave(200, 1, 0.5),
                                       SimpleWave(300, 1, 1)])])
        new_note.amplitude = 0.5
        new_temp = new_note.play()
        self.assertCountEqual(temp * 0.5, new_temp)
        self.assertTrue(max(temp) <= 1 and max(new_temp) <= 1)
        self.assertTrue(min(temp) >= -1 and min(temp) >= -1)


class test_sawtooth(unittest.TestCase):
    def setUp(self):
        self.saw = SawtoothWave(150, 2.2, 0.8)

    def test_setUp(self):
        self.assertTrue(isinstance(self.saw, ComplexWave))
        self.assertTrue(len(self.saw.get_waves()) == 10)
        self.assertTrue(self.saw.get_duration() == 2.2)
        waves = [wave for wave in self.saw.get_waves()]
        waves_freq = [(wave._frequency // 150) for wave in waves]
        self.assertCountEqual(waves_freq,[i for i in range(1,11)])
        waves_amp = [wave._amplitude for wave in waves]
        self.assertCountEqual(waves_amp, [0.8 / i for i in range(1, 11)])

    def test_complexity(self):
        self.assertTrue(10, self.saw.complexity())
        self.assertTrue(11, (self.saw + SimpleWave(100, 1, 1)).complexity())
        self.assertTrue(12, (self.saw + ComplexWave([SimpleWave(100, 1, 1)]*2)))

    def test_add(self):
        complex = self.saw + SimpleWave(120, 0.8,0.8)
        self.assertTrue(isinstance(complex, ComplexWave))
        self.assertTrue(len(complex.get_waves()) == 11)
        waves = complex.get_waves()
        temp = []
        for w in waves:
            temp.append(w == SimpleWave(120, 0.8, 0.8))
        self.assertTrue(any(temp))
        temp = []
        for w in waves:
            for wave in self.saw.get_waves():
                temp.append(w == wave)
        self.assertTrue(temp.count(True) == 10)

    def test_play(self):
        complex = self.saw + self.saw
        temp = self.saw.play()
        res = complex.play()
        np.testing.assert_allclose(temp, res, atol = 0.001)


class test_square(unittest.TestCase):
    def setUp(self):
        self.squ = SquareWave(150, 2.2, 0.8)

    def test_setUp(self):
        self.assertTrue(isinstance(self.squ, ComplexWave))
        self.assertTrue(len(self.squ.get_waves()) == 10)
        self.assertTrue(self.squ.get_duration() == 2.2)
        waves = [wave for wave in self.squ.get_waves()]
        waves_freq = [(wave._frequency // 150) for wave in waves]
        self.assertCountEqual(waves_freq,[i for i in range(1,20,2)])
        waves_amp = [wave._amplitude for wave in waves]
        self.assertCountEqual(waves_amp, [0.8 / i for i in range(1, 20,2)])

    def test_complexity(self):
        self.assertTrue(10, self.squ.complexity())
        self.assertTrue(11, (self.squ + SimpleWave(100, 1, 1)).complexity())
        self.assertTrue(12, (self.squ + ComplexWave([SimpleWave(100, 1, 1)]*2)))

    def test_add(self):
        complex = self.squ + SimpleWave(120, 0.8,0.8)
        self.assertTrue(isinstance(complex, ComplexWave))
        self.assertTrue(len(complex.get_waves()) == 11)
        waves = complex.get_waves()
        temp = []
        for w in waves:
            temp.append(w == SimpleWave(120, 0.8, 0.8))
        self.assertTrue(any(temp))
        temp = []
        for w in waves:
            for wave in self.squ.get_waves():
                temp.append(w == wave)
        self.assertTrue(temp.count(True) == 10)

    def test_play(self):
        complex = self.squ + self.squ
        assert max(complex.play() <= 1)
        assert min(complex.play() >= -1)


class test_rest(unittest.TestCase):
    def setUp(self):
        self.rest = Rest(10)
        self.assertTrue(isinstance(self.rest, ComplexWave))
        self.assertTrue(isinstance((self.rest + ComplexWave([SimpleWave(100, 1, 1)]*2)), ComplexWave))
        self.assertTrue((self.rest + ComplexWave([SimpleWave(100, 1, 1)]*2)).complexity() > 2)


class test_stutternote(unittest.TestCase):
    def setUp(self):
        self.stu = StutterNote(440, 1.5, 1)
        self.stu2 = StutterNote(440, 1.01, 0.8)

    def test_init(self):
        temp = self.stu.get_waves()
        self.assertEqual(60, len(temp), "You should have 60 waves in your stutter note")
        (num_swa, num_rset) = count_wave(temp)
        self.assertEqual(30, num_swa, "You should have 30 SawtoothWave")
        self.assertEqual(30, num_rset, "You should have 30 rset")
        self.assertTrue(self.stu.amplitude == 1, "You should make the amplitude be 1")
        self.assertTrue(self.stu2.amplitude == 0.8, "")

    def test_init_corner(self):
        temp = self.stu2.get_waves()
        self.assertEqual(41, len(temp), "You should add an extra wave beyond 40 simple waves for the extra 0.01 second")
        (num_swa, num_rset) = count_wave(temp)
        self.assertEqual(True, num_swa >= 20, "You should have at least 20 SawtoothWave")
        self.assertEqual(True, num_rset >= 20, "You should have at least 20 Rset")
        self.assertEqual(True, num_rset + num_swa == 41, "You should only contain SawtoothWave and Rset in your solution")

    def test_duration(self):
        duration = self.stu.get_duration()
        self.assertEqual(True, isinstance(duration, float), "Return type should be float")
        self.assertAlmostEqual(1.5, duration)

    def test_duration_2(self):
        duration = self.stu2.get_duration()
        self.assertEqual(True, isinstance(duration, float), "Return type should be float")
        self.assertAlmostEqual(1.01, duration)


class test_baliset(unittest.TestCase):
    def setUp(self):
        self.baliset = Baliset()
        self.note_info = [("1:2", 1, 1)]
        self.note_info2 = [("1:3", 0.5, 0.5)]
        self.note_info3 = [("1:2", 0.5, 0.5), ("1:4", 0.5, 0.5)]

    def test_getduration(self):
        self.baliset.next_notes([("1:2",1, 1)])
        self.assertTrue(self.baliset.get_duration() == 1)

    def test_getduration2(self):
        self.baliset.next_notes(self.note_info2)
        self.assertAlmostEqual(self.baliset.get_duration(), 0.5)

    def test_play(self):
        self.baliset.next_notes(self.note_info)
        act = self.baliset.play()
        exp = SawtoothWave(98, 1, 1).play()
        np.testing.assert_allclose(exp, act, atol=0.01)

    def test_play2(self):
        self.baliset.next_notes(self.note_info3)
        act = self.baliset.play()
        exp = SawtoothWave(98, 0.5, 0.5).play()
        exp2 = SawtoothWave(49, 0.5, 0.5).play()
        np.testing.assert_allclose(act[:len(exp) - 10], exp[:-10],
                                   atol=0.0001)
        np.testing.assert_allclose(act[-len(exp2) + 10:], exp2[10:],
                                   atol=0.0001)


class test_Holophonor(unittest.TestCase):
    def setUp(self):
        self.holophonor = Holophonor()
        self.note_info = [("1:5", 1, 1)]
        self.note_info2 = [("1:3", 0.5, 0.5)]
        self.note_info3 = [("1:5", 0.5, 0.5), ("1:13", 0.5, 0.5)]

    def test_getduration(self):
        self.holophonor.next_notes([("1:5", 1, 1)])
        self.assertAlmostEqual(self.holophonor.get_duration(), 1)

    def test_getduration2(self):
        self.holophonor.next_notes(self.note_info2)
        self.assertAlmostEqual(self.holophonor.get_duration(), 0.5)

    def test_play(self):
        self.holophonor.next_notes(self.note_info)
        act = self.holophonor.play()
        np.testing.assert_allclose(StutterNote(13, 1, 1).play(),
                                   self.holophonor.play())

    def test_play2(self):
        self.holophonor.next_notes(self.note_info3)
        act = self.holophonor.play()
        stu1 = StutterNote(13, 0.5, 0.5)
        stu2 = StutterNote(5, 0.5, 0.5)
        stu = stu1+ stu2
        exp = stu.play()
        np.testing.assert_allclose(act, exp, atol=0.001)


class test_gaffophone(unittest.TestCase):
    def setUp(self):
        self.gaff = Gaffophone()
        self.note_info = [("2:1", 1, 1)]
        self.note_info2 = [("3:1", 0.5, 0.5)]
        self.note_info3 = [("2:1", 0.5, 0.41), ("4:1", 0.5, 0.59)]

    def test_getduration(self):
        self.gaff.next_notes(self.note_info)
        self.assertAlmostEqual(self.gaff.get_duration(), 1)

    def test_getduration2(self):
        self.gaff.next_notes(self.note_info2)
        self.assertAlmostEqual(self.gaff.get_duration(), 0.5)

    def test_nextnote(self):
        self.gaff.next_notes(self.note_info)
        act = self.gaff.play()
        sq1 = SquareWave(262, 1, 1)
        sq2 = SquareWave(393, 1, 1)
        sq = sq1 + sq2
        exp = sq.play()
        note = Note([sq1 + sq2])
        np.testing.assert_allclose(exp, act)
        np.testing.assert_allclose(exp, note.play())

    def test_nextnote2(self):
        self.gaff.next_notes(self.note_info3)
        act = self.gaff.play()
        sq1 = SquareWave(262, 0.41, 0.5)
        sq2 = SquareWave(393, 0.41, 0.5)
        sq3 = SquareWave(524, 0.59, 0.5)
        sq4 = SquareWave(786, 0.59, 0.5)
        note = Note([(sq1 + sq2),(sq3 + sq4)])
        exp = note.play()
        np.testing.assert_allclose(exp, act)





if __name__ == "__main__":
    unittest.main(exit=False)
