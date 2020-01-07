
from __future__ import annotations
import typing
import csv
import numpy
from helpers import play_sound, play_sounds, make_sine_wave_array
from make_some_noise import SimpleWave, ComplexWave, SawtoothWave, Note, \
    StutterNote, Instrument, play_song

# play_song('song.csv', 1)
play_song('swan_lake.csv', 0.2)
# play_song('swan_lake2.csv', 0.2)
# play_song('swan_lake3.csv', 1.4)
# play_song('spanish_violin.csv', 0.2)


# s = SawtoothWave(150, 2.2, 0.8)
# play = s.play()
# should the play.max() == 0.8 or play.max() == 1 ???????

# numpy.testing.assert_allclose(play_one, play_two, atol=0.001)
#
# print(len(play_one))
# print(len(play_two))
# print(saw.complexity())
# print(saw_two.complexity())
# print(play_one.max())
# print(play_two.max())


# with open('song.csv') as csv_file:
#     a = next(csv.reader(csv_file))
#     print(a)
#     for row in csv.reader(csv_file):
#         print(row)
