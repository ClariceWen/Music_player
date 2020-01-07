"""CSC148 Assignment 1 - Making Music

=== CSC148 Summer 2019 ===
Department of Computer Science,
University of Toronto

=== Module Description ===

This file contains classes that describe sound waves, instruments that play
these sounds and a function that plays multiple instruments simultaneously.

As discussed in the handout, you may not change any of the public behaviour
(attributes, methods) given in the starter code, but you can definitely add
new attributes, functions, classes and methods to complete your work here.

"""
from __future__ import annotations
import typing
import csv
import numpy
from helpers import play_sounds, make_sine_wave_array


class SimpleWave:
    """ A simple, sine-like sound wave.
    === Attribute ===
    _frequency: frequency of this wave in Hertz
    _duration: total time of this simple wave in seconds
    _amplitude:farthest distance in a cycle in this simple wave
    """
    _frequency: int
    _duration: float
    _amplitude: float

    def __init__(self, frequency: int,
                 duration: float, amplitude: float) -> None:
        """ Initialize this sine wave
        """
        self._frequency = frequency
        self._duration = duration
        if amplitude <= 1:
            self._amplitude = amplitude
        else:
            self._amplitude = 1

    def __eq__(self, other: SimpleWave) -> bool:
        """ Determined whether two simple wave is the same
        """
        return self._frequency == other._frequency and \
            self._duration == other._duration and \
            self._amplitude == other._amplitude

    def __ne__(self, other: SimpleWave) -> bool:
        """ Determined whether two simple wave is not the same
        """
        return self._frequency != other._frequency or \
            self._duration != other._duration or \
            self._amplitude != other._amplitude

    def __add__(self,
                other: ANYWAVE) -> ComplexWave:
        """ Add two wave together to combine a complex wave
        """
        if isinstance(other, SimpleWave):
            return ComplexWave([self, other])
        else:
            waves = [self] + other.get_waves()
            return ComplexWave(waves)

    def get_duration(self) -> float:
        """ Get the time of play in seconds of this simple wave
        """
        return self._duration

    def play(self) -> numpy.ndarray:
        """ Get a sine like numpy n-dimensional array
        """
        if self._amplitude > 0:
            return make_sine_wave_array(self._frequency, self._duration) * \
                   self._amplitude
        else:
            return make_sine_wave_array(self._frequency, self._duration)


class ComplexWave:
    """ A complex wave, not following the sine pattern
    """
    complexities: int
    _waves: typing.List[SimpleWave]

    def __init__(self, waves: typing.List[SimpleWave]) -> None:
        """ Initialize a complex wave
        """
        self._waves = waves
        self.complexities = len(waves)

    def __add__(self,
                other: ANYWAVE) -> ComplexWave:
        """ Add two wave together to get a new wave
        """
        if isinstance(other, SimpleWave):
            return ComplexWave(self.get_waves() + [other])
        else:
            return ComplexWave(self.get_waves() + other.get_waves())

    def complexity(self) -> int:
        """ Get the number of waves which used to combine this complex wave
        """
        return self.complexities

    def play(self) -> numpy.ndarray:
        """ Get the numpy array of this complex wave
        """
        if len(self.get_waves()) == 0:
            return numpy.array([])

        re = []
        length = 0
        for wave in self._waves:
            play = wave.play()
            re.append(play)
            length = max(length, len(play))
        zero_array = numpy.array(0)
        alter_result = []
        for array in re:
            alter_array = array
            if len(array) < length:
                for _ in range(length - len(array)):
                    alter_array = numpy.append(alter_array, zero_array)
            alter_result.append(alter_array)
        if len(alter_result) > 0:
            result = alter_result[0]
        else:
            result = None
        if len(alter_result) > 1:
            for array in alter_result[1:]:
                result += array
        else:
            pass
        if len(result) > 0:
            s = max(abs(result.max()), abs(result.min()))
            if s > 1:
                result = result / s
        return result

    def get_waves(self) -> typing.List[SimpleWave]:
        """ Get the list of waves used to combine this complex wave
        """
        return self._waves

    def get_duration(self) -> float:
        """ Get the whole duration of this complex wave
        Follow the longest simple component.
        """
        result = []
        waves = self.get_waves()
        for wave in waves:
            result.append(wave.get_duration())
        return max(result)


class SawtoothWave(ComplexWave):
    """ A Square wave """
    def __init__(self, frequency: int,
                 duration: float, amplitude: float) -> None:
        """ Initialize a square wave """
        waves = []
        for i in range(1, 11):
            wave = SimpleWave(frequency * i, duration, amplitude / i)
            waves.append(wave)
        ComplexWave.__init__(self, waves)


class SquareWave(ComplexWave):
    """ A SawtoothWave """

    def __init__(self, frequency: int,
                 duration: float, amplitude: float) -> None:
        """ Initialize a saw tooth wave """
        waves = []
        for i in range(1, 11):
            wave = SimpleWave(frequency * (2 * i - 1),
                              duration, amplitude / (2 * i - 1))
            waves.append(wave)
        ComplexWave.__init__(self, waves)


class Rest(ComplexWave):
    """ Silence """
    def __init__(self, duration: float) -> None:
        """ Initialize a silence sound """
        wave = SimpleWave(0, duration, 0)
        ComplexWave.__init__(self, [wave])


class Note:
    """ A note of sound """
    amplitude: float
    _waves: typing.List[ANYWAVE]

    def __init__(self, waves: typing.List[ANYWAVE]) -> None:
        """ Initialize this sound note """
        self.amplitude = 1.0
        self._waves = waves

    def __add__(self, other: Note) -> Note:
        """ Add two note to get a new one """

        n = Note(self._waves + other._waves)
        n.amplitude = max(self.amplitude, other.amplitude)
        return n

    def get_waves(self) -> typing.List[ANYWAVE]:
        """ Get the list of waves combining this note in specific order """
        return self._waves

    def get_duration(self) -> float:
        """ Get the whole duration of this note """
        result = 0.0
        for wave in self._waves:
            result += wave.get_duration()
        return result

    def play(self) -> numpy.ndarray:
        """Return a numpy array which represents each of the Note instance's
        component waves played in order.
        """
        if len(self.get_waves()) == 0:
            return numpy.array([])

        result = self._waves[0].play()
        for wave in self._waves[1:]:
            result = numpy.append(result, wave.play())
        if len(result) > 0 and result is not None:
            s = max(abs(result.max()), abs(result.min()))
            if s != 0:
                result = result * (self.amplitude / s)
        return result


class StutterNote(Note):
    """ A special note """
    amplitude: float

    def __init__(self, frequency: int,
                 duration: float, amplitude: float) -> None:
        """ Initialize a stutter note """

        sound = SawtoothWave(frequency, 0.025, amplitude)
        silence = Rest(0.025)

        integer_cycle = int(40 * duration)
        float_cycle = 40 * duration - integer_cycle

        if float_cycle > 0:
            if integer_cycle % 2 == 0:
                waves = [sound, silence] * int(integer_cycle / 2) + \
                        [SawtoothWave(frequency, float_cycle * 0.025,
                                      amplitude)]
            else:
                waves = [sound, silence] * int(integer_cycle // 2) + \
                        [sound] + [Rest(float_cycle * 0.025)]
        else:
            if integer_cycle % 2 == 0:
                waves = [sound, silence] * int(integer_cycle / 2)
            else:
                waves = [sound, silence] * int(integer_cycle // 2) + [sound]

        Note.__init__(self, waves)
        self.amplitude = amplitude


class Instrument:
    """ A instrument """
    _f_frequency: int
    _play_type: str
    _next: Note

    def __init__(self, f_frequency: int, play_type: str) -> None:
        """ Initialize an instrument """
        self._f_frequency = f_frequency
        self._play_type = play_type
        self._next = Note([])

    def get_duration(self) -> float:
        """ Get playing duration of this instrument """
        return self._next.get_duration()

    def next_notes(self,
                   note_info: typing.List[typing.Tuple[str, float, float]]
                   ) -> None:
        """ Playing next note
        [(ratio, amplitude, duration), (ratio, amplitude, duration)......]
        """
        raise NotImplementedError

    def play(self) -> numpy.ndarray:
        """ Play this instrument """
        return self._next.play()


class Baliset(Instrument):
    """ A Baliset Instruments """
    _f_frequency: int
    _play_type: str
    _next: Note
    _play: list

    def __init__(self) -> None:
        """ Initialize a baliset """
        Instrument.__init__(self, 196, 'SaSawtoothWave')
        self._play = []

    def next_notes(self,
                   note_info: typing.List[typing.Tuple[str, float, float]]) \
            -> None:
        """ Playing next note
        [(ratio, amplitude, duration), (ratio, amplitude, duration)......]
        """
        waves = []
        play = []
        for wave in note_info:
            if wave[0] != 'rest':
                ratio_part = wave[0].strip('"').split(":")
                ratio = int(ratio_part[0]) / int(ratio_part[1])
                duration = float(wave[2])
                amplitude = float(wave[1])

                sound_wave = SawtoothWave(int(ratio * self._f_frequency),
                                          duration, amplitude)
                waves.append(sound_wave)
                play.append(sound_wave)
            else:
                duration = float(wave[-1])
                waves.append(Rest(duration))
                play.append(Rest(duration))
        self._next = Note(waves)
        self._play = play

    def play(self) -> numpy.ndarray:
        """Play Baliset Instrument and return a numpy array.
                """
        n = numpy.array([])
        for each in self._play:
            n = numpy.concatenate((n, each.play()), 0)
        return n


class Holophonor(Instrument):
    """ A Holophonor Instruments """
    _f_frequency: int
    _play_type: str
    _next: Note
    _play: list

    def __init__(self) -> None:
        """ Initialize a Holophonor """
        Instrument.__init__(self, 65, 'StutterNote')
        self._play = []

    def next_notes(self, note_info: typing.List[typing.Tuple \
            [str, float, float]]) -> None:
        """ Playing next note
        [(ratio, amplitude, duration), (ratio, amplitude, duration)......]
        """

        waves = []
        play = []
        for wave in note_info:
            if wave[0] != 'rest':
                ratio_part = wave[0].strip("'").split(":")
                ratio = int(ratio_part[0]) / int(ratio_part[1])
                duration = float(wave[2])
                amplitude = float(wave[1])

                sound_note = StutterNote(round(ratio * self._f_frequency),
                                         duration, amplitude)
                waves.extend(sound_note.get_waves())
                play.append(sound_note)

            else:
                duration = float(wave[2])
                waves.append(Rest(duration))
                play.append(Rest(duration))
        self._next = Note(waves)
        self._play = play

    def play(self) -> numpy.ndarray:
        """Play Holophonor Instrument and return a numpy array.
        """
        n = numpy.array([])
        for each in self._play:
            n = numpy.concatenate((n, each.play()))
        return n


class Gaffophone(Instrument):
    """ A Gaffophone Instruments """
    _f_frequency: int
    _play_type: str
    _next: Note

    def __init__(self) -> None:
        """ Initialize a Gaffophone """
        Instrument.__init__(self, 131, 'Square-complex-wave')

    def next_notes(self,
                   note_info: typing.List[typing.Tuple[str, float, float]]
                   ) -> None:
        """ Playing next note
        [(ratio, amplitude, duration), (ratio, amplitude, duration)......]
        """
        waves = []
        for wave in note_info:
            if wave[0] != 'rest':
                ratio_part = wave[0].strip("'").split(":")
                ratio = int(ratio_part[0]) / int(ratio_part[1])

                frequency = int(ratio * self._f_frequency)
                duration = float(wave[2])
                amplitude = float(wave[1])

                square_wave1 = SquareWave(frequency, duration, amplitude)
                square_wave2 = SquareWave(int(frequency * 3 / 2), duration,
                                          amplitude)
                sound_wave = ComplexWave(square_wave1.get_waves() +
                                         square_wave2.get_waves())
                waves.append(sound_wave)
            else:
                duration = float(wave[2])
                waves.append(Rest(duration))
        self._next = Note(waves)


def _get_data(song_file: str) -> typing.List:
    # return list = [b_list, h_list, g_list]
    baliset_ = []
    holophonor = []
    gaffophone = []

    with open(song_file) as csv_file:
        instruments = next(csv.reader(csv_file))
        first_i = instruments[0]
        if len(instruments) > 1:
            second_i = instruments[1]
        if len(instruments) > 2:
            third_i = instruments[2]
        for row in csv.reader(csv_file):
            if first_i == 'Baliset':
                baliset_.append(row[0])
            elif first_i == 'Holophonor':
                holophonor.append(row[0])
            else:
                gaffophone.append(row[0])
            if len(instruments) > 1:
                if second_i == 'Baliset':
                    baliset_.append(row[1])
                elif second_i == 'Holophonor':
                    holophonor.append(row[1])
                else:
                    gaffophone.append(row[1])
            if len(instruments) > 2:
                if third_i == 'Baliset':
                    baliset_.append(row[2])
                elif third_i == 'Holophonor':
                    holophonor.append(row[2])
                else:
                    gaffophone.append(row[2])
    return [baliset_, holophonor, gaffophone, len(instruments)]


def play_song(song_file: str, beat: float) -> None:
    """ Play sound file according to beats """

    data = _get_data(song_file)
    baliset_ = data[0]
    holophonor = data[1]
    gaffophone = data[2]

    gaff = _to_note_info(_tupe(gaffophone, beat))
    holo = _to_note_info(_tupe(holophonor, beat))
    baliset = _to_note_info(_tupe(baliset_, beat))

    b = Baliset()
    h = Holophonor()
    g = Gaffophone()

    if data[3] == 1:
        if baliset_:
            for note in baliset:
                b.next_notes(note)
                play_sounds([b])
        elif holophonor:
            for note in holo:
                h.next_notes(note)
                play_sounds([h])
        else:
            for note in gaff:
                g.next_notes(note)
                play_sounds([g])

    elif data[3] == 2:
        if baliset_ and holophonor:
            _play_two(baliset, holo, b, h)
            # a list of note_info and instrument
        elif baliset_ and gaffophone:
            _play_two(baliset, gaff, b, g)
        else:
            _play_two(holo, gaff, h, g)

    else:
        if len(baliset) == min(len(baliset), len(holo), len(gaff)):
            _l1small(baliset, holo, gaff, [b, h, g])
        elif len(holo) == min(len(baliset), len(holo), len(gaff)):
            _l1small(holo, baliset, gaff, [h, b, g])
        else:
            _l1small(gaff, baliset, holo, [g, b, h])


def _play_two(note_info: typing.List, note_info2: typing.List,
              instrument: Instrument, instrument2: Instrument) -> None:
    len1, len2 = len(note_info), len(note_info2)
    if len1 < len2:
        for i in range(len1):
            instrument.next_notes(note_info[i])
            instrument2.next_notes(note_info2[i])
            play_sounds([instrument, instrument2])
        for i in range(len1, len2):
            instrument2.next_notes(note_info2[i])
            play_sounds([instrument2])
    elif len1 == len2:
        for i in range(len1):
            instrument.next_notes(note_info[i])
            instrument2.next_notes(note_info2[i])
            play_sounds([instrument, instrument2])
    else:
        for i in range(len2):
            instrument.next_notes(note_info[i])
            instrument2.next_notes(note_info2[i])
            play_sounds([instrument, instrument2])
        for i in range(len2, len1):
            instrument.next_notes(note_info[i])
            play_sounds([instrument])


def _l1small(l1: typing.List[typing.List[typing.Tuple[str, float, float]]],
             l2: typing.List[typing.List[typing.Tuple[str, float, float]]],
             l3: typing.List[typing.List[typing.Tuple[str, float, float]]],
             instruments: typing.List[Instrument]) -> None:
    """Playing sounds in 1 second by 1 second. Assuming l1 with the
    smallest length.
    first corresponding to l1
    second corresponding to l2
    third correspondfing to l3
    """
    len1, len2, len3 = len(l1), len(l2), len(l3)
    first, second, third = instruments[0], instruments[1], instruments[2]
    if len1 == 0:
        if len2 == 0:
            for i in range(len3):
                instruments[2].next_notes(l3[i])
                play_sounds([third])
        elif len3 == 0:
            for i in range(len2):
                instruments[1].next_notes(l2[i])
                play_sounds([second])

        elif len2 <= len3:
            for i in range(len2):
                instruments[1].next_notes(l2[i])
                third.next_notes(l3[i])
                play_sounds([second, third])
            for j in range(len2, len3):
                third.next_notes(l3[j])
                play_sounds([third])
        else:
            for i in range(len3):
                second.next_notes(l2[i])
                third.next_notes(l3[i])
                play_sounds([second, third])
            for j in range(len3, len2):
                third.next_notes(l2[j])
                play_sounds([second])

    else:
        for i in range(len1):
            first.next_notes(l1[i])
            second.next_notes(l2[i])
            third.next_notes(l3[i])
            play_sounds([first, second, third])
        if len2 <= len3:
            for j in range(len1, len2):
                second.next_notes(l2[j])
                third.next_notes(l3[j])
                play_sounds([second, third])
            for k in range(len2, len3):
                third.next_notes(l3[k])
                play_sounds([third])
        else:
            for j in range(len1, len3):
                third.next_notes(l3[j])
                second.next_notes(l2[j])
                play_sounds([second, third])
            for k in range(len3, len2):
                second.next_notes(l2[k])
                play_sounds([second])


def _to_note_info(lst: typing.List[typing.Tuple[str, float, float]]) -> \
        typing.List[typing.List[typing.Tuple[str, float, float]]]:
    t = 0
    j = 0
    k = 0
    new = []
    while j < len(lst):
        if t < 1:
            pre = t
            t += lst[j][-1]
            if t < 1:
                j += 1

            elif t == 1:
                new.append(lst[k:j + 1])
                k = j + 1
                j += 1
                t = 0
            else:
                d1 = 1 - pre
                po1 = lst[j][0]
                po2 = lst[j][1]
                po3 = lst[j][2]
                v = lst[k:j]
                v.append((po1, po2, d1))
                lst[j] = (po1, po2, po3 - d1)
                new.append(v)
                k = j
                t = 0
    if t > 0:
        last = lst[k: j] + [('rest', 0, 1 - t)]
        new.append(last)
    return new


def _tupe(lst: typing.List[str], beat: float) -> \
        typing.List[typing.Tuple[str, float, float]]:
    """Return a list of tuples.
    Step1 to extract a list until encounter an empty str
    Step2 transfer the remaining list of str to list of tuples
    """
    i = 0
    while i < len(lst) and lst[i] != '':
        i += 1
    lst = lst[:i]
    new = []
    for each in lst:
        if 'rest' in each:
            p1 = each.split(':')[0]
            p2 = float(each.split(':')[1]) * beat
            new.append((p1, 0, p2))
        else:
            l = each.split(':')
            new.append((l[0]+':'+l[1], float(l[2]), float(l[3]) * beat))
    return new


# This is a custom type for type annotations that
# refers to any of the following classes (do not
# change this code)
ANYWAVE = typing.TypeVar('ANYWAVE',
                         SimpleWave,
                         ComplexWave,
                         SawtoothWave,
                         SquareWave,
                         Rest)

if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={'extra-imports': ['helpers',
                                                  'typing',
                                                  'csv',
                                                  'numpy'],
                                'disable': ['E9997', 'E9998']})
