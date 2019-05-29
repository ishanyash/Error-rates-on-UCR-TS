# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:04:44 2019

@author: Ishan Yash
"""

"""Implements Alphabet cuts."""
import numpy as np


def cuts_for_asize(a_size):
    """Generate a set of alphabet cuts for its size."""
    """ Typically, we generate cuts in R as follows:
        get_cuts_for_num <- function(num) {
        cuts = c(-Inf)
        for (i in 1:(num-1)) {
            cuts = c(cuts, qnorm(i * 1/num))
            }
            cuts
        }

        get_cuts_for_num(3) """
    options = {
        2: np.array([-np.inf,  0.00]),
        3: np.array([-np.inf, -0.4307273, 0.4307273]),
        4: np.array([-np.inf, -0.6744898, 0, 0.6744898]),
        5: np.array([-np.inf, -0.841621233572914, -0.2533471031358,
                    0.2533471031358, 0.841621233572914]),
        6: np.array([-np.inf, -0.967421566101701, -0.430727299295457, 0,
                    0.430727299295457, 0.967421566101701]),
        7: np.array([-np.inf, -1.06757052387814, -0.565948821932863,
                    -0.180012369792705, 0.180012369792705, 0.565948821932863,
                    1.06757052387814]),
        8: np.array([-np.inf, -1.15034938037601, -0.674489750196082,
                    -0.318639363964375, 0, 0.318639363964375,
                    0.674489750196082, 1.15034938037601]),
        9: np.array([-np.inf, -1.22064034884735, -0.764709673786387,
                    -0.430727299295457, -0.139710298881862, 0.139710298881862,
                    0.430727299295457, 0.764709673786387, 1.22064034884735]),
        10: np.array([-np.inf, -1.2815515655446, -0.841621233572914,
                     -0.524400512708041, -0.2533471031358, 0, 0.2533471031358,
                     0.524400512708041, 0.841621233572914, 1.2815515655446]),
        11: np.array([-np.inf, -1.33517773611894, -0.908457868537385,
                     -0.604585346583237, -0.348755695517045,
                     -0.114185294321428, 0.114185294321428, 0.348755695517045,
                     0.604585346583237, 0.908457868537385, 1.33517773611894]),
        12: np.array([-np.inf, -1.38299412710064, -0.967421566101701,
                     -0.674489750196082, -0.430727299295457,
                     -0.210428394247925, 0, 0.210428394247925,
                     0.430727299295457, 0.674489750196082, 0.967421566101701,
                     1.38299412710064]),
        13: np.array([-np.inf, -1.42607687227285, -1.0200762327862,
                     -0.736315917376129, -0.502402223373355,
                     -0.293381232121193, -0.0965586152896391,
                     0.0965586152896394, 0.293381232121194, 0.502402223373355,
                     0.73631591737613, 1.0200762327862, 1.42607687227285]),
        14: np.array([-np.inf, -1.46523379268552, -1.06757052387814,
                     -0.791638607743375, -0.565948821932863, -0.36610635680057,
                     -0.180012369792705, 0, 0.180012369792705,
                     0.36610635680057, 0.565948821932863, 0.791638607743375,
                     1.06757052387814, 1.46523379268552]),
        15: np.array([-np.inf, -1.50108594604402, -1.11077161663679,
                     -0.841621233572914, -0.622925723210088,
                     -0.430727299295457, -0.2533471031358, -0.0836517339071291,
                     0.0836517339071291, 0.2533471031358, 0.430727299295457,
                     0.622925723210088, 0.841621233572914, 1.11077161663679,
                     1.50108594604402]),
        16: np.array([-np.inf, -1.53412054435255, -1.15034938037601,
                     -0.887146559018876, -0.674489750196082,
                     -0.488776411114669, -0.318639363964375,
                     -0.157310684610171, 0, 0.157310684610171,
                     0.318639363964375, 0.488776411114669, 0.674489750196082,
                     0.887146559018876, 1.15034938037601, 1.53412054435255]),
        17: np.array([-np.inf, -1.5647264713618, -1.18683143275582,
                     -0.928899491647271, -0.721522283982343,
                     -0.541395085129088, -0.377391943828554,
                     -0.223007830940367, -0.0737912738082727,
                     0.0737912738082727, 0.223007830940367, 0.377391943828554,
                     0.541395085129088, 0.721522283982343, 0.928899491647271,
                     1.18683143275582, 1.5647264713618]),
        18: np.array([-np.inf, -1.59321881802305, -1.22064034884735,
                     -0.967421566101701, -0.764709673786387,
                     -0.589455797849779, -0.430727299295457,
                     -0.282216147062508, -0.139710298881862, 0,
                     0.139710298881862, 0.282216147062508, 0.430727299295457,
                     0.589455797849779, 0.764709673786387, 0.967421566101701,
                     1.22064034884735, 1.59321881802305]),
        19: np.array([-np.inf, -1.61985625863827, -1.25211952026522,
                     -1.00314796766253, -0.8045963803603, -0.633640000779701,
                     -0.47950565333095, -0.336038140371823, -0.199201324789267,
                     -0.0660118123758407, 0.0660118123758406,
                     0.199201324789267, 0.336038140371823, 0.47950565333095,
                     0.633640000779701, 0.8045963803603, 1.00314796766253,
                     1.25211952026522, 1.61985625863827]),
        20: np.array([-np.inf, -1.64485362695147, -1.2815515655446,
                     -1.03643338949379, -0.841621233572914, -0.674489750196082,
                     -0.524400512708041, -0.385320466407568, -0.2533471031358,
                     -0.125661346855074, 0, 0.125661346855074, 0.2533471031358,
                     0.385320466407568, 0.524400512708041, 0.674489750196082,
                     0.841621233572914, 1.03643338949379, 1.2815515655446,
                     1.64485362695147]),
    }

    return options[a_size]

"""Discord discovery routines."""
import numpy as np
from saxpy.visit_registry import VisitRegistry
from saxpy.distance import early_abandoned_dist
from saxpy.znorm import znorm


def find_best_discord_brute_force(series, win_size, global_registry,
                                  z_threshold=0.01):
    """Early-abandoned distance-based discord discovery."""
    best_so_far_distance = -1.0
    best_so_far_index = -1

    outerRegistry = global_registry.clone()

    outer_idx = outerRegistry.get_next_unvisited()

    while ~np.isnan(outer_idx):

        outerRegistry.mark_visited(outer_idx)

        candidate_seq = znorm(series[outer_idx:(outer_idx+win_size)],
                              z_threshold)

        nnDistance = np.inf
        innerRegistry = VisitRegistry(len(series) - win_size)

        inner_idx = innerRegistry.get_next_unvisited()

        while ~np.isnan(inner_idx):
            innerRegistry.mark_visited(inner_idx)

            if abs(inner_idx - outer_idx) > win_size:

                curr_seq = znorm(series[inner_idx:(inner_idx+win_size)],
                                 z_threshold)
                dist = early_abandoned_dist(candidate_seq,
                                            curr_seq, nnDistance)

                if (~np.isnan(dist)) and (dist < nnDistance):
                    nnDistance = dist

            inner_idx = innerRegistry.get_next_unvisited()

        if ~(np.inf == nnDistance) and (nnDistance > best_so_far_distance):
            best_so_far_distance = nnDistance
            best_so_far_index = outer_idx

        outer_idx = outerRegistry.get_next_unvisited()

    return (best_so_far_index, best_so_far_distance)


def find_discords_brute_force(series, win_size, num_discords=2,
                              z_threshold=0.01):
    """Early-abandoned distance-based discord discovery."""
    discords = list()

    globalRegistry = VisitRegistry(len(series))
    globalRegistry.mark_visited_range(len(series) - win_size, len(series))

    while (len(discords) < num_discords):

        bestDiscord = find_best_discord_brute_force(series, win_size,
                                                    globalRegistry,
                                                    z_threshold)

        if -1 == bestDiscord[0]:
            break

        discords.append(bestDiscord)

        mark_start = bestDiscord[0] - win_size
        if 0 > mark_start:
            mark_start = 0

        mark_end = bestDiscord[0] + win_size
        '''if len(series) < mark_end:
            mark_end = len(series)'''

        globalRegistry.mark_visited_range(mark_start, mark_end)

    return discords

"""Distance computation."""
import numpy as np


def euclidean(a, b):
    """Compute a Euclidean distance value."""
    return np.sqrt(np.sum((a-b)**2))


def early_abandoned_dist(a, b, upper_limit):
    """Compute a Euclidean distance value in early abandoning fashion."""
    lim = upper_limit * upper_limit
    res = 0.
    for i in range(0, len(a)):
        res += (a[i]-b[i])*(a[i]-b[i])
        if res > lim:
            return np.nan
    return np.sqrt(res)

"""Implements HOT-SAX."""
import numpy as np
from saxpy.znorm import znorm
from saxpy.sax import sax_via_window
from saxpy.distance import euclidean


def find_discords_hotsax(series, win_size=100, num_discords=2, a_size=3,
                         paa_size=3, z_threshold=0.01):
    """HOT-SAX-driven discords discovery."""
    discords = list()

    globalRegistry = set()

    while (len(discords) < num_discords):

        bestDiscord = find_best_discord_hotsax(series, win_size, a_size,
                                               paa_size, z_threshold,
                                               globalRegistry)

        if -1 == bestDiscord[0]:
            break

        discords.append(bestDiscord)

        mark_start = bestDiscord[0] - win_size
        if 0 > mark_start:
            mark_start = 0

        mark_end = bestDiscord[0] + win_size
        '''if len(series) < mark_end:
            mark_end = len(series)'''

        for i in range(mark_start, mark_end):
            globalRegistry.add(i)

    return discords


def find_best_discord_hotsax(series, win_size, a_size, paa_size,
                             znorm_threshold, globalRegistry): # noqa: C901
    """Find the best discord with hotsax."""
    """[1.0] get the sax data first"""
    sax_none = sax_via_window(series, win_size, a_size, paa_size, "none", 0.01)

    """[2.0] build the 'magic' array"""
    magic_array = list()
    for k, v in sax_none.items():
        magic_array.append((k, len(v)))

    """[2.1] sort it desc by the key"""
    m_arr = sorted(magic_array, key=lambda tup: tup[1])

    """[3.0] define the key vars"""
    bestSoFarPosition = -1
    bestSoFarDistance = 0.

    distanceCalls = 0

    visit_array = np.zeros(len(series), dtype=np.int)

    """[4.0] and we are off iterating over the magic array entries"""
    for entry in m_arr:

        """[5.0] some moar of teh vars"""
        curr_word = entry[0]
        occurrences = sax_none[curr_word]

        """[6.0] jumping around by the same word occurrences makes it easier to
        nail down the possibly small distance value -- so we can be efficient
        and all that..."""
        for curr_pos in occurrences:

            if curr_pos in globalRegistry:
                continue

            """[7.0] we don't want an overlapping subsequence"""
            mark_start = curr_pos - win_size
            mark_end = curr_pos + win_size
            visit_set = set(range(mark_start, mark_end))

            """[8.0] here is our subsequence in question"""
            cur_seq = znorm(series[curr_pos:(curr_pos + win_size)],
                            znorm_threshold)

            """[9.0] let's see what is NN distance"""
            nn_dist = np.inf
            do_random_search = 1

            """[10.0] ordered by occurrences search first"""
            for next_pos in occurrences:

                """[11.0] skip bad pos"""
                if next_pos in visit_set:
                    continue
                else:
                    visit_set.add(next_pos)

                """[12.0] distance we compute"""
                dist = euclidean(cur_seq, znorm(series[next_pos:(
                                 next_pos+win_size)], znorm_threshold))
                distanceCalls += 1

                """[13.0] keep the books up-to-date"""
                if dist < nn_dist:
                    nn_dist = dist
                if dist < bestSoFarDistance:
                    do_random_search = 0
                    break

            """[13.0] if not broken above,
            we shall proceed with random search"""
            if do_random_search:
                """[14.0] build that random visit order array"""
                curr_idx = 0
                for i in range(0, (len(series) - win_size)):
                    if not(i in visit_set):
                        visit_array[curr_idx] = i
                        curr_idx += 1
                it_order = np.random.permutation(visit_array[0:curr_idx])
                curr_idx -= 1

                """[15.0] and go random"""
                while curr_idx >= 0:
                    rand_pos = it_order[curr_idx]
                    curr_idx -= 1

                    dist = euclidean(cur_seq, znorm(series[rand_pos:(
                                     rand_pos + win_size)], znorm_threshold))
                    distanceCalls += 1

                    """[16.0] keep the books up-to-date again"""
                    if dist < nn_dist:
                        nn_dist = dist
                    if dist < bestSoFarDistance:
                        nn_dist = dist
                        break

            """[17.0] and BIGGER books"""
            if (nn_dist > bestSoFarDistance) and (nn_dist < np.inf):
                bestSoFarDistance = nn_dist
                bestSoFarPosition = curr_pos

    return (bestSoFarPosition, bestSoFarDistance)

"""Implements PAA."""
import numpy as np


def paa(series, paa_segments):
    """PAA implementation."""
    series_len = len(series)

    # check for the trivial case
    if (series_len == paa_segments):
        return np.copy(series)
    else:
        res = np.zeros(paa_segments)
        # check when we are even
        if (series_len % paa_segments == 0):
            inc = series_len // paa_segments
            for i in range(0, series_len):
                idx = i // inc
                np.add.at(res, idx, series[i])
                # res[idx] = res[idx] + series[i]
            return res / inc
        # and process when we are odd
        else:
            for i in range(0, paa_segments * series_len):
                idx = i // series_len
                pos = i // paa_segments
                np.add.at(res, idx, series[pos])
                # res[idx] = res[idx] + series[pos]
            return res / series_len

"""Converts a normlized timeseries to SAX symbols."""
from collections import defaultdict
from saxpy.strfunc import idx2letter
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.alphabet import cuts_for_asize


def ts_to_string(series, cuts):
    """A straightforward num-to-string conversion."""
    a_size = len(cuts)
    sax = list()
    for i in range(0, len(series)):
        num = series[i]
        # if teh number below 0, start from the bottom, or else from the top
        if(num >= 0):
            j = a_size - 1
            while ((j > 0) and (cuts[j] >= num)):
                j = j - 1
            sax.append(idx2letter(j))
        else:
            j = 1
            while (j < a_size and cuts[j] <= num):
                j = j + 1
            sax.append(idx2letter(j-1))
    return ''.join(sax)


def is_mindist_zero(a, b):
    """Check mindist."""
    if len(a) != len(b):
        return 0
    else:
        for i in range(0, len(b)):
            if abs(ord(a[i]) - ord(b[i])) > 1:
                return 0
    return 1


def sax_by_chunking(series, paa_size, alphabet_size=3, z_threshold=0.01):
    """Simple chunking conversion implementation."""
    paa_rep = paa(znorm(series, z_threshold), paa_size)
    cuts = cuts_for_asize(alphabet_size)
    return ts_to_string(paa_rep, cuts)


def sax_via_window(series, win_size, paa_size, alphabet_size=3,
                   nr_strategy='exact', z_threshold=0.01):
    """Simple via window conversion implementation."""
    cuts = cuts_for_asize(alphabet_size)
    sax = defaultdict(list)

    prev_word = ''

    for i in range(0, len(series) - win_size):

        sub_section = series[i:(i+win_size)]

        zn = znorm(sub_section, z_threshold)

        paa_rep = paa(zn, paa_size)

        curr_word = ts_to_string(paa_rep, cuts)

        if '' != prev_word:
            if 'exact' == nr_strategy and prev_word == curr_word:
                continue
            elif 'mindist' == nr_strategy and\
                    is_mindist_zero(prev_word, curr_word):
                continue

        prev_word = curr_word

        sax[curr_word].append(i)

    return sax

"""Implements VSM."""
import math
import numpy as np
from saxpy.sax import sax_via_window


def series_to_wordbag(series, win_size, paa_size, alphabet_size=3,
                      nr_strategy='exact', z_threshold=0.01):
    """VSM implementation."""
    sax = sax_via_window(series, win_size, paa_size, alphabet_size,
                         nr_strategy, z_threshold)

    # convert the dict to a wordbag
    frequencies = {}
    for k, v in sax.items():
        frequencies[k] = len(v)
    return frequencies


def manyseries_to_wordbag(series_npmatrix, win_size, paa_size, alphabet_size=3,
                          nr_strategy='exact', z_threshold=0.01):
    """VSM implementation."""
    frequencies = {}

    for row in series_npmatrix:
        tmp_freq = series_to_wordbag(np.squeeze(np.asarray(row)),
                                     win_size, paa_size, alphabet_size,
                                     nr_strategy, z_threshold)
        for k, v in tmp_freq.items():
            if k in frequencies:
                frequencies[k] += v
            else:
                frequencies[k] = v

    return frequencies


def bags_to_tfidf(bags_dict):
    """VSM implementation."""

    # classes
    count_size = len(bags_dict)
    classes = [*bags_dict.copy()]

    # word occurrence frequency counts
    counts = {}

    # compute frequencies
    idx = 0
    for name in classes:
        for word, count in bags_dict[name].items():
            if word in counts:
                counts[word][idx] = count
            else:
                counts[word] = [0] * count_size
                counts[word][idx] = count
        idx = idx + 1

    # compute tf*idf
    tfidf = {}  # the resulting vectors dictionary
    idx = 0
    for word, freqs in counts.items():

        # document frequency
        df_counter = 0
        for i in freqs:
            if i > 0:
                df_counter = df_counter + 1

        # if the word is everywhere, dismiss it
        if df_counter == len(freqs):
            continue

        # tf*idf vector
        tf_idf = [0.0] * len(freqs)
        i_idx = 0
        for i in freqs:
            if i != 0:
                tf = np.log(1 + i)
                idf = np.log(len(freqs) / df_counter)
                tf_idf[i_idx] = tf * idf
            i_idx = i_idx + 1

        tfidf[word] = tf_idf

        idx = idx + 1

    return {"vectors": tfidf, "classes": classes}


def tfidf_to_vector(tfidf, vector_label):
    """VSM implementation."""
    if vector_label in tfidf['classes']:
        idx = tfidf['classes'].index(vector_label)
        weight_vec = {}
        for word, weights in tfidf['vectors'].items():
            weight_vec[word] = weights[idx]
        return weight_vec
    else:
        return []


def cosine_measure(weight_vec, test_bag):
    """VSM implementation."""
    sumxx, sumxy, sumyy = 0, 0, 0
    for word in set([*weight_vec.copy()]).union([*test_bag.copy()]):
        x, y = 0, 0
        if word in weight_vec.keys():
            x = weight_vec[word]
        if word in test_bag.keys():
            y = test_bag[word]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


def cosine_similarity(tfidf, test_bag):
    """VSM implementation."""
    res = {}
    for cls in tfidf['classes']:
        res[cls] = 1. - cosine_measure(tfidf_to_vector(tfidf, cls), test_bag)

    return res


def class_for_bag(similarity_dict):
    # do i need to take care about equal values?
    return min(similarity_dict, key=lambda x: similarity_dict[x])

"""Convert a normlized timeseries to SAX symbols."""


def idx2letter(idx):
    """Convert a numerical index to a char."""
    if 0 <= idx < 20:
        return chr(97 + idx)
    else:
        raise ValueError('A wrong idx value supplied.')

"""Some helpers."""
import re


def read_ucr_data(fname):
    data = []

    with open(fname, 'r') as fp:
        read_lines = fp.readlines()
        for line in read_lines:
            tokens = re.split("\\s+", line.strip())

            t0 = tokens.pop(0)
            if re.match("^\d+?\.\d+?e[\+\-]?\d+?$", t0) is None:
                class_label = t0
            else:
                class_label = str(int(float(t0)))

            data_line = []
            for token in tokens:
                data_line.append(float(token))

            data.append((class_label, data_line))

    res = {}
    for key, arr in data:
        if key in res.keys():
            res[key].append(arr)
        else:
            dat = [arr]
            res[key] = dat

    return res

"""Keeps visited indexes in check."""
import numpy as np


class VisitRegistry:
    """A straightforward visit array implementation."""

    def __init__(self, capacity):
        """Constructor."""
        self.capacity = capacity
        self.visit_array = np.zeros(capacity, dtype=np.uint8)
        self.unvisited_count = capacity

    def get_unvisited_count(self):
        """An unvisited count getter."""
        return self.unvisited_count

    def mark_visited(self, index):
        """Set a single index as visited."""
        if (0 == self.visit_array[index]):
            self.visit_array[index] = 1
            self.unvisited_count -= 1

    def mark_visited_range(self, start, stop):
        """Set a range as visited."""
        for i in range(start, stop):
            self.mark_visited(i)

    def get_next_unvisited(self):
        """Memory-optimized version."""
        if 0 == self.unvisited_count:
            return np.nan
        else:
            i = np.random.randint(0, self.capacity)
            while 1 == self.visit_array[i]:
                i = np.random.randint(0, self.capacity)
            return i

    def clone(self):
        """Make the array's copy."""
        clone = VisitRegistry(self.capacity)
        setattr(clone, 'visit_array', self.visit_array.copy())
        setattr(clone, 'unvisited_count', self.unvisited_count)
        return clone

    '''def get_next_unvisited2(self):
        """Speed-optimized version."""
        if 0 == self.unvisited_count:
            return np.nan
        else:
            tmp_order = np.zeros(self.unvisited_count, dtype=np.int32)
            j = 0
            for i in range(0, self.capacity):
                if 0 == self.visit_array[i]:
                    tmp_order[j] = i
                    j += 1
            np.random.shuffle(tmp_order)
            return tmp_order[0]'''

"""Implements znorm."""
import numpy as np


def znorm(series, znorm_threshold=0.01):
    """Znorm implementation."""
    sd = np.std(series)
    if (sd < znorm_threshold):
        return series
    mean = np.mean(series)
    return (series - mean) / sd

path = "G:/Coding/ML/UCRArchive_2018/"
dataset = 'UMD'
file_train = path + str(dataset) + "/" + str(dataset) + "_TRAIN.tsv"
file_test = path + str(dataset) + "/" + str(dataset) + "_TEST.tsv"

dd = read_ucr_data(file_train)

win = 30
paa = 6
alp = 6
na_strategy = "exact"
ztresh = 0.01

bags = {}

for key, arr in dd.items():
    print(key)
    bags[key] = manyseries_to_wordbag(dd[key], win, paa, alp, na_strategy, ztresh)

[*bags.copy()]
vectors = bags_to_tfidf(bags)
vectors['classes']

dt = read_ucr_data(file_test)

test_bag = series_to_wordbag(series, 30, 6, 6, "exact", 0.01)
res = cosine_similarity(vectors, test_bag)
class_for_bag(res)

for cls in [*dt.copy()]:
    print(cls)
    i = 0
    for s in dt[cls]:
        sim = cosine_similarity(vectors, 
                                series_to_wordbag(s, 30, 6, 6, "none", 0.01))
        res = class_for_bag(sim)
        if res != cls:
            print(" misclassified", i, "as", res, sim)
        i = i + 1

