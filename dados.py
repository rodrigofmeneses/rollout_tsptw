from statistics import mean

def chunks(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def percent_distance(best: list[float], obtained: list[int]):
    """
    Calculate percentual diference betwen values of 2 lists,
    utilizing the relation |mc - rollout| / mc 

    best: best results know
    obtained: results obtained by rollout algorithm

    return: list of percentual distances
    """
    return [
        abs((mc - rollout) / mc) for mc, rollout in zip(best, obtained)
        if rollout != 0.
    ]

def values_to_plot(mc, rollout):
    '''Values to plot a bar graph from mc and rollout values'''
    mc_chunks = [chunk for chunk in chunks(mc, 5)]
    rollout_chunks = [chunk for chunk in chunks(rollout, 5)]

    percent_distances = [
        percent_distance(mc, rollout)
        for mc, rollout in zip(mc_chunks, rollout_chunks)
    ]

    return [
        mean(chunk) * 100 if len(chunk) else 0 
        for chunk in percent_distances
    ]



MC_DUMAS_20 = [
    378, 286, 394, 396, 352, 254, 333, 317, 388, \
    288, 335, 244, 352, 280, 338, 329, 338, 320, 304, 264]

ROLLOUT_DUMAS_20 = [
    382, 286, 394, 396, 355, 258, 343, 317, 391, \
    305, 336, 248, 260, 300, 366, 330, 352, 330, 309, 296]

MC_DUMAS_40 = [
    500, 552, 478, 404, 499, 465, 461, 474, 452, 453, \
    494, 470, 408, 382, 328, 395, 431, 412, 417, 344
]
ROLLOUT_DUMAS_40 = [
    504, 0, 489, 415, 0, 0, 0, 486, 0, 482, 504, \
    523, 0, 0, 420, 0, 455, 434, 0, 394
]

MC_DUMAS_60 = [
    551, 605, 533, 616, 603, 591, 621, 603, 597, 539, 609, 566, \
        548, 571, 569, 458, 498, 550, 566, 468
]

ROLLOUT_DUMAS_60 = [ 
    0, 642, 562, 0, 622, 620, 0, 0, 0, \
    578, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]

MC_GENDREAU_20 = [
    267, 218, 303, 300, 240, 176, 272, 236, 255, 225, 241, \
    201, 201, 203, 245, 253, 265, 271, 201, 193, 233, 203, 249, 293, 227
]

ROLLOUT_GENDREAU_20 = [
    309, 245, 348, 326, 331, 224, 305, 268, 283, 228, 302, \
        254, 284, 228, 330, 339, 319, 315, 284, 218, 266, 235, 264, 332, 240
]

MC_GENDREAU_40 = [ 
    265.6, 265.6, 265.6, 265.6, 265.6, 232.8, 232.8, 232.8, 232.8, 232.8, 232.8, \
        348, 337, 346, 288, 315, 236.6, 236.6, 236.6, 236.6, 236.6, 330, 303, 339, 301, 300
]

ROLLOUT_GENDREAU_40 = [
    444, 544, 0, 0, 439, 0, 426, 0, 446, 452, 425, 470, 393, 0, 354, 0, \
    427, 382, 430, 366, 0, 319, 402, 412, 377,
]

MC_GENDREAU_60 = [
    384, 427, 407, 490, 547, 423, 462, 427, 488, 460, 448.6, 448.6, 448.6, 448.6, \
    448.6, 411, 399, 445, 456, 395, 410, 414, 455, 431, 427
]

ROLLOUT_GENDREAU_60 = [0, 589, 541, 623, 0, 587, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    546, 0, 0, 0, 0, 0, 0, 579, 632, 555
]

dumas_20 = values_to_plot(MC_DUMAS_20, ROLLOUT_DUMAS_20)
dumas_40 = values_to_plot(MC_DUMAS_40, ROLLOUT_DUMAS_40)
dumas_60 = values_to_plot(MC_DUMAS_60, ROLLOUT_DUMAS_60)
gendreau_20 = values_to_plot(MC_GENDREAU_20, ROLLOUT_GENDREAU_20)
gendreau_40 = values_to_plot(MC_GENDREAU_40, ROLLOUT_GENDREAU_40)
gendreau_60 = values_to_plot(MC_GENDREAU_60, ROLLOUT_GENDREAU_60)