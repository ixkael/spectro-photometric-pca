# -*- coding: utf-8 -*-

import time


def process_time(start_time, end_time, multiply=1, no_hours=False):
    dt = (end_time - start_time) * multiply
    if no_hours:
        hour = 0
    else:
        hour = int(dt / 3600)
    min = int((dt - 3600 * hour) / 60)
    sec = int(dt - 3600 * hour - 60 * min)
    if no_hours:
        return (min, sec)
    else:
        return (hour, min, sec)


def print_elapsed_time(start_time, end="\n"):
    print(
        "> Elapsed time: %dh %dm %ds" % process_time(start_time, time.time()),
        end=end,
    )


def print_remaining_time(start_time, i_start, i, n_epochs, end="\n"):
    print(
        "> Remaining time: %dh %dm %ds"
        % process_time(
            start_time,
            time.time(),
            multiply=(n_epochs - i) / float(i + 1 - i_start),
        ),
        end=end,
    )
