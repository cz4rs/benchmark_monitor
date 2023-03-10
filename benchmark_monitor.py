#!/usr/bin/env python

import json
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from textwrap import fill

import mpld3
import numpy as np
from jinja2 import Environment, FileSystemLoader
from matplotlib import pyplot as plt
from mpld3 import plugins
from scipy import stats
from scipy.stats import mannwhitneyu


def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def parse_arguments():
    print("args = " + str(sys.argv))

    parser = ArgumentParser(
        description="Generates a chart for each google benchmark across a benchmark history with optional step change detection."
    )
    parser.add_argument(
        "-d",
        "--directory",
        help="Directory containing benchmark result json files to process",
        default=os.getcwd(),
    )
    parser.add_argument(
        "-w",
        "--slidingwindow",
        help="The size of the benchmark comparison sliding window",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-s",
        "--maxsamples",
        help="The maximum number of benchmarks (including slidingwindow) to run analysis on (0 == all builds)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-f",
        "--medianfilter",
        help="The median filter kernel size i.e. the number of points around each data value to smooth accross in order to eliminate temporary peaks and troughs in benchmark performance",
        type=int,
        default=9,
    )
    parser.add_argument(
        "-a",
        "--alphavalue",
        help="The alpha value at which we reject the hypothesis that the sliding window of benchmarks equals the benchmark history. Typical value is around 0.05 to 0.01. The noisier the environment the lower this value should be.",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "-c",
        "--controlbenchmarkname",
        help="The control benchmark name (not yet implemented)",
    )
    parser.add_argument(
        "-x",
        "--discard",
        help="(DEBUG) The number of (most recent) records to ignore. This is useful when wanting to debug scenarios in a sub region of the history",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "-sx",
        "--startindex",
        help="(DEBUG - Alternative addressing scheme) The index to start the analysis at",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "-ex",
        "--endindex",
        help="(DEBUG - Alternative addressing scheme) The index to end the analysis at",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "-m",
        "--metric",
        help="The benchmark metric to track",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--outputdirectory",
        help="The index.html report output directory",
        default=os.getcwd(),
    )
    parser.add_argument(
        "-sc",
        "--detectstepchanges",
        help="Detect step changes",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    ensure_dir(args.outputdirectory)
    return args


def parse_benchmark_file(file, benchmarks, metric, git_hashes, git_descriptions):
    """Parse a single benchmark file
    @param: benchmarks dictionary of lists where the key is the benchmark name
            and the value is a list of values recorded for that benchmark/metric
            accross all files
    """
    # print("parsing " + file)

    with open(file, "r", encoding="utf8") as json_file:
        data = json.load(json_file)
        git_hashes.append(data["context"]["GIT_COMMIT_HASH"])
        git_descriptions.append(data["context"]["GIT_COMMIT_DESCRIPTION"])

        for b in data["benchmarks"]:
            if metric is None:
                for key in b:
                    if key.startswith("FOM"):
                        metric = key
                        break
            # if there's no metric marked as a figure of merit, use real time
            if metric is None:
                metric = "real_time"

            # print("\t" + b["name"] + "." + metric + " = " + str(b[metric]))
            if benchmarks.get(b["name"]) is None:
                benchmarks[b["name"]] = [b[metric]]
            else:
                benchmarks[b["name"]].append(b[metric])


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def turningpoints(x):
    peaks = []
    troughs = []
    for i in range(1, len(x) - 1):
        if x[i - 1] < x[i] and x[i + 1] < x[i]:
            peaks.append(i)
        elif x[i - 1] > x[i] and x[i + 1] > x[i]:
            troughs.append(i)
    return peaks, troughs


def estimate_step_location(values):
    # https://stackoverflow.com/questions/48000663/step-detection-in-one-dimensional-data/48001937)
    dary = np.array(values)
    avg = np.average(dary)
    dary -= avg
    step = np.hstack((np.ones(len(dary)), -1 * np.ones(len(dary))))
    dary_step = np.convolve(dary, step, mode="valid")
    print(np.argmax(dary_step))

    # get location of step change
    peaks, _ = turningpoints(dary_step)
    if (len(peaks)) == 0:
        return 0

    step_max_idx = peaks[-1]
    return step_max_idx


def has_slowed_down(benchmark, raw_values, smoothedvalues, slidingwindow, alphavalue):
    sample_count = len(raw_values)
    sample_a_len = sample_count - slidingwindow

    # mw test
    sample_a = smoothedvalues[:sample_a_len]
    sample_b = smoothedvalues[sample_a_len:]
    print(
        "len(sample_a) = "
        + str(len(sample_a))
        + " len(sample_b) = "
        + str(len(sample_b))
    )
    stat, p = mannwhitneyu(sample_a, sample_b)
    print(f"BENCHMARK {benchmark} STATS={stat:.3f}, p={p:.3f}")
    if p < alphavalue:
        print("\tStep change possibly found, performing t-test...")

        # confirm with Welch's t-test as mw can reject if sd is big
        # https://thestatsgeek.com/2014/04/12/is-the-wilcoxon-mann-whitney-test-a-good-non-parametric-alternative-to-the-t-test/
        stat, p = stats.ttest_ind(sample_a, sample_b, equal_var=False)
        if p < alphavalue:
            return True
        print("\tStep change doesnt appear to be part of a trend")
    return False


def smooth(x, window_len):
    # references: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    w = np.hanning(window_len)
    y = np.convolve(w / w.sum(), s, mode="valid")

    return y[0 : len(x)]


def get_json_files(directory, args):
    files = []
    for entry in os.scandir(directory):
        if entry.path.endswith(".json") and entry.is_file():
            files.append(entry)

    if len(files) == 0:
        print("no benchmark data")
        exit()

    # sort files in order of creation time (oldest to newest)
    # FIXME: use date/time from the JSON file (context / date)
    # for now take the date/time from the file's name
    files.sort(key=lambda file: file.name[-25:-5])

    # check if the user is addressing a subset of records using the range addressing scheme
    # (startindex to endindex)
    if args.startindex != -1 and args.endindex != -1:
        files = files[args.startindex : args.endindex]
    else:
        # discard all records after endindex (if set)
        if args.discard != -1:
            files = files[: len(files) - args.discard]

        # limit the number of test samples
        if args.maxsamples != 0:
            file_count = len(files)
            maxsamples = clamp(args.maxsamples, 0, file_count)
            files = files[file_count - maxsamples - 1 : file_count - 1]

    return files


def get_benchmarks(files, metric):
    benchmarks = {}
    git_hashes = []
    git_descriptions = []

    for entry in files:
        if entry.path.endswith(".json") and entry.is_file():
            try:
                parse_benchmark_file(
                    entry.path, benchmarks, metric, git_hashes, git_descriptions
                )
            except Exception as e:
                print(
                    f"Skipping corrupt benchmark file:\n"
                    f"\t - file: {entry.path}\n"
                    f"\t - exception: {e}"
                )
    return benchmarks, git_hashes, git_descriptions


def analyse_benchmark(
    benchmark,
    metric,
    raw_values,
    git_hashes,
    git_descriptions,
    args,
    plots,
    output_subdir,
    dir_name,
):
    # check we have enough records for this benchmark (if not then skip it)
    sample_count = len(raw_values)
    print(f"Found {str(sample_count)} benchmark records for benchmark {benchmark}")

    if sample_count < 10 + args.slidingwindow:
        print("\t - benchmark needs more data, skipping step change detection.")
        args.detectstepchanges = False

    # plot raw and smoothed values
    fig, ax = plt.subplots()
    raw_points = ax.plot(
        raw_values,
        color="green",
        marker="o",
        markersize=10,
        linestyle="dashed",
        label="raw",
    )
    ax.set_ylabel(metric)
    ax.set_xlabel("sample #")

    if len(raw_values) >= args.medianfilter and args.medianfilter > 2:
        # apply a median filter to smooth out temporary spikes
        smoothed_values = smooth(np.array(raw_values), args.medianfilter)
        ax.plot(smoothed_values, "-b", label="smoothed")

        # plot line fit
        x_vals = np.arange(0, len(raw_values), 1)
        y_vals = raw_values
        model = np.polyfit(x_vals, y_vals, 1)
        predict = np.poly1d(model)
        lrx = range(0, len(x_vals))
        lry = predict(lrx)
        ax.plot(lrx, lry, "tab:orange", label="linear regression")
    else:
        print(
            "\t - benchmark needs more data, "
            "skipping smoothing and linear regression."
        )

    # has it slowed down?
    if args.detectstepchanges and has_slowed_down(
        benchmark,
        raw_values,
        smoothed_values,
        args.slidingwindow,
        args.alphavalue,
    ):
        # estimate step location
        step_max_idx = estimate_step_location(smoothed_values)
        if step_max_idx > 0 and step_max_idx < sample_count:
            print("step_max_idx = " + str(step_max_idx))
            if smoothed_values[step_max_idx + 1] > smoothed_values[step_max_idx - 1]:
                print(
                    "\tBENCHMARK "
                    + benchmark
                    + " STEP CHANGE IN PERFORMANCE ENCOUNTERED (SLOWDOWN) - likely occurred somewhere between this build and this build minus "
                    + str(sample_count - step_max_idx)
                    + "]"
                )

                # plot step location
                plt.plot(
                    (step_max_idx, step_max_idx),
                    (np.min(raw_values), np.max(raw_values)),
                    "r",
                    label="slowdown location estimation",
                )
            else:
                print(
                    "\tBENCHMARK "
                    + benchmark
                    + " STEP CHANGE IN PERFORMANCE ENCOUNTERED (SPEEDUP) - ignoring"
                )
        else:
            print(
                "\tBENCHMARK "
                + benchmark
                + " step index is 0 - likely speedup, ignoring"
            )

    plt.title(fill(benchmark, 50))
    plt.legend(loc="upper left")
    fig.tight_layout()

    labels = [
        f"<p><b>#{git_hashes[i]}</b></br>{fill(git_descriptions[i], 50)}</p>"
        for i in range(sample_count)
    ]
    targets = [
        f"https://github.com/kokkos/kokkos/commit/{git_hashes[i]}"
        for i in range(sample_count)
    ]
    tooltip = plugins.PointHTMLTooltip(raw_points[0], labels, targets)
    plugins.connect(fig, tooltip)

    namename = os.path.join(
        output_subdir, remove_special_characters(benchmark) + ".html"
    )
    mpld3.save_html(fig, namename)
    plot_item = dict(
        benchmark=benchmark,
        path=os.path.join(
            remove_special_characters(dir_name),
            remove_special_characters(benchmark) + ".html",
        ),
    )
    plots.append(plot_item)
    plots = sorted(plots, key=lambda d: d["benchmark"])
    plt.close(fig)


def parse_directory(dir_name, args, env):
    print(f"Parsing directory {dir_name}")
    output_subdir = os.path.join(
        args.outputdirectory, remove_special_characters(dir_name)
    )
    ensure_dir(output_subdir)

    # get list of files to parse
    files = get_json_files(os.path.join(args.directory, dir_name), args)

    plots = []
    benchmarks, git_hashes, git_descriptions = get_benchmarks(files, args.metric)

    # analyse benchmarks
    for benchmark, raw_values in benchmarks.items():
        analyse_benchmark(
            benchmark,
            args.metric,
            raw_values,
            git_hashes,
            git_descriptions,
            args,
            plots,
            output_subdir,
            dir_name,
        )

    # generate report
    template = env.get_template("benchmarks.html")
    output_file_path = os.path.join(
        args.outputdirectory, remove_special_characters(dir_name) + ".html"
    )
    with open(output_file_path, "w", encoding="utf8") as file:
        file.write(template.render(plots=plots))


def remove_special_characters(s):
    return re.sub(r"[^\w_. -]", "_", s)


def parse_subdirectories(args, env):
    subdirectories = []
    for entry in os.scandir(args.directory):
        if entry.is_dir() and not entry.name.startswith("."):
            parse_directory(entry.name, args, env)
            subdirectories.append(entry.name)
    return subdirectories


def get_jinja_environment():
    env = Environment(
        loader=FileSystemLoader(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "templates")
        ),
        autoescape=False,
    )
    env.globals["remove_special_characters"] = remove_special_characters
    return env


def create_index_file(output_dir, subdirectories, env):
    template = env.get_template("index.html")
    index_file_path = os.path.join(output_dir, "index.html")
    with open(index_file_path, "w", encoding="utf8") as file:
        file.write(template.render(dirs=subdirectories))


def main():
    args = parse_arguments()
    env = get_jinja_environment()
    sub_dirs = parse_subdirectories(args, env)
    create_index_file(args.outputdirectory, sub_dirs, env)


if __name__ == "__main__":
    main()
