#!/usr/bin/env python3
"""
Needed tools:
- mediainfo
- x264_encoder with ffmpegsource support
- ffmpegsource
- mkvtoolnix (mkvmerge)
"""

import bisect
import subprocess
import os
import re
import tempfile
import contextlib


def cut_file_by_cutlist(f_in, f_out,
                        cutlist_frm,
                        cutlist_t=None,
                        workingdir=None,
                        x264_profile="HQ"):
    """ Main routine for cutting """
    with workingdir_context_manager(workingdir)() as _workingdir:
        f_info = get_encoder_infos(f_in)
        if cutlist_t is None:
            cutlist_t = cutlist_time_from_frames(
                cutlist_frm, f_info.get("framerate", 25.0))

        audio_file = os.path.join(_workingdir, "audio_copy.mkv")
        cut_audio(
            f_in, audio_file, cutlist_t)
        keyframes = load_keyframes_from_file(f_in)
        segments = []
        for start, end in cutlist_frm:
            segments += segment(start, end, keyframes)
        video_parts = process_segments(
            segments, f_in, f_info, x264_profiles[x264_profile],
            _workingdir)
        merge_parts(video_parts, audio_file, f_out)
        cleanup(video_parts + [audio_file])


def workingdir_context_manager(workingdir):
    """ TemporaryDirectory if workingdir is None else trivial contextmanager """
    if workingdir is None:
        return tempfile.TemporaryDirectory
    return contextlib.contextmanager(lambda: (yield workingdir))


def cleanup(files):
    for f in files:
        if os.path.isfile(f):
            os.remove(f)


class Segment:

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop


class Copy(Segment):
    pass


class Recode(Segment):
    pass


def cmd_exec(cmd, ok_return_codes=(0,)):
    cmd_str = " ".join(cmd)
    print(cmd_str)
    try:
        result = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True)
        result.wait()
    except FileNotFoundError:
        raise FileNotFoundError("Could not call {}".format(cmd_str))
    if result.returncode not in ok_return_codes:
        raise RuntimeError("stdout: \n".join(result.stdout) +
                           "\nstderr:" + "\n".join(result.stderr))
    return result.stdout


def load_keyframes_from_file(filename):
    """ returns keyframe list - in frame numbers"""

    if not os.path.isfile(filename + ".ffindex_track00.kf.txt"):
        cmd_exec(("ffmsindex", "-p", "-f", "-k", filename))
        if os.path.isfile(filename + ".ffindex"):
            os.remove(filename + ".ffindex")

    filename_keyframes = None
    for i in range(3):
        f = filename + ".ffindex_track0{}.kf.txt".format(i)
        if os.path.isfile(f):
            filename_keyframes = f

    try:
        with open(filename_keyframes, 'r') as index:
            index.readline()
            index.readline()
            try:
                keyframes = tuple(map(int, index.read().splitlines()))
            except ValueError:
                raise ValueError("Could not determinew Keyframes.")
    except IOError:
        raise IOError(
            "Could not open keyframe file {}.".format(filename_keyframes))

    return keyframes


def segment(start, end, keyframes):
    if start in keyframes:
        if end in keyframes or end > keyframes[-1]:
            if end <= start:
                return []
            # copy from keyframe to keyframe
            return [Copy(start, end)]
        else:
            # copy to keyframe before end
            lt_kf_before_end = get_keyframe_in_front_of_frame(
                keyframes, end)
            if lt_kf_before_end is None:
                raise ValueError("No keyframe in front of {}!".format(end))
            if lt_kf_before_end <= start:
                return [Recode(start, end)]
            return [Copy(start, lt_kf_before_end),
                    Recode(lt_kf_before_end, end)]
    else:
        nt_kf_from_start = get_keyframe_after_frame(keyframes, start)
        if nt_kf_from_start is None:
            raise ValueError(
                "No keyframe after {} (last keyframe: {})!".format(
                    start, keyframes[-1]))
        segments = [Recode(start, nt_kf_from_start)]
        if end <= nt_kf_from_start:
            return segments
        return segments + segment(nt_kf_from_start, end, keyframes)


def get_keyframe_in_front_of_frame(keyframes, frame):
    """ Find keyframe less-than to frame. """
    i = bisect.bisect_left(keyframes, frame)
    if i:
        return keyframes[i - 1]
    return None


def get_keyframe_after_frame(keyframes, frame):
    """ Find keyframe greater-than to frame. """
    i = bisect.bisect_right(keyframes, frame)
    if i != len(keyframes):
        return keyframes[i]
    return None


def cut_audio(f_in, f_out, cutlist_t):
    audio_timecodes = ",+".join((get_timecode(start) + "-" + get_timecode(
        stop) for start, stop in cutlist_t))
    audio_timecodes = audio_timecodes.lstrip(",+")
    cmd_exec(("mkvmerge", "-D", "--split",
              "parts:" + audio_timecodes,
              "-o", f_out, f_in))


def get_timecode(time):
    """
    Converts the seconds into a timecode-format that mkvmerge
    understands
    """
    minute, second = divmod(int(time), 60)		# discards milliseconds
    hour, minute = divmod(minute, 60)
    second = time - minute * 60 - hour * 3600  # for the milliseconds
    return "%02i:%02i:%f" % (hour, minute, second)


def encode(f_in, f_out, slc, encoder_info, x264_profile, workingdir):
    cmd_exec(
        ("x264",) + get_x264_opts(encoder_info, x264_profile)
        + ("--demuxer", "ffms",
           "--index", os.path.join(workingdir, "x264.index"),
           "--seek", str(slc.start),
           "--frames", str(slc.stop - slc.start),
           "--output", f_out, f_in)
    )


x264_profiles = {
    "HD": ("--tune", "film",
           "--direct", "auto",
           "--force-cfr",
           "--rc-lookahead", "60",
           "--b-adapt", "2",
           "--weightp", "0"),
    "HQ": ("--tune", "film",
           "--direct", "auto",
           "--force-cfr",
           "--rc-lookahead", "60",
           "--b-adapt", "2",
           "--aq-mode", "2",
           "--weightp", "0")
}


def get_x264_opts(encoder_infos, profile):
    cp = encoder_infos["color_primaries"].split()[0].replace("BT.", "bt")
    if cp in {"bt709", "bt470m", "bt470bg", "bt2020"}:
        cp_opts = ("--videoformat", "pal", "--colorprim", cp,
                   "--transfer", cp, "--colormatrix", cp)
    else:
        cp_opts = ()
    return profile + cp_opts + (
        "--profile", encoder_infos[
            "format_profile_profile"].lower(),
        "--level", str(encoder_infos["format_profile_level"]),
        "--fps", str(encoder_infos["framerate"]))


class Tgt:

    def __init__(self, name, tp=(lambda x: x)):
        self.name = name
        self.tp = tp


def get_encoder_infos(filename):
    output = cmd_exec(("mediainfo", filename))
    return parse(
        output,
        {
            "Writing library *: *x264 core ([0-9]*)": Tgt("x264_core"),
            "Color primaries *: (.*[^ ]) *$": Tgt("color_primaries"),
            "Format profile *: ([^@]*)@L([0-9]*) *$":
            (Tgt("format_profile_profile"),
             Tgt("format_profile_level", float)),
            "Frame rate *: ([0-9.]*) FPS$": Tgt("framerate", float)
        })


def parse(s, mapping):
    re_map = re_mapping(mapping)
    out = {}
    for line in s:
        for pattern, targets in re_map.items():
            match = pattern.match(line)
            if not match:
                continue
            for tgt, grp in zip(targets, match.groups()):
                val = tgt.tp(grp)
                if tgt.name not in out:
                    out[tgt.name] = val
                else:
                    entry = out[tgt.name]
                    if not isinstance(entry, list):
                        out[tgt.name] = [entry]
                    out[tgt.name].append(val)
            break

    return out


def re_mapping(mapping):
    return {re.compile(pattern): tuplify(key)
            for pattern, key in mapping.items()}


def process_segments(segments, f_in, f_info, x264_profile, outdir):
    video_files = []
    copy_file_indices = []
    copy_segments = []
    i_copy = 1
    i_encode = 1
    copy_tpl = os.path.join(outdir, "segment_copy{}.mkv")
    for seg in segments:
        if isinstance(seg, Copy):
            copy_file_indices.append(len(video_files))
            video_files.append(copy_tpl.format("-{:03d}".format(i_copy)))
            copy_segments.append(seg)
            i_copy += 1
        elif isinstance(seg, Recode):
            f_out = os.path.join(outdir, "segment-{:03d}.mkv".format(i_encode))
            video_files.append(f_out)
            encode(f_in, f_out, seg, f_info, x264_profile, outdir)
            i_encode += 1
    # If there is only one copy segment, mkvmerge will not add a number suffix
    if len(copy_segments) == 1:
        video_files[copy_file_indices[0]] = copy_tpl.format("")
    split_video(f_in, copy_tpl.format(""), copy_segments)
    return video_files


def split_video(f_in, f_out, copy_segments):
    copy_segments = ((str(seg.start + 1) + "-" + str(seg.stop + 1))
                     for seg in copy_segments)
    cmd_exec(("mkvmerge", "-A",
              "--split", "parts-frames:" + ",".join(copy_segments),
              "-o", f_out, f_in))


def merge_parts(video_files, audio_file, f_out):
    cmd_exec(("mkvmerge", "--engage", "no_cue_duration", "--engage",
              "no_cue_relative_position",
              "-o", f_out, video_files[0])
             + tuple('+' + v for v in video_files[1:]) + (audio_file,),
             ok_return_codes=(0, 1))


def load_cutlist(stream):
    mapping = {
        r"StartFrame ?= ?(.*)$": Tgt("start", int),
        r"DurationFrames ?= ?(.*)$": Tgt("duration", int),
        r"Start ?= ?(.*)$": Tgt("t_start", float),
        r"Duration ?= ?(.*)$": Tgt("t_duration", float),
        r"ApplyToFile ?= ?(.*)$": Tgt("in_file")
    }
    out = parse(stream, mapping)

    cutlist_duration = tuple(zip(listify(out.pop("start", [])),
                                 listify(out.pop("duration", []))))
    t_cutlist_duration = tuple(zip(listify(out.pop("t_start", [])),
                                   listify(out.pop("t_duration", []))))
    out["cutlist_frames"] = tuple((start, start + duration)
                                  for start, duration in cutlist_duration)
    out["cutlist_time"] = tuple((t_start, t_start + t_duration)
                                for t_start, t_duration in t_cutlist_duration)
    return out


def load_cutlist_avidemux(stream):
    mapping = {
        r"adm.addSegment\(0, ([0-9]*), ([0-9]*)\);?":
        (Tgt("start", lambda s: int(s) // 40000 - 6),
         Tgt("duration", lambda s: int(s) // 40000)),
        r'adm.loadVideo\("([^"]*)"\);?': Tgt("in_file")
    }
    out = parse(stream, mapping)
    out["cutlist_frames"] = tuple((max(0, start), (start + duration))
                                  for start, duration in
                                  zip(listify(out.pop("start", [])),
                                      listify(out.pop("duration", []))))
    return out


def cutlist_time_from_frames(cutlist_frm, fps):
    return tuple((start / fps, stop / fps)
                 for start, stop in cutlist_frm)


cutlist_type_catalogue = {
    "//AD  <- Needed to identify //": load_cutlist_avidemux,
    "#PY  <- Needed to identify #": load_cutlist_avidemux,
    "[General]": load_cutlist,
    None: load_cutlist
}


def determine_cut_list_type(stream):
    line = next(stream).strip("\n")
    return cutlist_type_catalogue.get(line, cutlist_type_catalogue[None])


def tuplify(key):
    if not isinstance(key, tuple):
        return (key, )
    return key


def listify(key):
    if not isinstance(key, list):
        return [key]
    return key


def _main(cutlist_file_name, video_file_name, out_file_name, **kwargs):
    with open(cutlist_file_name, 'r') as stream:
        loader = determine_cut_list_type(stream)
        cutlist_info = loader(stream)
    f_in = video_file_name or cutlist_info.get("in_file")
    if f_in is None:
        raise ValueError("Input video file not specified!")
    f_out = out_file_name or cutlist_info.get("out_file") \
        or cutlist_file_name.rsplit(".", 1)[0] + ".mkv"
    if os.path.isfile(f_out) and os.path.samefile(f_in, f_out):
        f_out = f_out.rsplit(".", 1)[0] + ".cut.mkv"

    cut_file_by_cutlist(f_in,
                        f_out,
                        cutlist_info["cutlist_frames"],
                        cutlist_t=cutlist_info.get("cutlist_time", None),
                        **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="smartcut")
    parser.add_argument("-o", dest="out", default=None,
                        help="Output file (default: <input file>.cut )")
    parser.add_argument("--profile", default="HQ",
                        help="x264 profile HQ|HD (default: HQ)")
    parser.add_argument("--workingdir", default=None,
                        help="Working directory (default: temporary)")
    parser.add_argument("cutlist", help="Cutlist")
    parser.add_argument(
        "-i", default=None, help="Input video file (must only be given if"
        " not specified in the cutlist file)")

    cmd_args = parser.parse_args()

    _main(cmd_args.cutlist, cmd_args.i, cmd_args.out,
          workingdir=cmd_args.workingdir,
          x264_profile=cmd_args.profile)
