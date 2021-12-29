#!/usr/bin/env python3
"""
Needed tools:
- mediainfo
- x264_encoder with ffmpegsource support
- ffmpegsource
- ffmpeg
- mkvtoolnix (mkvmerge)
"""

import bisect
import subprocess
import os
import sys
import re
import tempfile
import contextlib
from itertools import count

from collections.abc import Sequence
from typing import Optional

CutList = Sequence[tuple[int, int]]


def smart_cut_file_by_cutlist(
    f_in,
    f_out,
    cutlist_frm: CutList,
    workingdir: Optional[str] = None,
    segmented: bool = False,
    x264_profile: str = "HQ",
):
    """Cut a video file, reencoding GOP's containing cuts."""
    f_info = get_encoder_infos(f_in)
    fps = f_info.get("framerate", 25.0)
    with workingdir_context_manager(workingdir)() as _workingdir:
        audio_file = os.path.join(_workingdir, "audio_copy.mkv")
        cut_audio(f_in, audio_file, cutlist_frm, fps)
        keyframes = load_keyframes_from_file(f_in)
        segments: list[Segment] = sum(
            (make_segments(start, end, keyframes) for start, end in cutlist_frm), []
        )
        video_parts = process_segments(
            segments, f_in, f_info, x264_profiles[x264_profile], _workingdir
        )
        if not segmented:
            merge_parts(video_parts, audio_file, f_out)
            cleanup(video_parts + [audio_file])


def smart_cut_file_by_cutlist_hd(
    f_in,
    f_out,
    cutlist_frm: CutList,
    workingdir: Optional[str] = None,
    segmented: bool = False,
):
    smart_cut_file_by_cutlist(f_in, f_out, cutlist_frm, workingdir, segmented)


def cut_file_by_cutlist_reencode(
    f_in,
    f_out,
    cutlist_frm: CutList,
    workingdir: Optional[str] = None,
    segmented: bool = False,
):
    """Cut a video file, reencoding everything with ffmpeg."""

    fps = get_fps(f_in)
    if segmented and workingdir:
        f_out = os.path.join(workingdir, os.path.basename(f_out))
    if segmented:
        length = 20
        cutlist_frm = sum(
            (
                [(start, start + length), (max(0, stop - length), stop)]
                for start, stop in cutlist_frm
            ),
            [],
        )
    segments = tuple(f_out + f".{n}.mkv" for n in range(len(cutlist_frm)))
    cmd = ("ffmpeg", "-y", "-i", f_in) + sum(
        (
            (
                "-map",
                "0",
                "-c:s",
                "copy",
                "-ss",
                to_seconds(start, fps),
                "-to",
                to_seconds(end, fps),
                segment,
            )
            for segment, (start, end) in zip(segments, cutlist_frm)
        ),
        (),
    )
    cmd_exec(cmd)
    if segmented:
        return
    cmd = (
        ("mkvmerge", "-o", f_out)
        + segments[:1]
        + tuple("+" + segment for segment in segments[1:])
    )
    print(" ".join(cmd))
    cmd_exec(cmd)
    for segment in segments:
        os.unlink(segment)


def cut_file_by_cutlist(
    f_in,
    f_out,
    cutlist_frm: CutList,
    workingdir: Optional[str] = None,
    segmented: bool = False,
):
    """Only cut at I frames."""
    fps = get_fps(f_in)
    mkvmerge_segments = to_mkvmerge_time_segments(
        cutlist_frm, fps=fps, separator="," if segmented else ",+"
    )
    if segmented and workingdir:
        f_out = os.path.join(workingdir, os.path.basename(f_out))
    cmd = (
        "mkvmerge",
        "--split",
        "parts:" + mkvmerge_segments,
        "-o",
        f_out,
        f_in,
    )
    print(" ".join(cmd))
    cmd_exec(cmd)


def workingdir_context_manager(workingdir: Optional[str]):
    """TemporaryDirectory if workingdir is None else trivial contextmanager"""
    if workingdir is None:
        return tempfile.TemporaryDirectory
    return lambda: contextlib.nullcontext(workingdir)


class Segment:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop


class Copy(Segment):
    pass


class Recode(Segment):
    pass


def cmd_exec(cmd, ok_return_codes=(0,)) -> subprocess.CompletedProcess:
    cmd_str = " ".join(cmd)
    print(cmd_str)
    try:
        process = subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        if e.returncode not in ok_return_codes:
            raise e
    return process


def load_keyframes_from_file(filename):
    """returns keyframe list - in frame numbers"""

    if not os.path.isfile(filename + ".ffindex_track00.kf.txt"):
        cmd_exec(("ffmsindex", "-p", "-f", "-k", filename))
        cleanup((filename + ".ffindex",))

    filename_keyframes = None
    for i in range(3):
        f = filename + ".ffindex_track0{}.kf.txt".format(i)
        if os.path.isfile(f):
            filename_keyframes = f

    try:
        with open(filename_keyframes, "r", encoding="utf-8") as index:
            index.readline()
            index.readline()
            try:
                keyframes = tuple(map(int, index.read().splitlines()))
            except ValueError:
                raise ValueError("Could not determine Keyframes.") from None
    except IOError as e:
        raise IOError(
            "Could not open keyframe file {}.".format(filename_keyframes)
        ) from e

    return keyframes


def make_segments(start, end, keyframes) -> list[Segment]:
    if start in keyframes:
        if end in keyframes or end > keyframes[-1]:
            if end <= start:
                return []
            # copy from keyframe to keyframe
            return [Copy(start, end)]
        # copy to keyframe before end
        lt_kf_before_end = get_keyframe_in_front_of_frame(keyframes, end)
        if lt_kf_before_end is None:
            raise ValueError("No keyframe in front of {}!".format(end))
        if lt_kf_before_end <= start:
            return [Recode(start, end)]
        return [Copy(start, lt_kf_before_end), Recode(lt_kf_before_end, end)]
    nt_kf_from_start = get_keyframe_after_frame(keyframes, start)
    if nt_kf_from_start is None:
        raise ValueError(
            "No keyframe after {} (last keyframe: {})!".format(start, keyframes[-1])
        )
    segments: list[Segment] = [Recode(start, nt_kf_from_start)]
    if end <= nt_kf_from_start:
        return segments
    return segments + make_segments(nt_kf_from_start, end, keyframes)


def get_keyframe_in_front_of_frame(keyframes, frame):
    """Find keyframe less-than to frame."""
    i = bisect.bisect_left(keyframes, frame)
    if i:
        return keyframes[i - 1]
    return None


def get_keyframe_after_frame(keyframes, frame):
    """Find keyframe greater-than to frame."""
    i = bisect.bisect_right(keyframes, frame)
    if i != len(keyframes):
        return keyframes[i]
    return None


def cut_audio(f_in, f_out, cutlist_frm: CutList, fps: float):
    mkvmerge_segments = to_mkvmerge_time_segments(cutlist_frm, fps)
    cmd_exec(
        ("mkvmerge", "-D", "--split", "parts:" + mkvmerge_segments, "-o", f_out, f_in)
    )


def encode(f_in, f_out, slc, encoder_info, x264_profile, workingdir):
    cmd_exec(
        ("x264",)
        + get_x264_opts(encoder_info, x264_profile)
        + (
            "--demuxer",
            "ffms",
            "--index",
            os.path.join(workingdir, "x264.index"),
            "--seek",
            str(slc.start),
            "--frames",
            str(slc.stop - slc.start),
            "--output",
            f_out,
            f_in,
        )
    )


x264_profiles = {
    "HD": (
        "--tune",
        "film",
        "--direct",
        "auto",
        "--force-cfr",
        "--rc-lookahead",
        "60",
        "--b-adapt",
        "2",
        "--weightp",
        "0",
    ),
    "HQ": (
        "--tune",
        "film",
        "--direct",
        "auto",
        "--force-cfr",
        "--rc-lookahead",
        "60",
        "--b-adapt",
        "2",
        "--aq-mode",
        "2",
        "--weightp",
        "0",
    ),
}


def get_x264_opts(encoder_infos, profile):
    cp = encoder_infos["color_primaries"].split()[0].replace("BT.", "bt")
    if cp in {"bt709", "bt470m", "bt470bg", "bt2020"}:
        cp_opts = (
            "--videoformat",
            "pal",
            "--colorprim",
            cp,
            "--transfer",
            cp,
            "--colormatrix",
            cp,
        )
    else:
        cp_opts = ()
    return (
        profile
        + cp_opts
        + (
            "--profile",
            encoder_infos["format_profile_profile"].lower(),
            "--level",
            str(encoder_infos["format_profile_level"]),
            "--fps",
            str(encoder_infos["framerate"]),
        )
    )


class Tgt:
    def __init__(self, name, tp=(lambda x: x)):
        self.name = name
        self.tp = tp


def get_fps(filename: str) -> float:
    try:
        output = cmd_exec(
            (
                "ffprobe",
                filename,
                "-v",
                "0",
                "-select_streams",
                "v",
                "-print_format",
                "flat",
                "-show_entries",
                "stream=r_frame_rate",
            )
        ).stdout
        fps_str = output.strip().split(b"=")[1].strip(b'"')
    except (FileNotFoundError, RuntimeError):
        sys.stderr.write("⚠ Could not determine fps, defaulting to 25\n")
        return 25.0
    if b"/" in fps_str:
        a, b = fps_str.split(b"/")
        return float(a) / float(b)
    return float(fps_str)


def get_encoder_infos(filename):
    try:
        output = cmd_exec(("mediainfo", filename)).stdout.decode()
    except FileNotFoundError:
        return {}
    return parse(
        output,
        {
            "Writing library *: *x264 core ([0-9]*)": Tgt("x264_core"),
            "Color primaries *: (.*[^ ]) *$": Tgt("color_primaries"),
            "Format profile *: ([^@]*)@L([0-9]*) *$": (
                Tgt("format_profile_profile"),
                Tgt("format_profile_level", float),
            ),
            "Frame rate *: ([0-9.]*) FPS$": Tgt("framerate", float),
        },
    )


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
    return {re.compile(pattern): tuplify(key) for pattern, key in mapping.items()}


def process_segments(segments: Sequence[Segment], f_in, f_info, x264_profile, outdir):
    video_files = []
    copy_segments = []
    copy_cnt = count(start=1)
    encode_cnt = count(start=1)
    copy_tpl = os.path.join(outdir, "segment_copy{}.mkv")
    n_copy = sum(1 for seg in segments if isinstance(seg, Copy))
    for seg in segments:
        if isinstance(seg, Copy):
            # If there is only one copy segment, mkvmerge will not add a number suffix:
            video_files.append(
                copy_tpl.format(f"-{next(copy_cnt):03d}" if n_copy > 1 else "")
            )
            copy_segments.append(seg)
        elif isinstance(seg, Recode):
            f_out = os.path.join(outdir, f"segment-{next(encode_cnt):03d}.mkv")
            video_files.append(f_out)
            encode(f_in, f_out, seg, f_info, x264_profile, outdir)
    split_video(
        f_in, copy_tpl.format(""), [(seg.start, seg.stop) for seg in copy_segments]
    )
    return video_files


def split_video(f_in, f_out, cutlist_frm: CutList):
    mkvmerge_segments = to_mkvmerge_frame_segments(cutlist_frm, separator=",")
    cmd_exec(
        (
            "mkvmerge",
            "-A",
            "--split",
            "parts-frames:" + mkvmerge_segments,
            "-o",
            f_out,
            f_in,
        )
    )


def merge_parts(video_files, audio_file, f_out):
    cmd_exec(
        (
            "mkvmerge",
            "--engage",
            "no_cue_duration",
            "--engage",
            "no_cue_relative_position",
            "-o",
            f_out,
            video_files[0],
        )
        + tuple("+" + v for v in video_files[1:])
        + (audio_file,),
        ok_return_codes=(0, 1),
    )


def load_cutlist(stream):
    mapping = {
        r"StartFrame ?= ?(.*)$": Tgt("start", int),
        r"DurationFrames ?= ?(.*)$": Tgt("duration", int),
        r"Start ?= ?(.*)$": Tgt("t_start", float),
        r"Duration ?= ?(.*)$": Tgt("t_duration", float),
        r"ApplyToFile ?= ?(.*)$": Tgt("in_file"),
    }
    out = parse(stream, mapping)

    cutlist_duration = tuple(
        zip(listify(out.pop("start", [])), listify(out.pop("duration", [])))
    )
    out["cutlist_frames"] = tuple(
        (start, start + duration) for start, duration in cutlist_duration
    )
    return out


def load_cutlist_avidemux(stream):
    """Avidemux seems to start at t>0 (depending on reference frames it has to load). Example: 5.2 sec <-> 13 frames @ 25 fps"""
    to_frm = lambda s: int(int(s) * fps // 1000000)
    mapping = {
        r"adm.addSegment\(0, ([0-9]*), ([0-9]*)\);?": (
            Tgt("start", int),
            Tgt("duration", int),
        ),
        r'(?:if not )?adm.loadVideo\("([^"]*)"\);?:?': Tgt("in_file"),
    }
    out = parse(stream, mapping)
    fps = get_fps(out["in_file"])
    out["cutlist_frames"] = tuple(
        (max(0, to_frm(start)), to_frm(start + duration))
        for start, duration in zip(
            listify(out.pop("start", [])), listify(out.pop("duration", []))
        )
    )
    return out


def cutlist_time_from_frames(cutlist_frm: CutList, fps):
    return tuple(
        (to_timestamp(start, fps), to_timestamp(stop, fps))
        for start, stop in cutlist_frm
    )


def to_mkvmerge_time_segments(
    cutlist_frm: CutList, fps: float = 25, separator: str = ",+"
) -> str:
    return separator.join(
        f"{to_timestamp(start, fps)}-{to_timestamp(end, fps)}"
        for start, end in cutlist_frm
    )


def to_mkvmerge_frame_segments(cutlist_frm: CutList, separator: str = ",+") -> str:
    return separator.join(
        (str(start + 1) + "-" + str(stop + 1)) for start, stop in cutlist_frm
    )


def to_timestamp(frm: int, fps: float = 25.0) -> str:
    sec, ms = to_sec_ms(frm, fps)
    mins, sec = divmod(sec, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02}:{mins:02}:{sec:02}.{ms:03}"


def to_seconds(frm: int, fps: float = 25.0) -> str:
    sec, ms = to_sec_ms(frm, fps)
    return f"{sec}.{ms:03}"


def to_sec_ms(frm: int, fps: float) -> tuple[int, int]:
    sec = int(frm // fps)
    ms = int(frm * 1000 / fps) - 1000 * sec
    return (sec, ms)


cutlist_type_catalogue = {
    "//AD  <- Needed to identify //": load_cutlist_avidemux,
    "#PY  <- Needed to identify #": load_cutlist_avidemux,
    "[General]": load_cutlist,
    None: load_cutlist,
}


def determine_cut_list_type(stream):
    line = next(stream).strip("\n")
    return cutlist_type_catalogue.get(line, cutlist_type_catalogue[None])


def tuplify(key):
    if not isinstance(key, tuple):
        return (key,)
    return key


def listify(key):
    if not isinstance(key, list):
        return [key]
    return key


def main(cutlist_path, video_path, out_path, cut_method, offset: int = 0, **kwargs):
    with open(cutlist_path, "r", encoding="utf-8") as stream:
        loader = determine_cut_list_type(stream)
        cutlist_info = loader(stream)
    f_in = video_path or cutlist_info.get("in_file")
    if f_in is None:
        raise ValueError("Input video file not specified!")
    f_out = (
        out_path
        or cutlist_info.get("out_file")
        or cutlist_path.rsplit(".", 1)[0] + ".mkv"
    )
    if os.path.isfile(f_out) and os.path.samefile(f_in, f_out):
        f_out = f_out.rsplit(".", 1)[0] + ".cut.mkv"
    cutlist_frm = [
        (max(0, start + offset), end + offset)
        for start, end in cutlist_info["cutlist_frames"]
    ]
    cut_method(f_in, f_out, cutlist_frm, **kwargs)


def cleanup(files):
    for f in files:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="smartcut")
    parser.add_argument(
        "-o", dest="out", default=None, help="Output file (default: <input file>.cut )"
    )
    parser.add_argument(
        "--workingdir", default=None, help="Working directory (default: temporary)"
    )
    parser.add_argument("cutlist", help="Cutlist")
    parser.add_argument(
        "-i",
        default=None,
        help="Input video file (must only be given if"
        " not specified in the cutlist file)",
    )
    parser.add_argument("--offset", default=0, help="frame offset", type=int)
    modes = {
        "smart": smart_cut_file_by_cutlist,
        "smart:HQ": smart_cut_file_by_cutlist,
        "smart:HD": smart_cut_file_by_cutlist_hd,
        "lossless": cut_file_by_cutlist,
        "reencode": cut_file_by_cutlist_reencode,
    }
    parser.add_argument(
        "--mode",
        default=smart_cut_file_by_cutlist,
        type=modes.__getitem__,
        help=f"Cut mode ({' / '.join(modes)})",
    )
    parser.add_argument(
        "--segmented",
        default=False,
        action="store_true",
        help="Output segments instead of one movie",
    )

    cmd_args = parser.parse_args()

    try:
        main(
            cmd_args.cutlist,
            cmd_args.i,
            cmd_args.out,
            offset=cmd_args.offset,
            workingdir=cmd_args.workingdir,
            cut_method=cmd_args.mode,
            segmented=cmd_args.segmented,
        )
    except subprocess.CalledProcessError as _e:
        sys.stderr.write(_e.stderr + "\n")
        sys.exit(1)
