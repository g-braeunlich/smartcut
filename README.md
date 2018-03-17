# smartcut

A command line python script for cutting video files frame accurately.
Reencoding is only performed for segments containing a cut at a
non key frame.

It is refactored out of
https://github.com/monarc99/otr-verwaltung
The tool is tested for files encoded by 
https://www.onlinetvrecorder.com
Other videos have not been tested.

## Dependencies

- python
- mediainfo
- x264_encoder with ffmpegsource support
- ffmpegsource
- mkvtoolnix (mkvmerge)

## Usage
```
python3 smartcut.py [-h] [-o OUT] [--profile HQ|HD] [--workingdir <WORKINGDIR>]
                   [-i <video file>]
                   <cutlist file>
```
Currently supported cutlist formats are:
- [avidemux](http://avidemux.sourceforge.net/) project files (cut in avidemux, File -> Project script ->
  save as project script)
- Cut list format used by [onlinetvrecorder.com](https://www.onlinetvrecorder.com)

If the cutlist specifies the input video file, the ```-i``` option is
not necessary but can be used to overwrite the value from the cutlist file.
