"""EchoNet-Dynamic Dataset."""

import collections
import os

import numpy as np
import pandas
import skimage.draw
import torch
import torchvision

import echonet


class Echo(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(
        self,
        root=None,
        split="train",
        target_type="EF",
        mean=0.0,
        std=1.0,
        length=16,
        period=2,
        max_length=250,
        clips=1,
        pad=None,
        noise=None,
        target_transform=None,
        external_test_location=None,
        rep=None,
    ):
        if root is None:
            root = echonet.config.DATA_DIR

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        self.rep = rep

        self.fnames, self.outcome = [], []

        # can use subsets like FileList_1.csv etc.
        if self.split.lower() == "train":
            filelist_name = os.environ.get("FILELIST_NAME", "FileList.csv")
        else:
            filelist_name = "FileList.csv"

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            with open(os.path.join(self.root, filelist_name)) as f:
                data = pandas.read_csv(f)
            if data["Split"].dtype == np.int64:
                offset = int(os.environ.get("S_OFFSET", "0"))
                print(
                    f"Warning: Split is kfold (0-9) instead of train/val/test; converting to train/val/test with offset {offset}."
                )
                sref = ["TRAIN"] * 8 + ["VAL"] + ["TEST"]
                smap = {(i + offset) % 10: sref[i] for i in range(10)}
                data["Split"] = data["Split"].map(smap)
                # {0: "TRAIN", 1: "TRAIN", 2: "TRAIN", 3: "TRAIN", 4: "TRAIN", 5: "TRAIN", 6: "TRAIN", 7: "TRAIN", 8: "VAL", 9: "TEST"})
            data["Split"].map(lambda x: x.upper())
            print(data["Split"].unique())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            print(len(self.fnames))
            self.fnames = [
                fn + ".avi" if os.path.splitext(fn)[1] == "" else fn
                for fn in self.fnames
            ]  # Assume avi if no suffix
            self.outcome = data.values.tolist()

            # Check that files are present
            missing = set(self.fnames) - set(
                os.listdir(os.path.join(self.root, "Videos"))
            )
            if len(missing) != 0:
                print(
                    "{} videos could not be found in {}:".format(
                        len(missing), os.path.join(self.root, "Videos")
                    )
                )
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(
                    os.path.join(self.root, "Videos", sorted(missing)[0])
                )

            # Load traces
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            if os.path.exists(os.path.join(self.root, "VolumeTracings.csv")):
                with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                    header = f.readline().strip().split(",")
                    # assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
                    if header != ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]:
                        header = "BAD_HEADER"
                    if not header == "BAD_HEADER":
                        for line in f:
                            filename, x1, y1, x2, y2, frame = line.strip().split(",")
                            filename = (
                                filename + ".avi"
                                if os.path.splitext(filename)[1] == ""
                                else filename
                            )
                            x1 = float(x1)
                            y1 = float(y1)
                            x2 = float(x2)
                            y2 = float(y2)
                            frame = int(frame)
                            if frame not in self.trace[filename]:
                                self.frames[filename].append(frame)
                            self.trace[filename][frame].append((x1, y1, x2, y2))
            else:
                header = "BAD_HEADER"

            if not header == "BAD_HEADER":
                for filename in self.frames:
                    for frame in self.frames[filename]:
                        self.trace[filename][frame] = np.array(
                            self.trace[filename][frame]
                        )

                # A small number of videos are missing traces; remove these videos
                keep = [len(self.frames[f]) >= 2 for f in self.fnames]
                self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
                self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

            assert len(self.fnames) > 0, "No videos found."

            if self.rep is not None and self.rep > 0:
                rep_factor = int(os.environ.get("REP_FACTOR", 10))
                self.ds_length = len(self.fnames) // rep_factor
                print(
                    f"Found {len(self.fnames)} with rep={self.rep} and {rep_factor}x rep factor."
                )
                print(f"DS length: {self.ds_length}")
                print(f"Last Index: {self.ds_length * self.rep}")
                self.rep_start = int(os.environ.get("REP_START", "0"))
                if self.rep_start > 0:
                    print(f"Starting at index {self.rep_start} for rep={self.rep}")
            else:
                self.ds_length = len(self.fnames)

            self.use_bn_sampling = False
            if filelist_name != "FileList.csv":
                self.use_bn_sampling = True
                self.basenames = list(set(data["BaseName"].tolist()))
                self.ds_length = len(self.basenames)
                self.bn_index = {}
                for i, f in enumerate(self.fnames):
                    bn = f[:-7]
                    if bn not in self.bn_index:
                        self.bn_index[bn] = []
                    self.bn_index[bn].append(i)
                # print(self.bn_index)
                print(f"Found {len(self.basenames)} basenames.")

    def __getitem__(self, index):
        if self.use_bn_sampling:
            index = self.get_bn_index(index)
        elif self.rep is not None:
            index = self.get_rep_idx(index)
        # Find filename of video
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(
                self.root, "ProcessedStrainStudyA4c", self.fnames[index]
            )
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = echonet.utils.loadvideo(video)  # C T H W
        if self.target_transform is not None:
            video = torch.from_numpy(video).permute(1, 0, 2, 3)
            video = self.target_transform(video)
            video = video.permute(1, 0, 2, 3).numpy()
        video = video.astype(np.float32)

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.period == "random":
            period = 1
            # rand_period = (
            #     0.8 + np.random.random(1)[0] * 1.2
            # )  # random period between 0.8 and 1.2
            values = [0, 1, 2, 3]
            weights = [0.85, 0.1, 0.035, 0.015]
            new_fps = 25 + np.random.choice(values, p=weights)
            video_indices = torch.linspace(
                0, f - 1, int(f * new_fps / 32), dtype=torch.long
            )
            video = video[:, video_indices, :, :]
            # print(f"New FPS: {new_fps}, Video frames: {f} -> {video.shape[1]}")
            c, f, h, w = video.shape
        else:
            period = self.period

        if self.length is None:
            # Take as many frames as possible
            length = f // period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate(
                (video, np.zeros((c, length * period - f, h, w), video.dtype)),
                axis=1,
            )
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * period, self.clips)

        # Gather targets
        target = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(
                    np.rint(y).astype(np.int),
                    np.rint(x).astype(np.int),
                    (video.shape[2], video.shape[3]),
                )
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    if not t in self.header:
                        t = "EF"
                    tmp = self.outcome[index][self.header.index(t)]
                    tmp = np.float32(tmp)
                    target.append(tmp)

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            # if self.target_transform is not None:
            #     target = self.target_transform(target)

        # Select clips from video
        video = tuple(video[:, s + period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros(
                (c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype
            )
            temp[:, :, self.pad : -self.pad, self.pad : -self.pad] = (
                video  # pylint: disable=E1130
            )
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i : (i + h), j : (j + w)]

        return video, target

    def __len__(self):
        return self.ds_length

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)

    def get_rep_idx(self, idx):
        offset = np.random.randint(self.rep) + self.rep_start
        return (idx + self.ds_length * offset) % len(self.fnames)

    def get_bn_index(self, idx):
        bn = self.basenames[idx]
        options = self.bn_index[bn]
        return options[np.random.randint(len(options))]


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)
    return collections.defaultdict(list)
