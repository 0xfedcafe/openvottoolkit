"""Transformer module for experiments."""

from __future__ import annotations

import os
from abc import abstractmethod
import typing

from PIL import Image

from attributee import Attributee, Integer, Float, Boolean, String, List

from vot.dataset import Sequence, InMemorySequence
from vot.dataset.proxy import FrameMapSequence
from vot.dataset.common import write_sequence, read_sequence
from vot.region import Rectangle
from vot.utilities import arg_hash

if typing.TYPE_CHECKING:
    from vot.workspace.storage import FilesystemStorage


class Transformer(Attributee):
    """Base class for transformers.

    Transformers are used to generate new modified sequences from existing ones.
    """

    def __init__(self, cache: FilesystemStorage | None, **kwargs):
        """Initialize the transformer.

        :param cache: The cache to be used for storing generated sequences.
        :type cache: FilesystemStorage
        """
        super().__init__(**kwargs)
        self._cache = cache

    @abstractmethod
    def __call__(self, sequence: Sequence) -> list[Sequence]:
        """Generate a list of sequences from the given sequence. The generated sequences
        are stored in the cache if needed.

        :param sequence: The sequence to be transformed.
        :type sequence: Sequence

        :returns: A list of generated sequences.
        :rtype: [list]"""
        raise NotImplementedError

class SingleObject(Transformer):
    """Transformer that generates a sequence for each object in the given sequence."""

    trim = Boolean(default=False, description="Trim each generated sequence to a visible subsection for the selected object")

    def __call__(self, sequence: Sequence) -> list[Sequence]:
        """Generate a list of sequences from the given sequence.

        :param sequence: The sequence to be transformed.
        :type sequence: Sequence
        """
        from vot.dataset.proxy import ObjectFilterSequence
        
        if len(sequence.objects()) == 1:
            return [sequence]
        
        return [ObjectFilterSequence(sequence, id, bool(self.trim)) for id in sequence.objects()]
        
class Redetection(Transformer):
    """Transformer that test redetection of the object in the sequence. The object is
    shown in several frames and then moved to a different location.

    This tranformer can only be used with single-object sequences.
    """

    length = Integer(default=100, val_min=1)
    initialization = Integer(default=5, val_min=1)
    padding = Float(default=2, val_min=0)
    scaling = Float(default=1, val_min=0.1, val_max=10)

    def __call__(self, sequence: Sequence) -> list[Sequence]:
        """Generate a list of sequences from the given sequence.

        :param sequence: The sequence to be transformed.
        :type sequence: Sequence
        """

        if self._cache is None:
            raise RuntimeError("Local cache is required for redetection transformer.")

        assert len(sequence.objects()) == 1, "Redetection transformer can only be used with single-object sequences."

        cache_dir = self._cache.directory(self, arg_hash(sequence.name, **self.dump()))

        if not os.path.isfile(os.path.join(cache_dir, "sequence")):
            channels_list = list(sequence.channels())
            if not channels_list:
                raise RuntimeError(f"Sequence {sequence.name} has no channels — cannot generate redetection sequence.")

            generated = InMemorySequence(sequence.name, channels_list)
            scaling = typing.cast(float, self.scaling)
            size = (int(sequence.size[0] * scaling), int(sequence.size[1] * scaling))

            initial_images: dict = dict()
            redetect_images: dict = dict()
            # ``template`` is needed *after* the loop to compute the moved offset
            # for the base groundtruth. Initialise to ``None`` and assert non-None
            # after the loop so the type checker sees a definite ``Image`` value.
            template: Image.Image | None = None
            for channel in channels_list:
                frame = sequence.frame(0)
                frame_img = frame.image(channel)

                if frame_img is None:
                    raise RuntimeError(f"Failed to read the first frame of the sequence for channel '{channel}'.")

                gt = frame.groundtruth()
                if gt is None:
                    raise RuntimeError(f"Sequence {sequence.name} is missing groundtruth in the first frame.")
                rect = Rectangle.convert(gt)

                halfsize = int(max(rect.width, rect.height) * scaling / 2)
                x, y = rect.center()

                image = Image.fromarray(frame_img)
                box = (x - halfsize, y - halfsize, x + halfsize, y + halfsize)
                template = image.crop(box)

                initial = Image.new(image.mode, size)
                initial.paste(image, (0, 0))

                redetect = Image.new(image.mode, size)
                redetect.paste(template, (size[0] - template.width, size[1] - template.height))

                initial_images[channel] = initial
                redetect_images[channel] = redetect

            assert template is not None, "Redetection loop did not execute even though channels exist."

            base_gt_region = sequence.frame(0).groundtruth()
            if base_gt_region is None:
                raise RuntimeError(f"Sequence {sequence.name} is missing groundtruth in the first frame.")
            base_gt = Rectangle.convert(base_gt_region)
            generated.append(initial_images, base_gt)
            generated.append(redetect_images, base_gt.move(size[0] - template.width, size[1] - template.height))

            write_sequence(cache_dir, generated)

        source = read_sequence(cache_dir)

        if source is None:
            raise RuntimeError("Failed to read generated sequence from cache.")

        initialization = typing.cast(int, self.initialization)
        mapping = [0] * initialization + [1] * (len(source) - initialization)
        return [FrameMapSequence(source, mapping)]

class IgnoreObjects(Transformer):
    """Transformer that hides objects with certain ids from the sequence."""

    ids = List(String(), default=[], description="List of ids to be ignored")

    def __call__(self, sequence: Sequence) -> list[Sequence]:
        """Generate a list of sequences from the given sequence.

        :param sequence: The sequence to be transformed.
        :type sequence: Sequence
        """
        from vot.dataset.proxy import ObjectsHideFilterSequence
        
        return [ObjectsHideFilterSequence(sequence, set(self.ids))]
    
class Downsample(Transformer):
    """Transformer that downsamples the sequence by a given factor."""

    factor = Integer(default=2, val_min=1, description="Downsampling factor")
    offset = Integer(default=0, val_min=0, description="Offset for the downsampling")

    def __call__(self, sequence: Sequence) -> list[Sequence]:
        """Generate a list of sequences from the given sequence.

        :param sequence: The sequence to be transformed.
        :type sequence: Sequence
        """
        from vot.dataset.proxy import FrameMapSequence
        
        offset = typing.cast(int, self.offset)
        factor = typing.cast(int, self.factor)
        map = [i for i in range(offset, len(sequence), factor)]
        
        return [FrameMapSequence(sequence, map)]
