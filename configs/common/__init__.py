"""
Shared config "building blocks". Each file here describes ONE piece of
an experiment (a model, a dataset's loaders, the optimizer, or the
training-loop settings) so experiment configs (see ../MNIST/) can
import and combine them without repeating the description each time.
None of these files is runnable on its own -- they're meant to be
imported.
"""
