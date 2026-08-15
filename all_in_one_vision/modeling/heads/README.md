# heads/

Placeholder package. In v1 the classification head (a Linear layer) is
small enough to live inline inside each meta_arch. Split it out here once
a head needs to be reused across multiple backbones/meta-archs.
