# bpemb downloads the model on first use. this script forces the download
# to happen during the build step so that each instance of the image
# doesn't have to do it individually
from bpemb import BPEmb
bp = BPEmb(lang='en', vs=10000, dim=100)
