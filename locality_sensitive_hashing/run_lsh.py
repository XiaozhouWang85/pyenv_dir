#!/usr/bin/env

import sys
import numpy as np
import faiss
import os

#################################################################
# Small I/O functions
#################################################################

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def read_file(fname,file_type=None):
	if file_type==None:
		file_type = os.path.splitext(fname)[-1]
	if file_type==".fvecs":
		return fvecs_read(fname)
	if file_type==".ivecs":
		return ivecs_read(fname)

#################################################################
#  Main program
#################################################################

#Training dataset must fit in memory
def train_index(data_vector,dir_path):
    xt = read_file(data_vector)
    index = faiss.index_factory(xt.shape[1], "IVF4096,Flat")
    index.train(xt)
    faiss.write_index(index, dir_path + "trained.index")

#Training dataset must fit in memory
def index_data(data_vector,blocks,dir_path):
    for block_no in range(blocks):
	    xb = read_file(data_vector)
	    i0, i1 = int(block_no * xb.shape[0] / blocks), int((block_no + 1) * xb.shape[0] / blocks)
	    index = faiss.read_index(dir_path + "trained.index")
	    index.add_with_ids(xb[i0:i1], np.arange(i0, i1))
	    print("write " + dir_path + "block_%d.index" % block_no)
	    faiss.write_index(index, dir_path + "block_%d.index" % block_no)

def merge_indices(blocks,dir_path):
	ivfs = []
	for bno in range(blocks):
	    # the IO_FLAG_MMAP is to avoid actually loading the data thus
	    # the total size of the inverted lists can exceed the
	    # available RAM
	    print("read " + dir_path + "block_%d.index" % bno)
	    index = faiss.read_index(dir_path + "block_%d.index" % bno,
	                             faiss.IO_FLAG_MMAP)
	    ivfs.append(index.invlists)

	    # avoid that the invlists get deallocated with the index
	    index.own_invlists = False

	# construct the output index
	index = faiss.read_index(dir_path + "trained.index")

	# prepare the output inverted lists. They will be written
	# to merged_index.ivfdata
	invlists = faiss.OnDiskInvertedLists(
	    index.nlist, index.code_size,
	    dir_path + "merged_index.ivfdata")

	# merge all the inverted lists
	ivf_vector = faiss.InvertedListsPtrVector()
	for ivf in ivfs:
	    ivf_vector.push_back(ivf)

	print("merge %d inverted lists " % ivf_vector.size())
	ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())
	print(ntotal)
	# now replace the inverted lists in the output index
	index.ntotal = ntotal
	index.replace_invlists(invlists)

	print("write " + dir_path + "populated.index")
	faiss.write_index(index, dir_path + "populated.index")

stage = int(sys.argv[1])
TMPDIR = 'tmp/'
INPUT_VECTOR_LEARN = "sift/sift_learn.fvecs"
INPUT_VECTOR_BASE = "sift/sift_base.fvecs"
INPUT_VECTOR_QUERY = "sift/sift_query.fvecs"
INPUT_VECTOR_TRUTH = "sift/sift_groundtruth.ivecs"

BLOCKS = 4

if stage == 0:
    # train the index
    train_index(INPUT_VECTOR_LEARN,TMPDIR)

    # add 1/4 of the database to 4 independent indexes
    index_data(INPUT_VECTOR_BASE,BLOCKS,TMPDIR)

    # merge the images into an on-disk index
    # first load the inverted lists
    merge_indices(BLOCKS,TMPDIR)

if stage == 1:
    # perform a search from disk
    print("read " + TMPDIR + "populated.index")
    index = faiss.read_index(TMPDIR + "populated.index")
    index.nprobe = 16

    # load query vectors and ground-truth
    xq = fvecs_read(INPUT_VECTOR_QUERY)
    gt = ivecs_read(INPUT_VECTOR_TRUTH)

    D, I = index.search(xq[:3], 5)
    print(D)
    print(I)
    #recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(xq.shape[0])
    #print("recall@1: %.3f" % recall_at_1)

if stage == 2:
    # perform a search from disk
    data_list = [INPUT_VECTOR_LEARN,INPUT_VECTOR_BASE,INPUT_VECTOR_QUERY]
    for data_vector in data_list:
	    #xt = fvecs_read(data_vector)
	    a = np.fromfile(data_vector, dtype='int32')
	    print(a.shape)

    #recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(xq.shape[0])
    #print("recall@1: %.3f" % recall_at_1)
