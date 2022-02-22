import os
import sys
import numpy as np
import h5py

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
#DATA_DIR = os.path.join(BASE_DIR, 'data')
#if not os.path.exists(DATA_DIR):
#    os.mkdir(DATA_DIR)
#if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
#    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#    zipfile = os.path.basename(www)
#    os.system('wget %s; unzip %s' % (www, zipfile))
#    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
#    os.system('rm %s' % (zipfile))

def shuffle_point_cloud(batch_data):
    """ Randomly shuffle points in the point clouds to augument the dataset
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shuffled batch of point clouds
    """
    shuffled_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        idx = np.arange(batch_data.shape[1])
        np.random.shuffle(idx)
        shuffled_data[k, ...] = batch_data[k, idx, ...]
    return shuffled_data


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = (0.1 - 0.2*np.random.uniform()) * np.pi # np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
									[sinval, cosval, 0],
									[0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_15_deg(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = (0.1 - 0.2*np.random.uniform()) * np.pi # np.random.uniform() * 2 * np.pi
        cosval_z = np.cos(rotation_angle)
        sinval_z = np.sin(rotation_angle)
        rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
                                      [sinval_z, cosval_z, 0],
                                      [0, 0, 1]])
        rotation_angle = (0.1 - 0.2*np.random.uniform()) * np.pi # np.random.uniform() * 2 * np.pi
        cosval_y = np.cos(rotation_angle)
        sinval_y = np.sin(rotation_angle)
        rotation_matrix_y = np.array([[cosval_y, 0, sinval_y],
                                      [0, 1, 0],
                                      [-sinval_y, 0, cosval_y]])
        rotation_angle = (0.1 - 0.2*np.random.uniform()) * np.pi # np.random.uniform() * 2 * np.pi
        cosval_x = np.cos(rotation_angle)
        sinval_x = np.sin(rotation_angle)
        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, cosval_x, -sinval_x],
                                      [0, sinval_x, cosval_x]])
        rotation_matrix = np.dot( np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_180_deg(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform(low=-1.0, high=1.0) * np.pi # np.random.uniform() * 2 * np.pi
        cosval_z = np.cos(rotation_angle)
        sinval_z = np.sin(rotation_angle)
        rotation_matrix_z = np.array([[cosval_z, -sinval_z, 0],
									  [sinval_z, cosval_z, 0],
									  [0, 0, 1]])
        rotation_angle = np.random.uniform(low=-1.0, high=1.0) * np.pi # np.random.uniform() * 2 * np.pi
        cosval_y = np.cos(rotation_angle)
        sinval_y = np.sin(rotation_angle)
        rotation_matrix_y = np.array([[cosval_y, 0, sinval_y],
								      [0, 1, 0],
								      [-sinval_y, 0, cosval_y]])
        rotation_angle = np.random.uniform(low=-1.0, high=1.0) * np.pi # np.random.uniform() * 2 * np.pi
        cosval_x = np.cos(rotation_angle)
        sinval_x = np.sin(rotation_angle)
        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, cosval_x, -sinval_x],
                                      [0, sinval_x, cosval_x]])
        rotation_matrix = np.dot( np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data



def jitter_point_cloud_xyz(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, 3), -1*clip, clip)
    #jittered_data += batch_data[:,:,:3]
    batch_data[:,:,:3] += jittered_data
    return batch_data

def jitter_point_cloud_global(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.ones(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def jitter_point_cloud_color(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.ones((B, N, 3)), -1*clip, clip)
    batch_data[:,:,-3:] += jittered_data
    return batch_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    seg = f['cat'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)
    
    
def load_h5_data_label_seg_cat(h5_filename):
    with h5py.File(h5_filename,'r') as f:
        data = np.array(f['data'][:])
        label = np.array(f['label'][:])
        cat = np.array(f['cat'][:])
        feat = np.array(f['feat'][:])
    return (data, label, cat, feat)

#def load_h5_data_label_seg_cat(h5_filename):
#    f = h5py.File(h5_filename,'r')
#    data = f['data'][:]
#    label = f['label'][:]
#    cat = f['cat'][:]
#    feat = f['feat'][:]
#    return (data, label, cat, feat)

def loadDataFile_with_seg_cat(filename):
    return load_h5_data_label_seg_cat(filename)
    
    
    
def load_h5_data_label_sem(h5_filename):
    with h5py.File(h5_filename,'r') as f:
        data = np.array(f['data'][:])
        cat = np.array(f['cat'][:])
        seg = np.array(f['seg'][:])
    return (data, cat, seg)


def loadDataFile_with_sem(filename):
    return load_h5_data_label_sem(filename)    
    
    
    
def load_h5_data_label_seg_cat_mult(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    obj_label = f['obj'][:]
    cat = f['cat'][:]
    feat = f['feat'][:]
    return (data, label, obj_label, cat, feat)


def loadDataFile_with_seg_cat_mult(filename):
    return load_h5_data_label_seg_cat_mult(filename)
    
    
